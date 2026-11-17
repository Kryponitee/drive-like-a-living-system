#!/usr/bin/env python3
"""D2m+: official SB3-Contrib RecurrentPPO policy bridge correctness + real MetaDrive update.

1. Instantiate each original D1 BC A/B/C as an official recurrent policy.
2. Compare deterministic actions against the unchanged original BC actor.
3. Recompute old log probabilities on official padded recurrent rollout batches.
4. One short official SB3-Contrib PPO learning job with 4 MetaDrive processes.
No long experiment, no new BC samples, no weights or env dependencies overwritten.
"""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from gymnasium import spaces
from metadrive.envs.metadrive_env import MetaDriveEnv
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from dls_p0 import model_info, nparams
from dls_sb3_recurrent_policy import DLSBCInitializedRecurrentPolicy

ROOT = Path("/mnt/datasets/lxb/dls")
OUT = ROOT / "reports/d2m-official-policy"


class SeedRangeMetaDriveEnv(MetaDriveEnv):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            start = int(self.start_index)
            n = int(self.num_scenarios)
            if not start <= int(seed) < start+n:
                seed=start+int(seed)%n
        assert options is None or options == {}
        return super().reset(seed=seed)


def make_env(i):
    return SeedRangeMetaDriveEnv({
        "use_render":False,"image_observation":False,
        "traffic_density":0.1,"map":3,
        "start_seed":22000+i*80,
        "num_scenarios":80,
        "horizon":800,"log_level":50,
    })


class RatioAudit(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.checked_rollouts=0
        self.max_probability_ratio_error=0.0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.model.policy.set_training_mode(False)
        n_real=0
        with torch.no_grad():
            for batch in self.model.rollout_buffer.get(batch_size=64):
                values,replayed,entropy=self.model.policy.evaluate_actions(
                    batch.observations,
                    batch.actions,
                    batch.lstm_states,
                    batch.episode_starts,
                )
                mask=batch.mask>1e-8
                assert int(mask.sum())>0
                ratio=torch.exp(replayed[mask]-batch.old_log_prob[mask])
                assert torch.isfinite(ratio).all()
                error=float((ratio-1).abs().max())
                self.max_probability_ratio_error=max(
                    self.max_probability_ratio_error,error)
                n_real+=int(mask.sum())
        assert n_real==self.model.n_steps*self.model.n_envs
        assert self.max_probability_ratio_error<2e-3, (
            "Official recurrent buffer pre-update logprob replay mismatch",
            self.max_probability_ratio_error,
        )
        self.checked_rollouts+=1
        print(f"Official recurrent PPO rollout ratio audit {self.checked_rollouts}: "
              f"max_abs_error={self.max_probability_ratio_error:.2e}",flush=True)


def cold_policy_checks(variant):
    checkpoint=ROOT/"checkpoints/d1-v1"/f"{variant}-seed42.pt"
    policy=DLSBCInitializedRecurrentPolicy(
        spaces.Box(-np.inf,np.inf,shape=(259,),dtype=np.float32),
        spaces.Box(-1.0,1.0,shape=(2,),dtype=np.float32),
        lr_schedule=lambda _: 1e-4,
        variant=variant,
        bc_checkpoint_path=str(checkpoint),
    )
    assert policy.actor is not None and policy.critic is not None
    assert policy.state_carrier_size==policy.vector_adapter.flat_state_size()
    assert policy.lstm_actor.hidden_size==policy.state_carrier_size

    n=7
    size=policy.state_carrier_size
    state=torch.zeros((1,n,size),dtype=torch.float32)
    states=RNNStates((state.clone(),state.clone()),(state.clone(),state.clone()))
    starts=torch.ones(n,dtype=torch.float32)
    obs=torch.randn((n,259),generator=torch.Generator().manual_seed(321))
    policy.eval()
    with torch.no_grad():
        original_mean,_=policy.actor.step(
            policy._normalized(obs),
            policy.actor.initial_state(n,torch.device("cpu")),0)
        distribution,next_state=policy.get_distribution(obs,states.pi,starts)
        deterministic=distribution.mode()
        torch.testing.assert_close(deterministic, original_mean,atol=3e-6,rtol=3e-6)
        actions,values,old_lp,next_states=policy.forward(
            obs,states,starts,deterministic=False)
        replay_values,replay_lp,entropy=policy.evaluate_actions(
            obs,actions,states,starts)
        assert torch.isfinite(old_lp).all()
        assert old_lp.shape==(n,) and values.shape==(n,1)
        assert float((old_lp-replay_lp).abs().max())<2e-3
        torch.testing.assert_close(values,replay_values,atol=1e-6,rtol=1e-6)
        assert next_states.pi[0].shape==(1,n,size)

        # Padded sequences as official recurrent rollout buffer sends to
        # evaluate_actions. Episode resets can occur at different times.
        n_seq=4
        length=11
        zeros=torch.zeros((1,n_seq,size))
        padded_states=RNNStates((zeros.clone(),zeros.clone()),(zeros.clone(),zeros.clone()))
        seq_obs=torch.randn((n_seq*length,259),generator=torch.Generator().manual_seed(234))
        seq_starts=torch.zeros((n_seq,length))
        seq_starts[:,0]=1
        seq_starts[1,4]=1
        seq_starts[2,9]=1
        seq_starts=seq_starts.flatten()
        _,seq_next=policy.get_distribution(seq_obs,padded_states.pi,seq_starts)
        assert seq_next[0].shape==(1,n_seq,size)
        assert torch.isfinite(seq_next[0]).all()
        assert torch.isfinite(policy.predict_values(seq_obs, padded_states.vf,seq_starts)).all()

    return {
        "variant":variant,
        "bc_trainable_params":nparams(policy.actor),
        "critic_trainable_params":nparams(policy.critic),
        "state_size":size,
        "original_BC_deterministic_action_match":True,
        "logprob_ratio_before_any_update":float((old_lp-replay_lp).abs().max()),
        "padded_sequence_reset_test":True,
        "unused_legacy_policy_trainable_params":sum(
            p.numel() for name,p in policy.named_parameters()
            if p.requires_grad and not (
                name.startswith("actor.") or name.startswith("critic.") or name=="log_std"
            )
        ),
    }


def main():
    assert os.path.ismount("/mnt/datasets")
    import stable_baselines3, sb3_contrib, gymnasium
    assert stable_baselines3.__version__==sb3_contrib.__version__=="2.6.0"
    assert gymnasium.__version__=="1.1.1"
    assert torch.__version__.startswith("2.5.1")
    torch.set_num_threads(2)
    torch.manual_seed(193)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"official-policy-bridge-smoke.json"
    assert not path.exists()
    before={}
    for v in "ABC":
        source=ROOT/"checkpoints/d1-v1"/f"{v}-seed42.pt"
        before[v]=hashlib.sha256(source.read_bytes()).hexdigest()
    reports={v:cold_policy_checks(v) for v in "ABC"}
    assert reports["B"]["bc_trainable_params"]==reports["C"]["bc_trainable_params"]
    assert reports["B"]["critic_trainable_params"]==reports["C"]["critic_trainable_params"]
    assert all(r["unused_legacy_policy_trainable_params"]==0 for r in reports.values())

    vec=SubprocVecEnv([lambda i=i:make_env(i) for i in range(4)],start_method="spawn")
    vec=VecNormalize(VecMonitor(vec),norm_obs=False,norm_reward=True,
                     clip_reward=10.0,gamma=0.99,training=True)
    assert not vec.norm_obs and vec.norm_reward
    model=RecurrentPPO(
        DLSBCInitializedRecurrentPolicy,vec,
        n_steps=32,batch_size=64,n_epochs=2,
        gamma=0.99,gae_lambda=0.95,
        learning_rate=1e-4,clip_range=0.2,
        ent_coef=0.0,vf_coef=0.5,max_grad_norm=0.5,
        seed=193,device="cpu",verbose=0,
        policy_kwargs={
            "variant":"B",
            "bc_checkpoint_path":str(ROOT/"checkpoints/d1-v1/B-seed42.pt"),
        },
    )
    assert model.n_envs==4
    actor_before={k:p.detach().clone() for k,p in model.policy.actor.state_dict().items()}
    critic_before={k:p.detach().clone() for k,p in model.policy.critic.state_dict().items()}
    cb=RatioAudit()
    started=time.monotonic()
    model.learn(total_timesteps=512,callback=cb,log_interval=4)
    seconds=time.monotonic()-started
    assert model.num_timesteps==512
    assert cb.checked_rollouts==4
    actor_change=sum(float((p-actor_before[k]).abs().sum()) for k,p in model.policy.actor.state_dict().items())
    critic_change=sum(float((p-critic_before[k]).abs().sum()) for k,p in model.policy.critic.state_dict().items())
    assert actor_change>0 and critic_change>0
    assert cb.max_probability_ratio_error<2e-3
    vec.close()

    after={}
    for v in "ABC":
        source=ROOT/"checkpoints/d1-v1"/f"{v}-seed42.pt"
        after[v]=hashlib.sha256(source.read_bytes()).hexdigest()
    assert before==after,"Unexpected original BC checkpoint change"

    report={
        "stage":"D2m+ official SB3-Contrib RecurrentPPO native A/B/C custom GRU actor policy bridge",
        "custom_policy_subclass":"RecurrentActorCriticPolicy",
        "original_BC_models_preserved":True,
        "official_ppo_rollout_buffer_and_loss_used":True,
        "actor_memory_codec":"DLS h + z + async clock phase as flattened recurrent buffer state",
        "baseline_policy_tests":reports,
        "real_MetaDrive_smoke_model_variant":"B",
        "real_MetaDrive_workers":4,
        "real_environment_frames":512,
        "ppo_rollouts_checked":cb.checked_rollouts,
        "max_old_policy_replay_ratio_error":cb.max_probability_ratio_error,
        "actor_update_L1":actor_change,
        "critic_update_L1":critic_change,
        "wall_seconds":seconds,
        "source_BC_checkpoint_sha256_unchanged":True,
        "source_venv":"dls-sb3-compat",
        "scientific_limits":"Pure implementation smoke. No driving success inference; A/C did not yet receive official PPO training, and no controlled 64-worker A/B/C equal-budget study yet.",
    }
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"PASS D2m+ official 512-frame B PPO: {cb.checked_rollouts} "
          f"rollouts; max_ratio_err={cb.max_probability_ratio_error:.2e}, "
          f"actor_grad_update={actor_change:.3f}, critic_update={critic_change:.3f}",flush=True)


if __name__=="__main__":
    main()
