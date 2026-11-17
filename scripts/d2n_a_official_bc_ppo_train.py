#!/usr/bin/env python3
"""D2n-A: 64-CPU-worker BC-initialized original DLS A/B/C trained with official PPO.

One variant per process, run sequentially on the same server. Does not replace
the A/B/C recurrent controller or its direct module communication. Only standard
SB3-Contrib 2.6.0 RecurrentPPO performs updates, including official GAE and
recurrent rollout-buffer masking.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from dls_p0 import model_info, nparams
from dls_sb3_recurrent_policy import DLSBCInitializedRecurrentPolicy


ROOT = Path("/mnt/datasets/lxb/dls")
N_ENVS = 64
N_STEPS = 128
TRAIN_FRAMES = 65_536
TRAIN_WORKER_SEED_BASE = 24_000


class SeedRangeMetaDriveEnv(MetaDriveEnv):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            start,n = int(self.start_index),int(self.num_scenarios)
            if not start <= int(seed) < start+n:
                seed=start+int(seed)%n
        assert options is None or options == {}
        return super().reset(seed=seed)


def make_env(worker_index):
    return SeedRangeMetaDriveEnv({
        "use_render": False, "image_observation": False,
        "traffic_density": 0.1, "map": 3,
        "start_seed": TRAIN_WORKER_SEED_BASE + worker_index * 80,
        "num_scenarios": 80, "horizon": 800, "log_level": 50,
    })


class OfficialPPOAudit(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.rollout_count = 0
        self.max_ratio_error = 0.0
        self.completed_episodes = 0
        self.successes = 0
        self.rollout_elapsed = []
        self.start_time = None

    def _on_rollout_start(self):
        self.start_time = time.perf_counter()

    def _on_step(self):
        dones = np.asarray(self.locals["dones"],dtype=bool)
        infos = self.locals["infos"]
        self.completed_episodes += int(dones.sum())
        self.successes += sum(bool(i.get("arrive_dest", False)) for done,i in zip(dones,infos) if done)
        return True

    def _on_rollout_end(self):
        self.rollout_elapsed.append(time.perf_counter()-self.start_time)
        self.model.policy.set_training_mode(False)
        # Official buffer is padded per recurring sequence; mask out padding.
        with torch.no_grad():
            batch = next(self.model.rollout_buffer.get(batch_size=512))
            values,new_lp,entropy=self.model.policy.evaluate_actions(
                batch.observations,batch.actions,
                batch.lstm_states,batch.episode_starts)
            mask=batch.mask>1e-8
            assert mask.sum()>0
            ratio = torch.exp(new_lp[mask]-batch.old_log_prob[mask])
            assert torch.isfinite(ratio).all()
            error=float((ratio-1).abs().max().item())
        assert error<2e-3, f"Old-policy recurrent replay mismatch {error}"
        self.max_ratio_error=max(self.max_ratio_error,error)
        self.rollout_count+=1
        print(f"ROLLOUT {self.rollout_count}/8 samples={self.model.rollout_buffer.buffer_size * self.model.n_envs} "
              f"replay_ratio_error={error:.2e} "
              f"completed={self.completed_episodes} successes={self.successes}",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("A","B","C"),required=True)
    args=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    import sb3_contrib, stable_baselines3, gymnasium
    assert sb3_contrib.__version__==stable_baselines3.__version__=="2.6.0"
    assert gymnasium.__version__=="1.1.1"
    assert torch.__version__.startswith("2.5.1")
    assert psutil.cpu_count(logical=True)>=64
    assert psutil.virtual_memory().available>70*1024**3
    torch.set_num_threads(2)

    variant=args.variant
    source=ROOT/"checkpoints/d1-v1"/f"{variant}-seed42.pt"
    sha_before=hashlib.sha256(source.read_bytes()).hexdigest()
    dest=ROOT/"checkpoints/d2n-a"/f"official-PPO-{variant}-seed42.zip"
    stats_file=ROOT/"checkpoints/d2n-a"/f"vecnormalize-{variant}-seed42.pkl"
    report_file=ROOT/"reports/d2n-a"/f"train-{variant}-seed42.json"
    for pth in (dest,stats_file,report_file):
        assert not pth.exists(), f"Refusing to overwrite existing experiment: {pth}"
        pth.parent.mkdir(parents=True,exist_ok=True)
    logs=ROOT/"tensorboard/d2n-a"/f"{variant}-seed42"
    logs.mkdir(parents=True,exist_ok=True)

    initial_actor,_=model_info(variant)
    ckpt=torch.load(source,weights_only=False,map_location="cpu")
    initial_actor.load_state_dict(ckpt["state_dict"],strict=True)
    assert nparams(initial_actor)==ckpt["num_params"]

    init_start=time.perf_counter()
    vec=SubprocVecEnv([lambda idx=i:make_env(idx) for i in range(N_ENVS)],
                      start_method="spawn")
    vec=VecNormalize(VecMonitor(vec),
                     norm_obs=False,norm_reward=True,
                     clip_reward=10.0,gamma=0.99,training=True)
    assert vec.num_envs==N_ENVS
    assert not vec.norm_obs and vec.norm_reward
    model=RecurrentPPO(
        DLSBCInitializedRecurrentPolicy,
        vec,
        n_steps=N_STEPS,batch_size=512,n_epochs=4,
        gamma=0.99,gae_lambda=0.95,
        learning_rate=5e-5,
        clip_range=0.2,
        ent_coef=0.0,vf_coef=0.5,
        max_grad_norm=0.5,
        seed=811,device="cpu",verbose=0,
        tensorboard_log=str(logs),
        policy_kwargs={
            "variant":variant,
            "bc_checkpoint_path":str(source),
        },
    )
    assert model.n_envs==N_ENVS and model.n_steps==N_STEPS
    assert model.policy.actor is not None
    assert model.policy.variant==variant
    assert model.num_timesteps==0

    with torch.no_grad():
        actor_initial={k:v.detach().clone() for k,v in model.policy.actor.state_dict().items()}
        critic_initial={k:v.detach().clone() for k,v in model.policy.critic.state_dict().items()}
    init_seconds=time.perf_counter()-init_start
    audit=OfficialPPOAudit()
    train_start=time.perf_counter()
    model.learn(total_timesteps=TRAIN_FRAMES,callback=audit,log_interval=4)
    train_seconds=time.perf_counter()-train_start
    assert model.num_timesteps==TRAIN_FRAMES
    assert audit.rollout_count==TRAIN_FRAMES // (N_ENVS*N_STEPS)==8
    assert audit.max_ratio_error < 2e-3

    delta_actor=sum(float((v-actor_initial[k]).abs().sum())
                    for k,v in model.policy.actor.state_dict().items())
    delta_critic=sum(float((v-critic_initial[k]).abs().sum())
                     for k,v in model.policy.critic.state_dict().items())
    assert delta_actor>0 and delta_critic>0
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())

    model.save(dest)
    vec.save(str(stats_file))
    result={
        "stage":"D2n-A first scientifically controlled original A/B/C BC-initialized official RecurrentPPO",
        "variant":variant,"bc_seed":42,"train_algorithm_seed":811,
        "initial_bc_checkpoint":str(source),
        "initial_bc_checkpoint_sha256":sha_before,
        "train_frames":model.num_timesteps,
        "num_workers":N_ENVS,
        "n_steps_per_worker":N_STEPS,
        "n_rollouts":audit.rollout_count,
        "ppo_minibatch":512,"ppo_epochs":4,
        "learning_rate":5e-5,"clip":0.2,"gamma":0.99,"GAE_lambda":0.95,
        "training_density":0.1,
        "training_seed_ranges":[[TRAIN_WORKER_SEED_BASE+i*80,TRAIN_WORKER_SEED_BASE+i*80+79]
                                for i in range(N_ENVS)],
        "vecnormalize_observations":False,
        "vecnormalize_rewards":True,
        "actor_normalization":"frozen D1 train_mean/train_std with clip [-10,10]",
        "initial_pre_tanh_sigma":0.05,
        "final_pre_tanh_sigma":model.policy.log_std.exp().detach().cpu().tolist(),
        "actor_params":nparams(model.policy.actor),
        "critic_params":nparams(model.policy.critic),
        "buffer_state_size":model.policy.state_carrier_size,
        "replayed_probability_ratio_max_error":audit.max_ratio_error,
        "completed_train_episodes":audit.completed_episodes,
        "successful_train_episodes":audit.successes,
        "actor_parameter_L1_change":delta_actor,
        "critic_parameter_L1_change":delta_critic,
        "setup_seconds":init_seconds,
        "train_seconds":train_seconds,
        "end_to_end_training_fps":TRAIN_FRAMES/train_seconds,
        "rollout_collection_seconds":audit.rollout_elapsed,
        "official_model_checkpoint":str(dest),
        "official_vecnormalize":str(stats_file),
        "original_model_unchanged":True,
        "interpretation":"First single-init seed official PPO research pilot; full 3-init and held-out frozen communication interventions are separate later steps.",
    }
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha_before
    report_file.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    vec.close()
    print(f"PASS D2n-A train {variant}: {model.num_timesteps} steps / {audit.rollout_count} "
          f"rollouts, {train_seconds:.1f}s, completed={audit.completed_episodes}, "
          f"successes={audit.successes}, ratio_err={audit.max_ratio_error:.2e}",flush=True)


if __name__=="__main__":
    main()
