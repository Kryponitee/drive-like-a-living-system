#!/usr/bin/env python3
"""D3a2: frozen C direct cross-module messages replaced by norm-matched own-module content."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv

from dls_p0 import (
    ACTION_DIM, FEATURE_DIM, MODULE_HIDDEN, MODULES, OBS_DIM,
    SLOW_DIM, SLOW_PERIOD, ModularController, model_info, nparams,
)

ROOT = Path("/mnt/datasets/lxb/dls")
SCENES = list(range(31000,31024))
DENSITIES = (0.10,0.25)


class CrossToSelfIntervention(ModularController):
    """Same parent code; optionally replace cross-message content with own-module projection."""

    def __init__(self, variant):
        assert variant == "C"
        super().__init__(variant)
        self.substitute_self = False
        self.message_ticks = 0
        self.message_original_squared_energy = 0.0
        self.message_energy_difference_max = 0.0
        self.message_recipient_difference_mean_sum = 0.0
        self.messages_all_equal_count = 0

    def step(self, obs, state, t):
        h, z = state
        x, z = self.encode(obs, z, t)
        q = self.query(h)
        k = self.key(h)
        v = torch.einsum("bkd,kdf->bkf",h,self.value_weights)
        if self.mode == "C":
            scores=torch.bmm(q,k.transpose(1,2))/math.sqrt(q.shape[-1])
            diag=torch.eye(MODULES,device=scores.device,dtype=torch.bool).unsqueeze(0)
            weights=scores.masked_fill(diag,float("-inf")).softmax(dim=-1)
            aux=torch.bmm(weights,v)
        else:
            gate=torch.sigmoid((q*k).sum(dim=-1,keepdim=True)/math.sqrt(q.shape[-1]))
            aux=v*gate

        assert aux.shape==(len(obs),MODULES,MODULE_HIDDEN)
        # Eliminate external h_j from aux_i using own-module v_i while preserving
        # its pre-intervention L2 norm for each recipient i at this tick.
        own_norm = torch.linalg.vector_norm(v,dim=-1,keepdim=True)
        incoming_norm = torch.linalg.vector_norm(aux,dim=-1,keepdim=True)
        matched_self = v * (incoming_norm / own_norm.clamp_min(1e-12))
        intervened = matched_self if self.substitute_self else aux
        assert torch.isfinite(intervened).all()
        with torch.no_grad():
            before=float(aux.square().sum())
            after=float(intervened.square().sum())
            self.message_ticks+=1
            self.message_original_squared_energy+=before
            self.message_energy_difference_max=max(
                self.message_energy_difference_max,abs(before-after)
            )
            mean_change=float((aux-intervened).abs().mean())
            self.message_recipient_difference_mean_sum+=mean_change
            self.messages_all_equal_count+=int(torch.allclose(aux,torch.roll(aux,1,1),atol=1e-8,rtol=1e-6))
        assert torch.isfinite(aux).all()
        assert self.message_energy_difference_max < 1e-3
        h_next=torch.stack([
            self.cells[i](torch.cat([x,z,intervened[:,i]],dim=-1),h[:,i])
            for i in range(MODULES)
        ],dim=1)
        routing_logits=self.router(torch.cat([x,z,h_next.flatten(1)],dim=-1))
        alpha=routing_logits.softmax(dim=-1)
        projected=torch.einsum("bkd,kdf->bkf",h_next,self.output_weights)
        pooled=(projected*alpha.unsqueeze(-1)).sum(dim=1)
        action=self.action_head(pooled).tanh()
        return action,(h_next,z)


def check_parity(variant, ckpt):
    original,_=model_info(variant)
    replica=CrossToSelfIntervention(variant)
    original.load_state_dict(ckpt["state_dict"],strict=True)
    replica.load_state_dict(ckpt["state_dict"],strict=True)
    original.eval(); replica.eval()
    state_a=original.initial_state(5,torch.device("cpu"))
    state_b=replica.initial_state(5,torch.device("cpu"))
    gen=torch.Generator().manual_seed(4301)
    max_error=0.0
    with torch.no_grad():
        for t in range(32):
            obs=torch.randn((5,OBS_DIM),generator=gen)
            aa,state_a=original.step(obs,state_a,t)
            bb,state_b=replica.step(obs,state_b,t)
            errors=[
                float((aa-bb).abs().max()),
                float((state_a[0]-state_b[0]).abs().max()),
                float((state_a[1]-state_b[1]).abs().max())
            ]
            max_error=max(max_error,*errors)
            assert max(errors)<2e-6,(variant,t,errors)
    assert nparams(original)==nparams(replica)==ckpt["num_params"]
    return max_error


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("C",),required=True)
    p.add_argument("--mode",choices=("self",),required=True)
    p.add_argument("--density",type=float,choices=DENSITIES,required=True)
    args=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    ckpt_path=ROOT/"checkpoints/d1-v1"/f"{args.variant}-seed42.pt"
    checksum=hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=False)
    assert ckpt["variant"]==args.variant
    error=check_parity(args.variant,ckpt)
    policy=CrossToSelfIntervention(args.variant)
    policy.substitute_self = True
    policy.load_state_dict(ckpt["state_dict"],strict=True)
    policy.eval().requires_grad_(False)
    mean=np.asarray(ckpt["train_mean"],dtype=np.float32)
    std=np.asarray(ckpt["train_std"],dtype=np.float32)
    assert mean.shape==std.shape==(OBS_DIM,) and std.min()>0
    env=MetaDriveEnv({
        "use_render":False,"image_observation":False,
        "traffic_density":args.density,"map":3,
        "start_seed":SCENES[0],"num_scenarios":len(SCENES),
        "horizon":800,"log_level":50,
    })
    episodes=[]
    started=time.monotonic()
    with torch.no_grad():
        for scene_seed in SCENES:
            obs,_=env.reset(seed=scene_seed)
            state=policy.initial_state(1,torch.device("cpu"))
            reward_total=0.0
            info={}
            for t in range(800):
                raw=np.asarray(obs,dtype=np.float32)
                assert raw.shape==(OBS_DIM,) and np.isfinite(raw).all()
                norm=np.clip((raw-mean)/std,-10.0,10.0).copy()
                act,state=policy.step(torch.from_numpy(norm).unsqueeze(0),state,t)
                assert act.shape==(1,ACTION_DIM) and torch.isfinite(act).all()
                obs,reward,terminated,truncated,info=env.step(
                    act[0].numpy().astype(np.float32)
                )
                reward_total+=float(reward)
                if terminated or truncated:
                    break
            success=bool(info.get("arrive_dest",False))
            collision=bool(info.get("crash_vehicle",False) or info.get("crash_object",False))
            offroad=bool(info.get("out_of_road",False))
            episodes.append({
                "seed":scene_seed,"steps":t+1,"success":success,
                "collision":collision,"offroad":offroad,
                "timeout":not (success or collision or offroad),
                "reward":reward_total,
                "route_completion":float(info.get("route_completion",0.0)),
            })
    env.close()
    assert len(episodes)==len(SCENES)
    assert policy.message_ticks==sum(ep["steps"] for ep in episodes)
    assert policy.message_original_squared_energy>0
    if args.mode=="self":
        assert policy.message_recipient_difference_mean_sum>0
        assert policy.messages_all_equal_count<policy.message_ticks
    assert hashlib.sha256(ckpt_path.read_bytes()).hexdigest()==checksum
    result={
        "stage":"D3a2 C-only frozen BC cross-module information substitute self at matched message norm",
        "variant":args.variant,"condition":args.mode,
        "bc_seed":42,"source_checkpoint_sha256":checksum,
        "max_original_step_parity_error":error,
        "intervention":"each C aux_i becomes own v_i normalized to the original incoming aux_i L2 norm, eliminating direct cross-module input at every step",
        "cross_message_replaced_and_per_recipient_L2_matched":True,
        "max_message_L2_squared_energy_difference":policy.message_energy_difference_max,
        "mean_original_message_squared_energy_per_tick":policy.message_original_squared_energy/policy.message_ticks,
        "mean_message_recipient_change_abs_per_tick":policy.message_recipient_difference_mean_sum/policy.message_ticks,
        "original_message_all_identical_ticks":policy.messages_all_equal_count,
        "traffic_density":args.density,"eval_seeds":SCENES,
        "model_training_updates":0,
        "success":sum(e["success"] for e in episodes),
        "collision":sum(e["collision"] for e in episodes),
        "offroad":sum(e["offroad"] for e in episodes),
        "mean_raw_reward":float(np.mean([e["reward"] for e in episodes])),
        "mean_route_completion":float(np.mean([e["route_completion"] for e in episodes])),
        "episodes":episodes,
        "wall_seconds":time.monotonic()-started,
        "interpretation_limit":"This is a sustained intervention replacing cross-message content with own state at matched norm; a large loss is sensitivity evidence, not proof of beneficial intermodule coordination; substitution can be off-distribution.",
    }
    out=ROOT/"reports/d3a2"/f"eval-{args.variant}-{args.mode}-density{args.density:.2f}.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"D3a2 {args.variant} {args.mode} density={args.density:.2f} "
          f"success={result['success']}/24 mean_reward={result['mean_raw_reward']:.2f} "
          f"message_energy_delta={result['max_message_L2_squared_energy_difference']:.2e} "
          f"parity_error={error:.2e}",flush=True)


if __name__=="__main__":
    main()
