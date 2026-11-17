#!/usr/bin/env python3
"""D3e: extract a strictly stateless policy from each frozen original C BC actor.

Each step uses the SAME existing original actor.step with h=z=0 and t=0, so the
slow encoder refreshes from current obs and no internal state is carried over.
No training, no model rewrite; previous D3c/D3d controls are reused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from dls_p0 import OBS_DIM, model_info, nparams

ROOT=Path("/mnt/datasets/lxb/dls")
SEEDS=list(range(32000,32024))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-seed",type=int,choices=(42,43,44),required=True)
    args=parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    file=ROOT/"checkpoints/d1-v1"/f"C-seed{args.model_seed}.pt"
    before=hashlib.sha256(file.read_bytes()).hexdigest()
    ckpt=torch.load(file,map_location="cpu",weights_only=False)
    assert ckpt["variant"]=="C"
    policy,_=model_info("C")
    policy.load_state_dict(ckpt["state_dict"],strict=True)
    assert nparams(policy)==ckpt["num_params"]
    policy.eval().requires_grad_(False)
    mean=np.asarray(ckpt["train_mean"],np.float32)
    std=np.asarray(ckpt["train_std"],np.float32)
    assert mean.shape==std.shape==(OBS_DIM,) and float(std.min())>0

    with torch.no_grad():
        inputs=torch.randn((8,OBS_DIM),generator=torch.Generator().manual_seed(3908))
        for t in (0,):
            state=policy.initial_state(8,torch.device("cpu"))
            baseline,_=policy.step(inputs,state,t)
            stateless,_=policy.step(inputs,policy.initial_state(8,torch.device("cpu")),0)
            torch.testing.assert_close(stateless,baseline,rtol=0.0,atol=0.0)

    env=MetaDriveEnv({
        "use_render":False,"image_observation":False,
        "traffic_density":0.10,"map":3,
        "start_seed":SEEDS[0],"num_scenarios":len(SEEDS),
        "horizon":800,"log_level":50,
    })
    episodes=[]
    started=time.monotonic()
    with torch.no_grad():
        for scene in SEEDS:
            obs,_=env.reset(seed=scene)
            reward_total=0.0
            info={}
            for t in range(800):
                raw=np.asarray(obs,np.float32)
                assert raw.shape==(OBS_DIM,) and np.isfinite(raw).all()
                normalized=np.clip((raw-mean)/std,-10.0,10.0).copy()
                # ALWAYS discard h and z, ALWAYS refresh z from current obs.
                state=policy.initial_state(1,torch.device("cpu"))
                action,_=policy.step(torch.from_numpy(normalized).unsqueeze(0),state,0)
                assert action.shape==(1,2) and torch.isfinite(action).all()
                obs,reward,term,trunc,info=env.step(action[0].numpy().astype(np.float32))
                reward_total+=float(reward)
                if term or trunc:
                    break
            success=bool(info.get("arrive_dest",False))
            collision=bool(info.get("crash_vehicle",False) or info.get("crash_object",False))
            offroad=bool(info.get("out_of_road",False))
            episodes.append({
                "seed":scene, "steps":t+1, "success":success,
                "collision":collision, "offroad":offroad,
                "timeout":not(success or collision or offroad),
                "reward":reward_total,
                "route_completion":float(info.get("route_completion",0.0)),
            })
    env.close()
    assert [e["seed"] for e in episodes]==SEEDS
    assert hashlib.sha256(file.read_bytes()).hexdigest()==before
    result={
        "stage":"D3e strictly stateless C policy extracted from unchanged original D1 BC model",
        "bc_seed":args.model_seed,
        "source_checkpoint_sha256":before,
        "scene_seeds":SEEDS,"density":0.10,
        "intervention":"h_t and z_t set to zero before EVERY tick, original actor.step evaluated with t=0 to refresh z; carry NO internal state",
        "same_model_weights":True,
        "new_training_steps":0,
        "success":sum(e["success"] for e in episodes),
        "collision":sum(e["collision"] for e in episodes),
        "offroad":sum(e["offroad"] for e in episodes),
        "mean_raw_reward":float(np.mean([e["reward"] for e in episodes])),
        "mean_route_completion":float(np.mean([e["route_completion"] for e in episodes])),
        "episodes":episodes,
        "wall_seconds":time.monotonic()-started,
        "interpretation_limit":"This extracted stateless map is an OOD model intervention, not an independently optimized MLP. Good performance suggests time-memory may be unnecessary for existing control behavior; bad performance cannot prove task demands memory.",
    }
    file=ROOT/"reports/d3e"/f"stateless-C-seed{args.model_seed}.json"
    file.parent.mkdir(parents=True,exist_ok=True)
    assert not file.exists()
    file.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"D3e C seed{args.model_seed} strictly stateless: "
          f"success={result['success']}/24 mean_reward={result['mean_raw_reward']:.2f} "
          f"offroad={result['offroad']} collision={result['collision']}",flush=True)


if __name__=="__main__":
    main()
