#!/usr/bin/env python3
"""D2c: isolate action exploration noise, no learning, on held-out MetaDrive scenes.

Only sigma of Gaussian noise in the PRE-tanh policy coordinates is changed.
Controller weights, observations, normalization, scenario seeds and dynamics are fixed.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from dls_p0 import model_info, nparams

ROOT = Path("/mnt/datasets/lxb/dls")
SIGMAS = (0.0, 0.05, 0.1, 0.27)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=("A","B","C"))
    p.add_argument("--sigma", type=float, required=True, choices=SIGMAS)
    args = p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    ckpt = torch.load(ROOT/"checkpoints"/"d1-v1"/f"{args.variant}-seed42.pt",
                      map_location="cpu", weights_only=False)
    model, arch = model_info(args.variant)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    assert nparams(model) == ckpt["num_params"]
    model.eval()
    norm_mean = np.asarray(ckpt["train_mean"], dtype=np.float32)
    norm_std = np.asarray(ckpt["train_std"], dtype=np.float32)
    assert norm_mean.shape == norm_std.shape == (259,)
    assert np.min(norm_std) > 0

    env = MetaDriveEnv({
        "use_render": False, "image_observation":False,
        "traffic_density":0.1, "map":3,
        "start_seed":4200, "num_scenarios":24,
        "horizon":800,"log_level":50,
    })
    episodes = []
    start=time.monotonic()
    with torch.no_grad():
        for seed in range(4200,4224):
            observation, _ = env.reset(seed=seed)
            state=model.initial_state(1,torch.device("cpu"))
            rng=torch.Generator(device="cpu").manual_seed(100000+seed)
            steps=0
            rewards=0.0
            deviation_steer=0.0
            deviation_pedal=0.0
            last_info={}
            for t in range(800):
                x=np.asarray(observation,dtype=np.float32)
                assert x.shape==(259,) and np.isfinite(x).all()
                x=torch.from_numpy(np.clip((x-norm_mean)/norm_std,-10,10).copy()).unsqueeze(0)
                mean_action,state=model.step(x,state,t)
                assert mean_action.shape==(1,2) and torch.isfinite(mean_action).all()
                if args.sigma == 0:
                    action=mean_action
                else:
                    assert float(mean_action.abs().max()) < 0.999, "Saturated tanh output"
                    latent=torch.atanh(mean_action)
                    noise=torch.randn((1,2),generator=rng,dtype=latent.dtype)
                    action=torch.tanh(latent+args.sigma*noise)
                deviation_steer += float((action[0,0]-mean_action[0,0]).abs())
                deviation_pedal += float((action[0,1]-mean_action[0,1]).abs())
                observation,reward,terminated,truncated,last_info=env.step(
                    action.squeeze(0).numpy().astype(np.float32)
                )
                steps+=1
                rewards+=float(reward)
                if terminated or truncated:
                    break
            episodes.append({
                "scene_seed":seed,"steps":steps,
                "success":bool(last_info.get("arrive_dest",False)),
                "collision":bool(last_info.get("crash_vehicle",False) or last_info.get("crash_object",False)),
                "out_of_road":bool(last_info.get("out_of_road",False)),
                "route_completion":float(last_info.get("route_completion",0.0)),
                "episode_reward":rewards,
                "mean_abs_steer_change":deviation_steer/steps,
                "mean_abs_pedal_change":deviation_pedal/steps,
            })
    env.close()
    result={
      "stage":"D2c exploration-scale frozen-weight diagnostic (NOT training)",
      "variant":args.variant,"bc_training_seed":42,
      "pre_tanh_sigma":args.sigma,"train_sigma_reference":math.exp(-1.3),
      "train_density":0.1,
      "held_out_scenarios":list(range(4200,4224)),
      "episodes":episodes,
      "total_steps":sum(e["steps"] for e in episodes),
      "success":sum(e["success"] for e in episodes),
      "collisions":sum(e["collision"] for e in episodes),
      "out_of_road":sum(e["out_of_road"] for e in episodes),
      "mean_route_completion":float(np.mean([e["route_completion"] for e in episodes])),
      "mean_abs_steer_change":float(np.mean([e["mean_abs_steer_change"] for e in episodes])),
      "mean_abs_pedal_change":float(np.mean([e["mean_abs_pedal_change"] for e in episodes])),
      "wall_seconds":time.monotonic()-start,
      "reward_mean":float(np.mean([e["episode_reward"] for e in episodes])),
      "episode_seed_noise_rng":"100000 + scene_seed, same across A/B/C",
      "checkpoint_used":str(ROOT/"checkpoints"/"d1-v1"/f"{args.variant}-seed42.pt"),
    }
    output=ROOT/"reports"/"d2c"/f"noise-{args.variant}-sigma{args.sigma:.2f}.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists(), "No overwrite of prior D2c diagnostics"
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="episodes"},indent=2),flush=True)

if __name__=="__main__":
    main()
