#!/usr/bin/env python3
"""D2n-A paired BC-before vs official PPO-after closed-loop evaluation.

Same unseen scenario seeds for A/B/C, three densities, 24 episodes per cell.
No PPO updates, no data collection for training, no changed model weights.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from sb3_contrib import RecurrentPPO
from dls_p0 import model_info

ROOT=Path("/mnt/datasets/lxb/dls")
SEEDS=list(range(30000,30024))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--variant",choices=("A","B","C"),required=True)
    parser.add_argument("--phase",choices=("bc","ppo"),required=True)
    parser.add_argument("--density",type=float,choices=(0.0,0.1,0.25),required=True)
    args=parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    bc=torch.load(ROOT/"checkpoints/d1-v1"/f"{args.variant}-seed42.pt",map_location="cpu",weights_only=False)
    assert bc["variant"]==args.variant
    if args.phase=="bc":
        actor,_=model_info(args.variant)
        actor.load_state_dict(bc["state_dict"],strict=True)
        actor.eval()
        norm_mean=np.asarray(bc["train_mean"],np.float32)
        norm_std=np.asarray(bc["train_std"],np.float32)
        assert norm_mean.shape==norm_std.shape==(259,) and norm_std.min()>0
        model=None
    else:
        checkpoint=ROOT/"checkpoints/d2n-a"/f"official-PPO-{args.variant}-seed42.zip"
        assert checkpoint.is_file()
        model=RecurrentPPO.load(checkpoint,device="cpu")
        assert model.policy.variant==args.variant
        assert model.num_timesteps==65536
        actor=None

    env=MetaDriveEnv({
        "use_render":False,"image_observation":False,
        "traffic_density":args.density,"map":3,
        "start_seed":SEEDS[0],"num_scenarios":len(SEEDS),
        "horizon":800,"log_level":50,
    })
    episodes=[]
    started=time.monotonic()
    with torch.no_grad():
        for seed in SEEDS:
            obs,_=env.reset(seed=seed)
            if actor is not None:
                memory=actor.initial_state(1,torch.device("cpu"))
            else:
                memory=None
            eps=np.ones((1,),dtype=bool)
            raw_total=0.0
            info={}
            for t in range(800):
                raw=np.asarray(obs,dtype=np.float32)
                assert raw.shape==(259,) and np.isfinite(raw).all()
                if actor is not None:
                    x=np.clip((raw-norm_mean)/norm_std,-10.0,10.0)
                    action_tensor,memory=actor.step(torch.from_numpy(x).unsqueeze(0),memory,t)
                    action=action_tensor[0].numpy().astype(np.float32)
                else:
                    action_tensor,memory=model.predict(raw.reshape(1,-1),
                        state=memory,episode_start=eps,deterministic=True)
                    action=np.asarray(action_tensor[0],dtype=np.float32)
                assert action.shape==(2,) and np.isfinite(action).all()
                obs,reward,terminated,truncated,info=env.step(np.clip(action,-1,1))
                raw_total+=float(reward)
                eps[:]=bool(terminated or truncated)
                if terminated or truncated:
                    break
            success=bool(info.get("arrive_dest",False))
            crash=bool(info.get("crash_vehicle",False) or info.get("crash_object",False))
            offroad=bool(info.get("out_of_road",False))
            episodes.append({
                "seed":seed,"steps":t+1,
                "success":success,"collision":crash,"offroad":offroad,
                "timeout":not(success or crash or offroad),
                "route_completion":float(info.get("route_completion",0.0)),
                "reward":raw_total,
            })
    env.close()
    assert [e["seed"] for e in episodes]==SEEDS
    result={
        "stage":"D2n-A independently held-out paired BC vs official A/B/C PPO",
        "variant":args.variant,"phase":args.phase,"bc_seed":42,
        "traffic_density":args.density,"eval_seeds":SEEDS,
        "trained_on_different_seeds":"24000..29119 (density=0.1)",
        "success":sum(e["success"] for e in episodes),
        "collision":sum(e["collision"] for e in episodes),
        "offroad":sum(e["offroad"] for e in episodes),
        "timeout":sum(e["timeout"] for e in episodes),
        "mean_route_completion":float(np.mean([e["route_completion"] for e in episodes])),
        "mean_raw_environment_reward":float(np.mean([e["reward"] for e in episodes])),
        "episodes":episodes,
        "wall_seconds":time.monotonic()-started,
        "no_model_updates":True,
    }
    output=ROOT/"reports/d2n-a"/f"eval-{args.phase}-{args.variant}-density{args.density:.2f}.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"EVAL {args.phase}/{args.variant} density={args.density:.2f}: success="
          f"{result['success']}/24 collisions={result['collision']} "
          f"offroad={result['offroad']} mean_reward={result['mean_raw_environment_reward']:.1f}",
          flush=True)


if __name__=="__main__":
    main()
