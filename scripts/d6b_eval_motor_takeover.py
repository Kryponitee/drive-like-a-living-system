#!/usr/bin/env python3
"""D6b: 逐场景同环境比较L-R, X-R和冻结X-R阻断消息的单执行单元失活。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv

from dls_local_motor import LocalMotorController, num_params, OBS_DIM, MOTORS


ROOT=Path("/mnt/datasets/lxb/dls")
BASE_SEED=40000
SHARD_SIZE=4
DENSITIES=(0.0,0.10,0.25)
MODES=("local","cross","cross_blocked")
LESIONS=(-1,*range(MOTORS))


def load(arm,seed):
    p=ROOT/"checkpoints/d6b"/f"{arm}-seed{seed}.pt"
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    ckpt=torch.load(p,map_location="cpu",weights_only=False)
    assert ckpt["variant"]=="D6b" and ckpt["arm"]==arm and ckpt["seed"]==seed
    assert ckpt["data_id"]=="d1-v1" and ckpt["num_params"]==114880
    model=LocalMotorController(arm)
    assert num_params(model)==114880
    model.load_state_dict(ckpt["state_dict"],strict=True)
    model.eval().requires_grad_(False)
    mean=np.asarray(ckpt["train_mean"],np.float32)
    std=np.asarray(ckpt["train_std"],np.float32)
    assert mean.shape==std.shape==(OBS_DIM,) and (std>0).all()
    old=ROOT/"checkpoints/d6"/f"{arm}-seed{seed}.pt"
    assert hashlib.sha256(old.read_bytes()).hexdigest()==ckpt["original_D6_sha256"]
    return model,(mean,std),sha,p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    parser.add_argument("--density",type=float,choices=DENSITIES,required=True)
    parser.add_argument("--shard",type=int,choices=tuple(range(6)),required=True)
    args=parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    first=BASE_SEED+SHARD_SIZE*args.shard
    scenes=list(range(first,first+SHARD_SIZE))
    assert scenes[-1]<=BASE_SEED+23
    L,(lmean,lstd),lsha,lpath=load("local",args.seed)
    X,(xmean,xstd),xsha,xpath=load("cross",args.seed)
    np.testing.assert_array_equal(lmean,xmean)
    np.testing.assert_array_equal(lstd,xstd)
    torch.testing.assert_close(L.projection,X.projection,rtol=0,atol=0)

    target=ROOT/"reports/d6b/eval"/f"seed{args.seed}-density{args.density:.2f}-shard{args.shard}.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists()
    env=MetaDriveEnv({
        "use_render":False,"image_observation":False,
        "traffic_density":args.density,"map":3,"start_seed":first,
        "num_scenarios":SHARD_SIZE,"horizon":800,"log_level":50,
    })
    episodes=[]
    started=time.monotonic()
    with torch.no_grad():
        for scene in scenes:
            for lesion in LESIONS:
                mask=torch.ones(MOTORS,dtype=torch.float32)
                if lesion>=0:
                    mask[lesion]=0.0
                for mode in MODES:
                    model=L if mode=="local" else X
                    blocked=mode=="cross_blocked"
                    obs,_=env.reset(seed=scene)
                    state=model.initial_state(1,torch.device("cpu"))
                    total_reward=0.0
                    abs_messages=np.zeros(MOTORS,dtype=np.float64)
                    abs_activity=np.zeros(MOTORS,dtype=np.float64)
                    info={}
                    for t in range(800):
                        raw=np.asarray(obs,np.float32)
                        assert raw.shape==(OBS_DIM,) and np.isfinite(raw).all()
                        x=np.clip((raw-lmean)/lstd,-10.,10.).copy()
                        action,state,diag=model.step(
                            torch.from_numpy(x).unsqueeze(0),state,t,
                            unit_mask=mask,block_messages=blocked,
                            return_diagnostics=True,
                        )
                        abs_messages+=diag["mean_abs_messages"].numpy()
                        abs_activity+=np.abs(diag["activities"].squeeze(0).numpy())
                        npaction=np.clip(action.squeeze(0).numpy(),-1.,1.).astype(np.float32)
                        obs,reward,terminated,truncated,info=env.step(npaction)
                        total_reward+=float(reward)
                        if terminated or truncated:
                            break
                    episodes.append({
                        "scene_seed":scene,"lesion":lesion,"mode":mode,
                        "steps":t+1,
                        "success":bool(info.get("arrive_dest",False)),
                        "collision":bool(info.get("crash_vehicle",False) or info.get("crash_object",False)),
                        "offroad":bool(info.get("out_of_road",False)),
                        "reward":total_reward,
                        "route_completion":float(info.get("route_completion",0.0)),
                        "mean_abs_unit_messages":(abs_messages/(t+1)).tolist(),
                        "mean_abs_unit_activity":(abs_activity/(t+1)).tolist(),
                    })
    env.close()
    assert len(episodes)==SHARD_SIZE*len(LESIONS)*len(MODES)==108
    for mode in MODES:
        for lesion in LESIONS:
            subset=[r for r in episodes if r["mode"]==mode and r["lesion"]==lesion]
            assert [r["scene_seed"] for r in subset]==scenes
    assert hashlib.sha256(lpath.read_bytes()).hexdigest()==lsha
    assert hashlib.sha256(xpath.read_bytes()).hexdigest()==xsha
    by_mode={}
    for mode in MODES:
        subset=[r for r in episodes if r["mode"]==mode]
        healthy=[r for r in subset if r["lesion"]==-1]
        damaged=[r for r in subset if r["lesion"]!=-1]
        by_mode[mode]={
            "healthy_success_of_4":sum(int(r["success"]) for r in healthy),
            "damaged_success_of_32":sum(int(r["success"]) for r in damaged),
            "damaged_mean_reward":float(np.mean([r["reward"] for r in damaged])),
        }
    report={
        "stage":"D6b paired robust local and cross info-takeover test",
        "seed":args.seed,"density":args.density,"shard":args.shard,
        "scene_seeds":scenes,"modes":list(MODES),
        "lesions":list(LESIONS),"num_episodes":len(episodes),
        "checkpoint_sha256":{"local":lsha,"cross":xsha},
        "by_mode":by_mode,"episodes":episodes,
        "wall_seconds":time.monotonic()-started,
    }
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(
        f"D6b seed{args.seed} density{args.density:.2f} shard{args.shard} "
        f"healthy local/cross/blocked="
        f"{by_mode['local']['healthy_success_of_4']}/"
        f"{by_mode['cross']['healthy_success_of_4']}/"
        f"{by_mode['cross_blocked']['healthy_success_of_4']}; "
        f"damaged local/cross/blocked="
        f"{by_mode['local']['damaged_success_of_32']}/"
        f"{by_mode['cross']['damaged_success_of_32']}/"
        f"{by_mode['cross_blocked']['damaged_success_of_32']}",
        flush=True,
    )


if __name__=="__main__":
    main()
