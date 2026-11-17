#!/usr/bin/env python3
"""D6c: 对冻结D6b-cross逐单元做输出关闭×发送信息关闭的2×2因果干预。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv

from dls_local_motor import LocalMotorController, MOTORS, OBS_DIM, num_params


ROOT = Path("/mnt/datasets/lxb/dls")
BASE_SEED = 42000
SHARD_SIZE = 4
DENSITIES = (0.0, 0.10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    parser.add_argument("--density",type=float,choices=DENSITIES,required=True)
    parser.add_argument("--shard",type=int,choices=(0,1,2),required=True)
    args=parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    scenes=list(range(BASE_SEED + SHARD_SIZE*args.shard,
                      BASE_SEED + SHARD_SIZE*(args.shard+1)))
    assert scenes[-1] <= BASE_SEED + 11

    ckpt_path=ROOT/"checkpoints/d6b"/f"cross-seed{args.seed}.pt"
    source_sha=hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=False)
    assert ckpt["variant"]=="D6b" and ckpt["arm"]=="cross"
    assert ckpt["seed"]==args.seed and ckpt["data_id"]=="d1-v1"
    model=LocalMotorController("cross")
    assert num_params(model)==ckpt["num_params"]==114880
    model.load_state_dict(ckpt["state_dict"],strict=True)
    model.eval().requires_grad_(False)
    mean=np.asarray(ckpt["train_mean"],np.float32)
    std=np.asarray(ckpt["train_std"],np.float32)
    assert mean.shape==std.shape==(OBS_DIM,) and (std>0).all()

    dest=ROOT/"reports/d6c/eval"/f"seed{args.seed}-density{args.density:.2f}-shard{args.shard}.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists()
    env=MetaDriveEnv({
        "use_render":False,"image_observation":False,
        "traffic_density":args.density,"map":3,"start_seed":scenes[0],
        "num_scenarios":SHARD_SIZE,"horizon":800,"log_level":50,
    })

    # 一次完整健康基线，逐i额外3组，节约重复8次同一健康环境。
    # sender_drop本身不影响损伤单元自己的局部观测、内部状态或潜在活动。
    conditions=[(-1,False),*[(i,True) for i in range(MOTORS)],
                *[(i,False) for i in range(MOTORS)]]
    # 这里条件必须同时包含 output lesion i + sender blocked i。
    cases=[("base",-1,False,None)]
    for i in range(MOTORS):
        cases.extend((
            ("sender_only",i,False,i),
            ("lesion_only",i,True,None),
            ("lesion_and_sender",i,True,i),
        ))
    assert len(cases)==1+MOTORS*3
    episodes=[]
    started=time.monotonic()
    with torch.no_grad():
        for scene in scenes:
            for case,unit,lesioned,blocked_sender in cases:
                obs,_=env.reset(seed=scene)
                state=model.initial_state(1,torch.device("cpu"))
                unit_mask=torch.ones(MOTORS,dtype=torch.float32)
                if lesioned:
                    assert unit>=0
                    unit_mask[unit]=0.0
                reward_total=0.0
                info={}
                for t in range(800):
                    raw=np.asarray(obs,np.float32)
                    assert raw.shape==(OBS_DIM,) and np.isfinite(raw).all()
                    x=np.clip((raw-mean)/std,-10.0,10.0).copy()
                    action,state=model.step(torch.from_numpy(x).unsqueeze(0),state,t,
                                            unit_mask=unit_mask,blocked_sender=blocked_sender)
                    assert action.shape==(1,2) and torch.isfinite(action).all()
                    obs,reward,terminated,truncated,info=env.step(
                        np.clip(action.squeeze(0).numpy(),-1,1).astype(np.float32))
                    reward_total+=float(reward)
                    if terminated or truncated:
                        break
                episodes.append({
                    "scene_seed":scene,"case":case,"motor":unit,
                    "output_lesioned":lesioned,"message_sender_blocked":blocked_sender,
                    "steps":t+1,"success":bool(info.get("arrive_dest",False)),
                    "collision":bool(info.get("crash_vehicle",False) or info.get("crash_object",False)),
                    "offroad":bool(info.get("out_of_road",False)),
                    "reward":reward_total,
                    "route_completion":float(info.get("route_completion",0.0)),
                })
    env.close()
    assert len(episodes)==SHARD_SIZE*(1+MOTORS*3)
    for scene in scenes:
        subset=[e for e in episodes if e["scene_seed"]==scene]
        assert len(subset)==25
        assert [e["case"] for e in subset]==[v[0] for v in cases]
    assert hashlib.sha256(ckpt_path.read_bytes()).hexdigest()==source_sha
    report={
        "stage":"D6c frozen cross peer-source relay necessity 2x2 intervention",
        "seed":args.seed,"density":args.density,"shard":args.shard,
        "scene_seeds":scenes,"source_sha256":source_sha,
        "num_episodes":len(episodes),"cases_per_scene":len(cases),
        "episodes":episodes,"wall_seconds":time.monotonic()-started,
    }
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"D6c seed{args.seed} density{args.density:.2f} shard{args.shard}: "
          f"{len(episodes)} episodes, {len(scenes)} scenes, "
          f"healthy_success={sum(e['success'] for e in episodes if e['case']=='base')}/{len(scenes)}",
          flush=True)


if __name__=="__main__":
    main()
