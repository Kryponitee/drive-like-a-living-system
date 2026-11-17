#!/usr/bin/env python3
"""D3d: C BC actor fast GRU hidden-state reset at EVERY tick; keep slow state z.

Use the unchanged original C controller and three existing D1 BC checkpoints.
This is a frozen inference test only. D3c baseline and cross-message controls
are reused rather than resimulated.
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
SCENES=list(range(32000,32024))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-seed",type=int,choices=(42,43,44),required=True)
    a=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    path=ROOT/"checkpoints/d1-v1"/f"C-seed{a.model_seed}.pt"
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    assert ckpt["variant"]=="C"
    actor,_=model_info("C")
    actor.load_state_dict(ckpt["state_dict"],strict=True)
    assert nparams(actor)==ckpt["num_params"]
    actor.eval().requires_grad_(False)
    mean=np.asarray(ckpt["train_mean"],np.float32)
    std=np.asarray(ckpt["train_std"],np.float32)
    assert mean.shape==std.shape==(OBS_DIM,) and std.min()>0

    # The very first tick is identical: no prior temporal state exists yet.
    with torch.no_grad():
        gen=torch.Generator().manual_seed(381)
        for t in (0,5,10):
            obs=torch.randn((3,OBS_DIM),generator=gen)
            h,z=actor.initial_state(3,torch.device("cpu"))
            act_a,s_a=actor.step(obs,(h,z),t)
            act_b,s_b=actor.step(obs,(torch.zeros_like(h),z),t)
            assert torch.equal(act_a,act_b)
            assert torch.equal(s_a[0],s_b[0]) and torch.equal(s_a[1],s_b[1])

    env=MetaDriveEnv({
        "use_render":False,"image_observation":False,
        "traffic_density":0.10,"map":3,
        "start_seed":SCENES[0],"num_scenarios":len(SCENES),
        "horizon":800,"log_level":50,
    })
    episodes=[]
    timer=time.monotonic()
    mean_discarded_hidden_abs=0.0
    hidden_discards=0
    with torch.no_grad():
        for scene in SCENES:
            obs,_=env.reset(seed=scene)
            state=actor.initial_state(1,torch.device("cpu"))
            reward_total=0.0
            info={}
            for t in range(800):
                raw=np.asarray(obs,np.float32)
                assert raw.shape==(OBS_DIM,) and np.isfinite(raw).all()
                x=np.clip((raw-mean)/std,-10.0,10.0).copy()
                hidden,slow=state
                mean_discarded_hidden_abs+=float(hidden.abs().mean())
                hidden_discards+=1
                action,state=actor.step(
                    torch.from_numpy(x).unsqueeze(0),
                    (torch.zeros_like(hidden),slow),
                    t,
                )
                assert action.shape==(1,2) and torch.isfinite(action).all()
                obs,reward,terminated,truncated,info=env.step(
                    action[0].numpy().astype(np.float32)
                )
                reward_total+=float(reward)
                if terminated or truncated:
                    break
            success=bool(info.get("arrive_dest",False))
            collision=bool(info.get("crash_vehicle",False) or info.get("crash_object",False))
            offroad=bool(info.get("out_of_road",False))
            episodes.append({
                "seed":scene,"steps":t+1,"success":success,
                "collision":collision,"offroad":offroad,
                "timeout":not (success or collision or offroad),
                "reward":reward_total,
                "route_completion":float(info.get("route_completion",0.0)),
            })
    env.close()
    assert len(episodes)==len(SCENES)
    assert hidden_discards==sum(e["steps"] for e in episodes)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    record={
        "stage":"D3d C BC fast GRU hidden reset every tick, slow z preserved",
        "model":"C","bc_seed":a.model_seed,
        "source_sha256":digest,
        "fresh_evaluation_scene_ids_from_D3c":SCENES,
        "traffic_density":0.10,
        "intervention":"set all module recurrent h_(t-1) to zero before actor.step, retain z and t",
        "mean_discarded_fast_hidden_L1_per_coordinate":mean_discarded_hidden_abs/hidden_discards,
        "no_training":True,"model_unchanged":True,
        "success":sum(e["success"] for e in episodes),
        "collision":sum(e["collision"] for e in episodes),
        "offroad":sum(e["offroad"] for e in episodes),
        "mean_raw_reward":float(np.mean([e["reward"] for e in episodes])),
        "mean_route_completion":float(np.mean([e["route_completion"] for e in episodes])),
        "episodes":episodes,
        "wall_seconds":time.monotonic()-timer,
        "caveat":"Loss of performance can reflect hidden-state OOD, not proof that environment genuinely needs history. z and observation are unchanged, so this tests only fast hidden-state retention.",
    }
    out=ROOT/"reports/d3d"/f"reset-C-seed{a.model_seed}.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists()
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"D3d seed={a.model_seed}: recurrent h reset, success={record['success']}/24, "
          f"mean_reward={record['mean_raw_reward']:.2f}, "
          f"mean_abs_discarded_h={record['mean_discarded_fast_hidden_L1_per_coordinate']:.3f}",flush=True)


if __name__=="__main__":
    main()
