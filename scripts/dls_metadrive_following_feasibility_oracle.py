#!/usr/bin/env python3
"""只读物理可行性上界：带前车真实距离/速度的规则式ACC控制器。

不同于DLS模型，它访问额外物理真值，只用于判断同一急刹场景能否安全前进。
不把特权成功算作公平的神经模型性能。
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from dls_metadrive_dynamic_following_v2 import DynamicFollowingEnv,DynamicLeadManager
from dls_metadrive_controlled_lead_smoke import get_relative
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_dual_pedal_adapter import install_dual_pedal_adapter,wrap_dual_pedal_env

ROOT=Path("/mnt/datasets/lxb/dls")
MAX_STEPS=160


def privileged_follow_control(ego,lead):
    gap,_=get_relative(ego,lead)
    surface_gap=gap-(float(ego.LENGTH)+float(lead.LENGTH))/2
    v=float(ego.speed)/3.6
    vl=float(lead.speed)/3.6
    closing=max(v-vl,0.)
    # 固定物理安全策略：速度适应前车与可用空间；紧急距离触发更强制动。
    desired_gap=3.5+1.1*v+1.0*closing
    target_v=min(9.,max(0.,vl+0.52*(surface_gap-desired_gap)))
    accel=1.0*(target_v-v)-0.65*closing
    if surface_gap<desired_gap:
        accel=min(accel,-0.85*(desired_gap-surface_gap))
    throttle=float(np.clip(accel/1.6,0.,1.))
    brake=float(np.clip(-accel/2.5,0.,1.))
    return np.asarray((throttle,brake),dtype=np.float32),float(surface_gap),float(desired_gap)


def run(seed,mode):
    env=wrap_dual_pedal_env(DynamicFollowingEnv({
        "use_render":False,"image_observation":False,"map":"S",
        "traffic_density":0.,"num_scenarios":1,"start_seed":seed,
        "horizon":MAX_STEPS+10,"log_level":50,
    }))
    trace=[]
    try:
        obs,_=env.reset(seed=seed)
        ego=env.agent
        install_dual_pedal_adapter(ego,steering_controller=lambda _:0.)
        mgr=env.unwrapped.engine.traffic_manager
        assert isinstance(mgr,DynamicLeadManager)
        mgr.mode=mode
        mgr.attach_lead(ego,gap_m=18)
        origin=np.asarray(ego.position,dtype=np.float64).copy()
        info={}
        for t in range(MAX_STEPS):
            action,gap,desired=privileged_follow_control(ego,mgr.lead)
            obs,reward,done,truncated,info=env.step(action)
            trace.append({
                "t":t,"phase":mgr.phase,
                "ego_speed_mps":float(ego.speed)/3.6,
                "lead_speed_mps":float(mgr.lead.speed)/3.6,
                "surface_gap_m":gap,
                "desired_surface_gap_m":desired,
                "throttle":float(action[0]),"brake":float(action[1]),
                "ego_distance_m":float(np.linalg.norm(np.asarray(ego.position)-origin)),
            })
            if done or truncated:break
        assert len(trace)>=100
        return {
            "seed":seed,"condition":mode,
            "steps":len(trace),
            "collision":bool(info.get("crash_vehicle",False)
                or info.get("crash_object",False)
                or info.get("crash_building",False)),
            "success":bool(info.get("arrive_dest",False)),
            "out_of_road":bool(info.get("out_of_road",False)),
            "mean_ego_speed_mps":float(np.mean([x["ego_speed_mps"] for x in trace])),
            "distance_progress_m":trace[-1]["ego_distance_m"],
            "min_surface_gap_m":min(x["surface_gap_m"] for x in trace),
            "phase_reached":max(x["phase"] for x in trace),
            "steps_trace":trace,
        }
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert not args.output.exists()
    records=[]
    for seed in (41001,41002):
        for mode in ("sham","pulse"):
            result=run(seed,mode)
            records.append(result)
            print("REFERENCE",json.dumps({
                k:v for k,v in result.items() if k!="steps_trace"
            }),flush=True)
    assert len(records)==4
    assert all(not e["collision"] for e in records), "reference control did not achieve safe following"
    assert all(e["distance_progress_m"]>25 for e in records), "reference trivial stopped vehicle"
    assert all(e["min_surface_gap_m"]>0 for e in records), "reference made physical contact"
    output={
        "stage":"DLS dynamic lead physical feasibility reference (privileged)",
        "conditions":"same physical scripted lead as frozen DLS network; uses true speed and distance unavailable directly to DLS",
        "scope":"physical nontrivial safe mobility feasibility, NOT matched input/controller comparison",
        "results":records,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print("PASS: privileged physics-feasible nonzero-progress safe controller",flush=True)


if __name__=="__main__":
    main()
