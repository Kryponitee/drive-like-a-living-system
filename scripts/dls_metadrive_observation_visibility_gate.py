#!/usr/bin/env python3
"""受控物理前车是否真正进入MetaDrive 259维观测的独立门槛。

同 seed、同 ego 指令，分别运行 free / sham / sudden brake。
比较物理速度、距离和 observation 全量向量；前70步必须相同。
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from dls_metadrive_dynamic_following_v2 import (
    DynamicFollowingEnv,DynamicLeadManager,SCENES,MAX_STEPS)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_dual_pedal_adapter import install_dual_pedal_adapter,wrap_dual_pedal_env

ROOT=Path("/mnt/datasets/lxb/dls")


def run(seed,mode,n_steps=105):
    env=wrap_dual_pedal_env(DynamicFollowingEnv({
        "use_render":False,"image_observation":False,"map":"S","traffic_density":0.,
        "num_scenarios":1,"start_seed":seed,"horizon":n_steps+10,"log_level":50,
    }))
    try:
        obs,_=env.reset(seed=seed)
        ego=env.agent
        install_dual_pedal_adapter(ego,steering_controller=lambda _:0.)
        mgr=env.unwrapped.engine.traffic_manager
        assert isinstance(mgr,DynamicLeadManager)
        mgr.mode=mode
        if mode!="free":mgr.attach_lead(ego,gap_m=18.)
        observations=[]
        lead_speeds=[]
        ego_positions=[]
        for t in range(n_steps):
            # 同一个静态ego控制，不引入神经网络造成的历史不匹配。
            pedal=np.asarray([0.47,0.],dtype=np.float32)
            obs,reward,term,trunc,info=env.step(pedal)
            x=np.asarray(obs,dtype=np.float32)
            assert x.shape==(259,) and np.isfinite(x).all()
            observations.append(x.copy())
            lead_speeds.append(float(mgr.lead.speed)/3.6 if mgr.lead is not None else None)
            ego_positions.append(np.asarray(ego.position,dtype=np.float32).copy())
            if term or trunc:
                raise AssertionError(("physical observation gate ended early",mode,t,info))
        return np.stack(observations),lead_speeds,np.stack(ego_positions)
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=41001)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert not args.output.exists()
    free,_,free_pos=run(args.seed,"free")
    sham,sham_speed,sham_pos=run(args.seed,"sham")
    pulse,pulse_speed,pulse_pos=run(args.seed,"pulse")
    # First 70 steps have same physical traffic and identical ego controls.
    assert np.max(np.abs(sham[:70]-pulse[:70]))<2e-5
    assert np.max(np.abs(sham_pos[:70]-pulse_pos[:70]))<2e-5
    # Lead physically changes speed after treatment.
    delta_speed=np.asarray(sham_speed)-np.asarray(pulse_speed)
    assert float(delta_speed[90:100].mean())>0.4
    deviation=np.linalg.norm(sham-pulse,axis=1)
    free_diff=np.linalg.norm(sham-free,axis=1)
    readout={
        "seed":args.seed,
        "mode":"fixed identical ego actions; one controlled lead absent / sham / emergency braking",
        "initial_ego_lead_visibility_obs_l2_steps_1_20":float(free_diff[1:20].mean()),
        "mean_obs_l2_sham_vs_brake_steps_0_69":float(deviation[:70].mean()),
        "mean_obs_l2_sham_vs_brake_steps_90_99":float(deviation[90:100].mean()),
        "mean_physical_lead_speed_difference_mps_90_99":float(delta_speed[90:100].mean()),
        "individual_observation_dims_that_change_90_99":int(np.count_nonzero(
            np.abs(sham[90:100]-pulse[90:100]).max(axis=0)>1e-5)),
        "lead_present_visible_initially":bool(free_diff[1:20].mean()>1e-4),
        "brake_pulse_observable_after_treatment":bool(deviation[90:100].mean()>1e-4),
        "note":"Uses full MetaDrive observation only; exact lidar channel semantics are not inferred.",
    }
    assert readout["lead_present_visible_initially"], "Lead not visible to input despite physical spawn"
    assert readout["brake_pulse_observable_after_treatment"], "Physical lead intervention absent from input"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(readout,indent=2,ensure_ascii=False))
    print(json.dumps(readout,indent=2,ensure_ascii=False))
    print("PASS: lead present and sudden braking both affect actual 259D sensor vector",flush=True)


if __name__=="__main__":
    main()
