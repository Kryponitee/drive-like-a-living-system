#!/usr/bin/env python3
"""DLS D1' 冻结模型在 MetaDrive 动态前车下闭环运行。

官方 BaseManager 控制实体前车，比较 free/constant/pulse；重用原 BC checkpoint。
不能将动态风险指标、可见的 module logit 变化直接等同于已涌现协调组。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv

sys.path.insert(0,str(Path(__file__).resolve().parent))
from dls_metadrive_controlled_lead_smoke import ScriptedLeadManager,get_relative
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_dual_pedal_adapter import install_dual_pedal_adapter,wrap_dual_pedal_env
from dls_longitudinal_motor import model_info,LocalMotorController

ROOT=Path("/mnt/datasets/lxb/dls")
DATA_ID="d1-longitudinal-v1-44117653285"
SCENES=(41001,41002)
CONDITIONS=("sham","pulse")
PULSE_SPEED_MPS=(5.0,7.0,0.0,6.0)
SHAM_SPEED_MPS=(5.0,7.0,7.0,6.0)
PHASE_LENGTH=35
MAX_STEPS=160


class DynamicLeadManager(ScriptedLeadManager):
    def __init__(self):
        super().__init__()
        self.mode="free"

    def reset(self):
        super().reset()
        self.mode="free"

    def before_step(self):
        if self.lead is None:
            assert self.mode=="free"
            return {}
        assert self.mode in ("pulse","sham"), self.mode
        self.phase=min(self.tick//PHASE_LENGTH,3)
        target=(PULSE_SPEED_MPS if self.mode=="pulse" else SHAM_SPEED_MPS)[self.phase]
        actual=float(self.lead.speed)/3.6
        control=float(np.clip(0.22*(target-actual),-1.0,1.0))
        self.lead.before_step([0.0,control])
        self.tick+=1
        return {}


class DynamicFollowingEnv(MetaDriveEnv):
    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("traffic_manager",DynamicLeadManager())


def make_env(seed):
    return wrap_dual_pedal_env(DynamicFollowingEnv({
        "use_render":False,
        "image_observation":False,
        "map":"S",
        "traffic_density":0.0,
        "num_scenarios":1,
        "start_seed":seed,
        "horizon":MAX_STEPS+10,
        "log_level":50,
    }))


def run_one(model,variant,mean,std,seed,condition):
    env=make_env(seed)
    records=[]
    try:
        obs,reset_info=env.reset(seed=seed)
        ego=env.agent
        install_dual_pedal_adapter(ego,steering_controller=lambda _:0.)
        manager=env.unwrapped.engine.traffic_manager
        assert isinstance(manager,DynamicLeadManager)
        manager.mode=condition
        if condition!="free":
            manager.attach_lead(ego,gap_m=18)
        hidden=model.initial_state(1,torch.device("cpu"))
        last_info=reset_info
        with torch.no_grad():
            for t in range(MAX_STEPS):
                state=np.asarray(obs,dtype=np.float32)
                assert state.shape==(259,) and np.isfinite(state).all()
                norm=np.clip((state-mean)/std,-10,10)
                x=torch.from_numpy(norm.copy()).unsqueeze(0)
                before=hidden
                if isinstance(model,LocalMotorController):
                    act,hidden,diag=model.step(x,before,t,return_diagnostics=True)
                    contrib=(diag["activities"][0].unsqueeze(-1)
                             *model.decoder.T)
                    assert torch.allclose(act,torch.sigmoid(contrib.sum(0))[None],atol=2e-6)
                    # 用相同 h_t，阻断当步直接 peer 消息；不改变真实闭环轨迹
                    if variant=="C":
                        no_msg,_,_=model.step(
                            x,before,t,block_messages=True,return_diagnostics=True)
                        message_action_shift=float((act-no_msg).abs().mean().item())
                    else:
                        message_action_shift=None
                    one_unit_removed=(torch.sigmoid(contrib.sum(0)[None]-contrib)
                                      -act).abs().mean(-1).tolist()
                    contribution=contrib.tolist()
                else:
                    act,hidden=model.step(x,before,t)
                    message_action_shift=None
                    one_unit_removed=None
                    contribution=None
                action=np.clip(act.numpy()[0],0.,1.).astype(np.float32)
                obs,reward,terminated,truncated,info=env.step(action)
                last_info=info
                gap=None
                surface_gap=None
                speed_lead=None
                phase=-1
                if manager.lead is not None:
                    gap,_=get_relative(ego,manager.lead)
                    surface_gap=float(gap-(float(ego.LENGTH)+float(manager.lead.LENGTH))/2)
                    speed_lead=float(manager.lead.speed)/3.6
                    phase=int(manager.phase)
                records.append({
                    "t":t,"phase":phase,"gap_m":gap,"surface_gap_m":surface_gap,
                    "ego_speed_mps":float(ego.speed)/3.6,
                    "lead_speed_mps":speed_lead,
                    "throttle":float(action[0]),"brake":float(action[1]),
                    "reward":float(reward),
                    "unit_logit_contributions":contribution,
                    "one_tick_module_readout_mask_action_shift":one_unit_removed,
                    "one_tick_message_block_action_shift":message_action_shift,
                })
                if terminated or truncated:
                    break
        assert len(records)>0
        speed_diffs=[rec["ego_speed_mps"]-rec["lead_speed_mps"]
                     for rec in records if rec["lead_speed_mps"] is not None]
        surface_gaps=[rec["surface_gap_m"] for rec in records
                      if rec["surface_gap_m"] is not None]
        for rec in records:
            if rec["unit_logit_contributions"] is not None:
                assert len(rec["unit_logit_contributions"])==8
                assert len(rec["unit_logit_contributions"][0])==2
        return {
            "scene_seed":seed,"condition":condition,
            "steps":len(records),
            "success":bool(last_info.get("arrive_dest",False)),
            "collision":bool(last_info.get("crash_vehicle",False)
                             or last_info.get("crash_object",False)
                             or last_info.get("crash_building",False)),
            "out_of_road":bool(last_info.get("out_of_road",False)),
            "min_surface_gap_m":min(surface_gaps) if surface_gaps else None,
            "last_surface_gap_m":surface_gaps[-1] if surface_gaps else None,
            "min_relative_speed_mps":min(speed_diffs) if speed_diffs else None,
            "max_closing_speed_mps":max(speed_diffs) if speed_diffs else None,
            "max_lead_speed_mps":(max(rec["lead_speed_mps"] for rec in records)
                                  if condition!="free" else None),
            "min_lead_speed_mps":(min(rec["lead_speed_mps"] for rec in records)
                                  if condition!="free" else None),
            "max_phase_seen":max(rec["phase"] for rec in records),
            "mean_reward_step":float(np.mean([r["reward"] for r in records])),
            "mean_coactive_fraction":float(np.mean([
                r["throttle"]>.1 and r["brake"]>.1 for r in records])),
            "mean_one_tick_message_shift":(float(np.mean([
                r["one_tick_message_block_action_shift"] for r in records]))
                if variant=="C" else None),
            "phase_samples":{
                str(p):sum(r["phase"]==p for r in records) for p in (-1,0,1,2,3)
            },
            "steps_trace":records,
        }
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--variant",choices=("A","B","C"),required=True)
    parser.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    parser.add_argument("--output-dir",type=Path,
                        default=ROOT/"reports"/DATA_ID/"dynamic-follow-v2")
    args=parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    ckpt=ROOT/"checkpoints"/DATA_ID/f"{args.variant}-seed{args.seed}.pt"
    report_path=args.output_dir/f"{args.variant}-seed{args.seed}.json"
    assert not report_path.exists()
    weights=torch.load(ckpt,map_location="cpu",weights_only=False)
    assert weights["variant"]==args.variant and weights["data_id"]==DATA_ID
    mean=np.asarray(weights["train_mean"],dtype=np.float32)
    std=np.asarray(weights["train_std"],dtype=np.float32)
    assert mean.shape==std.shape==(259,) and (std>0).all()
    model,info=model_info(args.variant)
    model.load_state_dict(weights["state_dict"],strict=True)
    model.eval()
    started=time.monotonic()
    outcomes=[]
    for seed in SCENES:
        for condition in CONDITIONS:
            item=run_one(model,args.variant,mean,std,seed,condition)
            outcomes.append(item)
            print(f"model={args.variant} seed={args.seed} scenario={seed} mode={condition} "
                  f"steps={item['steps']} crash={item['collision']} "
                  f"gap={item['min_surface_gap_m']} phases={item['max_phase_seen']}",
                  flush=True)
    assert len(outcomes)==len(CONDITIONS)*len(SCENES)
    # sham与急刹在前70步物理条件完全相同；保证差异仅来自t>=70的前车制动。
    for seed in SCENES:
        sham=next(z for z in outcomes if z["scene_seed"]==seed and z["condition"]=="sham")
        pulse=next(z for z in outcomes if z["scene_seed"]==seed and z["condition"]=="pulse")
        assert min(sham["steps"],pulse["steps"])>=PHASE_LENGTH*2, "Episode ended before intervention"
        for a,b in zip(sham["steps_trace"][:70],pulse["steps_trace"][:70]):
            for key in ("gap_m","ego_speed_mps","lead_speed_mps","throttle","brake"):
                assert np.isclose(a[key],b[key],atol=5e-5,rtol=0), (seed,a["t"],key,a[key],b[key])
    report={
        "stage":"DLS D1-prime frozen model dynamic MetaDrive lead following P0",
        "model_variant":args.variant,"training_seed":args.seed,
        "model_checkpoint":str(ckpt),
        "model_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "scene_seeds":SCENES,
        "conditions":CONDITIONS,
        "physical_lead_pulse_mps":PULSE_SPEED_MPS,
        "physical_lead_sham_mps":SHAM_SPEED_MPS,
        "phase_length_in_simulation_steps":PHASE_LENGTH,
        "initial_center_gap_m":18.,
        "fixed_steering":0,
        "max_steps":MAX_STEPS,
        "results":outcomes,
        "same_prefix_counterfactual_validated":True,
        "scope":"closed loop original pretrained controllers on new scripted physical traffic; no training on this task",
        "limitation":"only stepwise module output / message changes measured; not proof of causal long-horizon compensation",
        "runtime_s":time.monotonic()-started,
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("PASS trained DLS model executed in dynamically scripted MetaDrive following scene",flush=True)


if __name__=="__main__":
    main()
