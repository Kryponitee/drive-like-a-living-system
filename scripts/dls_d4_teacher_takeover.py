#!/usr/bin/env python3
"""D4 scientific diagnosis: preserve learned-policy history, replace ONLY post-event action.

The replacement teacher is privileged (true physical car gap and lead speed).
This tests recoverability of the already reached event-time physical state,
NOT the competence of an observation-only DLS controller.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_d4_following_scene import (
    D4TrafficManager,make_env,physical_gap,scene_spec,teacher_action,
    MAX_STEPS,assert_matched_pulse_sham_spec)
from dls_dual_pedal_adapter import install_dual_pedal_adapter
from dls_longitudinal_motor import model_info

ROOT=Path("/mnt/datasets/lxb/dls")
DATA_ID="d4-dynamic-follow-v1-44481405397"


def takeover_episode(model,variant,scene_seed,mean,std,baseline):
    assert assert_matched_pulse_sham_spec(scene_seed)
    spec=scene_spec(scene_seed,"pulse")
    assert spec["brake_start_step"]==baseline["event_step"]
    env=make_env(scene_seed)
    trace=[]
    try:
        obs,_=env.reset(seed=scene_seed)
        ego=env.agent
        install_dual_pedal_adapter(ego,steering_controller=lambda _:0.)
        mgr=env.unwrapped.engine.traffic_manager
        assert isinstance(mgr,D4TrafficManager)
        mgr.attach(ego,spec)
        state=model.initial_state(1,torch.device("cpu"))
        origin=np.asarray(ego.position,dtype=np.float64)
        last_info={}
        with torch.no_grad():
            for t in range(MAX_STEPS):
                if t<spec["brake_start_step"]:
                    normalized=np.clip((np.asarray(obs,dtype=np.float32)-mean)/std,-10.,10.)
                    x=torch.from_numpy(normalized.copy()).unsqueeze(0)
                    output,state=model.step(x,state,t)
                    pedal=output[0].numpy().astype(np.float32)
                else:
                    pedal=teacher_action(ego,mgr.lead)
                assert pedal.shape==(2,) and np.isfinite(pedal).all()
                obs,reward,done,truncated,last_info=env.step(pedal)
                _,gap,_=physical_gap(ego,mgr.lead)
                record={
                    "t":t,"gap_m":gap,"ego_speed_mps":float(ego.speed)/3.6,
                    "lead_speed_mps":float(mgr.lead.speed)/3.6,
                    "throttle":float(pedal[0]),"brake":float(pedal[1]),
                    "teacher_controls":t>=spec["brake_start_step"],
                }
                if t<spec["brake_start_step"]:
                    reference=baseline["time_series"][t]
                    for key,renamed in (("surface_gap_m","gap_m"),("ego_speed_mps","ego_speed_mps"),
                                        ("lead_speed_mps","lead_speed_mps"),("throttle","throttle"),
                                        ("brake","brake")):
                        assert np.isclose(reference[key],record[renamed],atol=8e-5,rtol=0), (
                            "prefix diverged",variant,scene_seed,t,key,reference[key],record[renamed])
                trace.append(record)
                if done or truncated:break
        assert len(trace)>spec["brake_start_step"]
        collide=bool(last_info.get("crash_vehicle",False)
                     or last_info.get("crash_object",False)
                     or last_info.get("crash_building",False))
        arrive=bool(last_info.get("arrive_dest",False))
        road=bool(last_info.get("out_of_road",False))
        return {
            "seed":scene_seed,"event_step":spec["brake_start_step"],
            "steps":len(trace),"collision":collide,"success":arrive,
            "out_of_road":road,"safe_arrival":arrive and not collide and not road,
            "min_surface_gap_m":min(r["gap_m"] for r in trace),
            "progress_m":float(np.linalg.norm(np.asarray(ego.position)-origin)),
            "baseline_collision":bool(baseline["collision"]),
            "baseline_arrival":bool(baseline["success"]),
            "baseline_safe_arrival":bool(baseline["success"] and not baseline["collision"] and not baseline["out_of_road"]),
            "same_pre_event_physical_and_action_prefix":True,
            "after_event_teacher_is_privileged":True,
        }
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("A","B","C"),required=True)
    p.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    args=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    folder=ROOT/"reports"/DATA_ID/"takeover-budget80"
    dest=folder/f"{args.variant}-seed{args.seed}.json"
    assert not dest.exists()
    baseline_path=ROOT/"reports"/DATA_ID/"closed-loop-budget80"/f"{args.variant}-seed{args.seed}.json"
    ckpt_file=ROOT/"checkpoints"/DATA_ID/"budget80"/f"{args.variant}-seed{args.seed}.pt"
    base=json.loads(baseline_path.read_text())
    ckpt=torch.load(ckpt_file,map_location="cpu",weights_only=False)
    model,_=model_info(args.variant)
    model.load_state_dict(ckpt["state_dict"],strict=True)
    model.eval()
    mean=np.asarray(ckpt["train_mean"],dtype=np.float32)
    std=np.asarray(ckpt["train_std"],dtype=np.float32)
    rows=[]
    for scene_seed in range(71000,71012):
        case=next(r for r in base["episodes"] if r["seed"]==scene_seed
                  and r["mode"]=="pulse" and not r["block_peer_after_event"])
        result=takeover_episode(model,args.variant,scene_seed,mean,std,case)
        rows.append(result)
        print("TAKEOVER",args.variant,args.seed,scene_seed,
              "baseline_safe",result["baseline_safe_arrival"],
              "teacher_takeover_safe",result["safe_arrival"],
              "min_gap",f"{result['min_surface_gap_m']:.2f}",flush=True)
    doc={
        "stage":"DLS D4 post-event privileged teacher takeover from exact learned-policy state",
        "variant":args.variant,"model_seed":args.seed,
        "data_id":DATA_ID,
        "checkpoint_sha256":hashlib.sha256(ckpt_file.read_bytes()).hexdigest(),
        "results":rows,
        "scope":"causal physical intervention after matched initial network history; teacher reads true leading-car distance and speed",
    }
    folder.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(doc,indent=2,ensure_ascii=False))
    print("DONE",args.variant,args.seed,"base safe",
          sum(r["baseline_safe_arrival"] for r in rows),
          "takeover safe",sum(r["safe_arrival"] for r in rows),flush=True)


if __name__=="__main__":
    main()
