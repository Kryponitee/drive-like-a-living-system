#!/usr/bin/env python3
"""D4 matched closed-loop MetaDrive evaluation, including post-event message blockade.

Only frozen D4 pretrained models see the normal 259D observation.
Pulse/sham differ only by lead braking after a seed-dependent trigger.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_d4_following_scene import (
    D4FollowingEnv,D4TrafficManager,MAX_STEPS,make_env,physical_gap,scene_spec,
    assert_matched_pulse_sham_spec)
from dls_dual_pedal_adapter import install_dual_pedal_adapter
from dls_longitudinal_motor import LocalMotorController,model_info

ROOT=Path("/mnt/datasets/lxb/dls")
TEST_SEEDS=tuple(range(71000,71012))


def rollout(model,variant,mean,std,scene_seed,mode,block_peer=False):
    assert mode in ("sham","pulse")
    assert not block_peer or (variant=="C" and mode=="pulse")
    spec=scene_spec(scene_seed,mode)
    env=make_env(scene_seed)
    records=[]
    try:
        obs,info=env.reset(seed=scene_seed)
        ego=env.agent
        install_dual_pedal_adapter(ego,steering_controller=lambda _:0.)
        manager=env.unwrapped.engine.traffic_manager
        assert isinstance(manager,D4TrafficManager)
        manager.attach(ego,spec)
        hidden=model.initial_state(1,torch.device("cpu"))
        initial_position=np.asarray(ego.position,dtype=np.float64).copy()
        with torch.no_grad():
            for t in range(MAX_STEPS):
                norm=np.clip((np.asarray(obs,dtype=np.float32)-mean)/std,-10.,10.)
                assert norm.shape==(259,) and np.isfinite(norm).all()
                x=torch.from_numpy(norm.copy())[None]
                if isinstance(model,LocalMotorController):
                    blocked=bool(block_peer and t>=spec["brake_start_step"])
                    action_tensor,hidden,diag=model.step(x,hidden,t,
                        block_messages=blocked,return_diagnostics=True)
                    per_module=(diag["activities"][0,:,None]*model.decoder.T)
                    torch.testing.assert_close(
                        action_tensor[0],torch.sigmoid(per_module.sum(0)),
                        atol=2e-6,rtol=1e-5)
                    module=per_module.numpy().tolist()
                else:
                    action_tensor,hidden=model.step(x,hidden,t)
                    module=None
                action=action_tensor[0].numpy().astype(np.float32)
                assert np.all(np.isfinite(action)) and np.all((action>=0)&(action<=1))
                obs,reward,done,truncated,info=env.step(action)
                center,gap,lateral=physical_gap(ego,manager.lead)
                records.append({
                    "t":t,"phase":int(manager.phase),
                    "surface_gap_m":gap,
                    "center_gap_m":center,
                    "lateral_m":lateral,
                    "ego_speed_mps":float(ego.speed)/3.6,
                    "lead_speed_mps":float(manager.lead.speed)/3.6,
                    "throttle":float(action[0]),"brake":float(action[1]),
                    "module_logits":module,
                })
                if done or truncated:
                    break
        event=spec["brake_start_step"]
        post=[r for r in records if r["t"]>=event]
        collision=bool(info.get("crash_vehicle",False)
                     or info.get("crash_object",False)
                     or info.get("crash_building",False))
        return {
            "seed":scene_seed,"mode":mode,"block_peer_after_event":block_peer,
            "event_step":event,
            "event_reached":bool(len(records)>event),
            "steps":len(records),"collision":collision,
            "success":bool(info.get("arrive_dest",False)),
            "out_of_road":bool(info.get("out_of_road",False)),
            "route_completion":float(info.get("route_completion",0.)),
            "min_surface_gap_m":min(r["surface_gap_m"] for r in records),
            "speed_after_event_mps":float(np.mean([r["ego_speed_mps"] for r in post])) if post else None,
            "brake_fraction_after_event":float(np.mean([r["brake"]>.1 for r in post])) if post else None,
            "mean_coactive_fraction":float(np.mean([
                r["throttle"]>.1 and r["brake"]>.1 for r in records])),
            "progress_m":float(np.linalg.norm(np.asarray(ego.position)-initial_position)),
            "time_series":records,
        }
    finally:
        env.close()


def verify_matching(a,b):
    assert a["seed"]==b["seed"] and a["event_step"]==b["event_step"]
    event=a["event_step"]
    if not a["event_reached"] or not b["event_reached"]:
        # If an episode ends before braking, report it; do not fabricate exposure.
        return False
    for x,y in zip(a["time_series"][:event],b["time_series"][:event]):
        for key in ("ego_speed_mps","lead_speed_mps","surface_gap_m","throttle","brake"):
            assert abs(x[key]-y[key])<8e-5,(a["seed"],x["t"],key,x[key],y[key])
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("A","B","C"),required=True)
    p.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    p.add_argument("--data-id",required=True)
    a=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    assert a.data_id.startswith("d4-dynamic-follow-v1-")
    file=ROOT/"checkpoints"/a.data_id/f"{a.variant}-seed{a.seed}.pt"
    path=ROOT/"reports"/a.data_id/"closed-loop"/f"{a.variant}-seed{a.seed}.json"
    assert file.is_file() and not path.exists()
    ckpt=torch.load(file,map_location="cpu",weights_only=False)
    assert ckpt["data_id"]==a.data_id and ckpt["variant"]==a.variant and ckpt["seed"]==a.seed
    model,_=model_info(a.variant)
    model.load_state_dict(ckpt["state_dict"],strict=True)
    model.eval()
    mean=np.asarray(ckpt["train_mean"],dtype=np.float32)
    std=np.asarray(ckpt["train_std"],dtype=np.float32)
    assert mean.shape==std.shape==(259,) and (std>0).all()
    experiments=[]
    started=time.monotonic()
    for seed in TEST_SEEDS:
        assert assert_matched_pulse_sham_spec(seed)
        sham=rollout(model,a.variant,mean,std,seed,"sham")
        pulse=rollout(model,a.variant,mean,std,seed,"pulse")
        sham_pulse_matched=verify_matching(sham,pulse)
        experiments.extend([sham,pulse])
        if a.variant=="C":
            blocked=rollout(model,a.variant,mean,std,seed,"pulse",block_peer=True)
            blocked_pulse_matched=verify_matching(pulse,blocked)
            experiments.append(blocked)
            assert blocked_pulse_matched==sham_pulse_matched
        print(f"TEST {a.variant} seed={a.seed} scene={seed} event={pulse['event_step']} "
              f"matched={sham_pulse_matched} sham={sham['success']} "
              f"pulse={pulse['success']} crash={pulse['collision']}",flush=True)
    assert len(experiments)==len(TEST_SEEDS)*(3 if a.variant=="C" else 2)
    report={
        "stage":"D4 matched fresh-scene MetaDrive dynamic following",
        "data_id":a.data_id,"variant":a.variant,"seed":a.seed,
        "checkpoint":str(file),
        "checkpoint_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
        "test_scene_seeds":list(TEST_SEEDS),
        "student_observation":"MetaDrive 259D, no privileged physics",
        "lead_modes":"sham/pulse differ only at random emergency event; mode is never given to model",
        "message_intervention":"C additionally blocks peer messages from the emergency step onward",
        "episodes":experiments,
        "wall_seconds":time.monotonic()-started,
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("EVAL_DONE",a.variant,a.seed,len(experiments),flush=True)


if __name__=="__main__":
    main()
