#!/usr/bin/env python3
"""D6 pilot: aggregate privileged teacher labels on frozen students' visited states.

Use three independent D4 policy families to explore physical states, all only on
new 52000..52011 training seeds. No D6 learner uses privileged state as input.
"""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_d4_following_scene import (
    make_env,scene_spec,assert_matched_pulse_sham_spec,
    D4TrafficManager,teacher_action,physical_gap,MAX_STEPS)
from dls_dual_pedal_adapter import install_dual_pedal_adapter
from dls_longitudinal_motor import model_info

ROOT=Path("/mnt/datasets/lxb/dls")
D4="d4-dynamic-follow-v1-44481405397"
OUT=ROOT/"demonstrations"/D4/"d6-online-coverage-v0.npz"
REPORT=ROOT/"reports"/D4/"d6-online-coverage-v0"/"teacher-labelled-sample-metadata.json"

def rollout(model,mean,std,seed,mode):
    spec=scene_spec(seed,mode)
    env=make_env(seed)
    ob=[];acts=[];gaps=[];records=[]
    try:
        o,_=env.reset(seed=seed)
        ego=env.agent
        install_dual_pedal_adapter(ego,steering_controller=lambda _:0.)
        manager=env.unwrapped.engine.traffic_manager
        assert isinstance(manager,D4TrafficManager)
        manager.attach(ego,spec)
        state=model.initial_state(1,torch.device("cpu"))
        last_info={}
        with torch.no_grad():
            for t in range(MAX_STEPS):
                x=np.asarray(o,dtype=np.float32)
                assert x.shape==(259,) and np.isfinite(x).all()
                ref=teacher_action(ego,manager.lead)
                gap=physical_gap(ego,manager.lead)[1]
                inp=np.clip((x-mean)/std,-10.,10.)
                pedal,state=model.step(torch.from_numpy(inp.copy())[None],state,t)
                action=pedal.numpy()[0].astype(np.float32)
                ob.append(x.copy())
                acts.append(ref.copy())
                gaps.append(gap)
                records.append((t,ref[1],action[1],gap))
                o,reward,done,truncated,last_info=env.step(action)
                if done or truncated:break
        assert len(ob)>spec["brake_start_step"]
        out={
            "seed":seed,"mode":mode,"length":len(ob),
            "event_step":spec["brake_start_step"],
            "student_collision":bool(last_info.get("crash_vehicle",False)),
            "teacher_brake_fraction":float((np.stack(acts)[:,1]>.1).mean()),
            "student_mean_brake":float(np.mean([r[2] for r in records])),
            "teacher_mean_brake":float(np.mean([r[1] for r in records])),
            "minimum_surface_gap_m":float(min(gaps)),
        }
        return np.stack(ob),np.stack(acts),out
    finally:
        env.close()

def main():
    assert os.path.ismount("/mnt/datasets")
    assert not OUT.exists() and not REPORT.exists()
    torch.set_num_threads(1)
    obs=[];labels=[];offsets=[0];modes=[];seeds=[];sources=[];metadata=[]
    for variant_index,variant in enumerate("ABC"):
        file=ROOT/"checkpoints"/D4/"full-seq80"/f"{variant}-seed42.pt"
        weights=torch.load(file,map_location="cpu",weights_only=False)
        assert weights["seed"]==42 and weights["variant"]==variant
        model,_=model_info(variant)
        model.load_state_dict(weights["state_dict"],strict=True)
        model.eval()
        mean=np.asarray(weights["train_mean"],dtype=np.float32)
        std=np.asarray(weights["train_std"],dtype=np.float32)
        for seed in range(52000,52012):
            assert assert_matched_pulse_sham_spec(seed)
            for mode_index,mode in enumerate(("pulse","mild","sham")):
                x,y,r=rollout(model,mean,std,seed,mode)
                obs.append(x);labels.append(y)
                offsets.append(offsets[-1]+len(x))
                seeds.append(seed);modes.append(mode_index);sources.append(variant_index)
                metadata.append({"variant":variant,**r})
            print("COLLECT_D6",variant,seed,"episodes",len(metadata),flush=True)
    x=np.concatenate(obs).astype(np.float32)
    y=np.concatenate(labels).astype(np.float32)
    assert len(metadata)==108 and y.shape==(len(x),2)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.all((y>=0)&(y<=1))
    assert len(x)>5000
    OUT.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT,observations=x,actions=y,
        episode_offsets=np.asarray(offsets,dtype=np.int64),
        episode_seeds=np.asarray(seeds,dtype=np.int32),
        episode_modes=np.asarray(modes,dtype=np.int8),
        source_policy=np.asarray(sources,dtype=np.int8))
    report={
        "stage":"D6 one-pass privileged teacher labels on states visited by frozen D4 A/B/C",
        "dataset_id":D4,
        "source_training":"frozen D4 fullseq80 seed42 A/B/C",
        "training_seed_range":[52000,52011],
        "episode_count":108,"labelled_steps":len(x),
        "privileged_input_only_for_teacher_label":True,
        "student_observation":"untouched MetaDrive 259D",
        "archive_sha256":hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "original_D4_teacher_episodes_for_training":72,
        "labelled_student_trajectory_episodes":metadata,
    }
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("D6_LABELS_READY",json.dumps({k:v for k,v in report.items()
          if k!="labelled_student_trajectory_episodes"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
