#!/usr/bin/env python3
"""D5: collect physical labels aligned to ordinary 259D MetaDrive observations.

Replays D4 teacher exactly, verifies equality with saved D4 teacher observations,
then additionally records post-training on-policy observations. Labels are
physics only for diagnostic readout and are NOT used by original driving models.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_d4_following_scene import (
    collect_teacher_episode,scene_spec,assert_matched_pulse_sham_spec,make_env,
    D4TrafficManager,physical_gap,MAX_STEPS)
from dls_dual_pedal_adapter import install_dual_pedal_adapter
from dls_longitudinal_motor import model_info

ROOT=Path("/mnt/datasets/lxb/dls")
DATA_ID="d4-dynamic-follow-v1-44481405397"
OUT=ROOT/"reports"/DATA_ID/"d5-observability-v1"
MODES=("pulse","mild","sham")


def package(episodes):
    obs=[];targets=[];times=[];events=[];ego_speeds=[];offsets=[0];meta=[]
    for x,physics,rec in episodes:
        assert x.ndim==2 and x.shape[1]==259
        assert physics.shape==(len(x),4)
        obs.append(x.astype(np.float32))
        # gap (m), longitudinal closing speed (m/s).
        targets.append(np.column_stack([physics[:,0],physics[:,1]-physics[:,2]]).astype(np.float32))
        ego_speeds.append(physics[:,1].astype(np.float32))
        times.append(np.arange(len(x),dtype=np.float32))
        events.append(np.full(len(x),rec["brake_start_step"],dtype=np.float32))
        offsets.append(offsets[-1]+len(x))
        meta.append(rec)
    return {
        "observations":np.concatenate(obs),
        "labels":np.concatenate(targets),
        "times":np.concatenate(times),
        "event_steps":np.concatenate(events),
        "ego_speeds":np.concatenate(ego_speeds),
        "episode_modes":np.asarray([MODES.index(m["mode"]) for m in meta],dtype=np.int8),
        "episode_seeds":np.asarray([m["seed"] for m in meta],dtype=np.int32),
        "episode_offsets":np.asarray(offsets,dtype=np.int64),
        "meta":meta,
    }


def collect_teacher(split):
    ranges={"train":range(51000,51024),"val":range(61000,61008),
            "test":range(71000,71012)}
    assert split in ranges
    manifest=json.loads((ROOT/"reports"/DATA_ID/f"teacher-{split}.json").read_text()) if split!="test" else None
    archive=ROOT/"demonstrations"/DATA_ID/f"{split}.npz" if split!="test" else None
    original=None
    if archive is not None:
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest["sha256"]
        with np.load(archive,allow_pickle=False) as d:
            original=(d["observations"].copy(),d["episode_offsets"].copy(),
                      d["episode_seeds"].copy(),d["episode_modes"].copy())
    episodes=[]
    indices=0;max_deviation=0.
    for seed in ranges[split]:
        assert assert_matched_pulse_sham_spec(seed)
        for mode in MODES:
            obs,acts,physics,rec=collect_teacher_episode(seed,mode)
            if original is not None:
                a=int(original[1][indices]);b=int(original[1][indices+1])
                assert b-a==len(obs)
                assert int(original[2][indices])==seed
                assert int(original[3][indices])==MODES.index(mode)
                deviation=float(np.max(np.abs(original[0][a:b]-obs)))
                max_deviation=max(max_deviation,deviation)
                assert deviation<1e-4,("teacher replay mismatch",split,seed,mode,deviation)
                indices+=1
            episodes.append((obs,physics,rec))
        print("TEACHER_PHYSICS",split,seed,"episodes",len(episodes),flush=True)
    p=package(episodes)
    return p,{"episodes":len(episodes),"steps":len(p["labels"]),
              "max_original_observation_replay_abs_error":max_deviation,
              "modes":list(MODES),
              "seed_start":min(ranges[split]),"seed_end":max(ranges[split])}


def collect_onpolicy(variant):
    assert variant in ("A","B","C")
    ckpt=ROOT/"checkpoints"/DATA_ID/"full-seq80"/f"{variant}-seed42.pt"
    weights=torch.load(ckpt,map_location="cpu",weights_only=False)
    assert weights["variant"]==variant and weights["seed"]==42
    model,_=model_info(variant)
    model.load_state_dict(weights["state_dict"],strict=True)
    model.eval()
    mean=np.asarray(weights["train_mean"],dtype=np.float32)
    std=np.asarray(weights["train_std"],dtype=np.float32)
    episodes=[]
    for seed in range(71000,71012):
        mode="pulse"
        assert assert_matched_pulse_sham_spec(seed)
        spec=scene_spec(seed,mode)
        env=make_env(seed)
        xs=[];states=[]
        try:
            obs,_=env.reset(seed=seed)
            ego=env.agent
            install_dual_pedal_adapter(ego,steering_controller=lambda _:0.)
            mgr=env.unwrapped.engine.traffic_manager
            assert isinstance(mgr,D4TrafficManager)
            mgr.attach(ego,spec)
            hidden=model.initial_state(1,torch.device("cpu"))
            info={}
            with torch.no_grad():
                for t in range(MAX_STEPS):
                    raw=np.asarray(obs,dtype=np.float32)
                    gap=physical_gap(ego,mgr.lead)[1]
                    states.append((gap,float(ego.speed)/3.6,
                                   float(mgr.lead.speed)/3.6,float(mgr.phase)))
                    xs.append(raw.copy())
                    x=np.clip((raw-mean)/std,-10.,10.)
                    pedal,hidden=model.step(torch.from_numpy(x.copy())[None],hidden,t)
                    obs,reward,done,truncated,info=env.step(pedal[0].numpy().astype(np.float32))
                    if done or truncated:break
            rec={
                **spec,
                "steps":len(xs),
                "collision":bool(info.get("crash_vehicle",False)),
                "success":bool(info.get("arrive_dest",False)),
                "variant":variant,
                "source":"fullseq80 on-policy model-seed42",
            }
            episodes.append((np.stack(xs),np.asarray(states,dtype=np.float32),rec))
            print("ONPOLICY",variant,seed,"steps",len(xs),flush=True)
        finally:
            env.close()
    p=package(episodes)
    return p,{"episodes":len(episodes),"steps":len(p["labels"]),
              "variant":variant,"training_seed":42,"test_scenes":[71000,71011]}


def save(path,records):
    assert not path.exists()
    values={k:v for k,v in records.items() if k!="meta"}
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**values)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("train","val","test","onpolicy-A","onpolicy-B","onpolicy-C"),required=True)
    a=parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    dest=OUT/f"{a.stage}.npz"
    report=OUT/f"{a.stage}.json"
    assert not dest.exists() and not report.exists()
    if a.stage.startswith("onpolicy-"):
        record,summary=collect_onpolicy(a.stage[-1])
    else:
        record,summary=collect_teacher(a.stage)
    h=save(dest,record)
    report.write_text(json.dumps({"stage":a.stage,"sha256":h,**summary},
                                 indent=2,ensure_ascii=False))
    print("D5_DATA_DONE",json.dumps({"stage":a.stage,"sha256":h,**summary},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
