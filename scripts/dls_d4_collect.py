#!/usr/bin/env python3
"""D4 dynamic following teacher demonstration collection, disjoint scene splits."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_d4_following_scene import collect_teacher_episode,assert_matched_pulse_sham_spec

ROOT=Path("/mnt/datasets/lxb/dls")
SEEDS={"train":range(51000,51024),"val":range(61000,61008)}
MODES=("pulse","mild","sham")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=SEEDS,required=True)
    p.add_argument("--data-id",required=True)
    a=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert a.data_id.startswith("d4-dynamic-follow-v1-")
    out=ROOT/"demonstrations"/a.data_id/f"{a.split}.npz"
    report=ROOT/"reports"/a.data_id/f"teacher-{a.split}.json"
    assert not out.exists() and not report.exists()
    xs=[];ys=[];offsets=[0];seeds=[];modes=[];events=[];records=[]
    for seed in SEEDS[a.split]:
        assert assert_matched_pulse_sham_spec(seed)
        for mode in MODES:
            x,y,states,rec=collect_teacher_episode(seed,mode)
            assert rec["success"] and not rec["collision"] and not rec["out_of_road"],rec
            assert rec["min_surface_gap_m"]>0 and rec["event_was_reached"],rec
            assert x.shape==(len(y),259) and y.shape[1]==2
            xs.append(x);ys.append(y);offsets.append(offsets[-1]+len(y))
            seeds.append(seed);modes.append(MODES.index(mode))
            events.append(rec["brake_start_step"]);records.append(rec)
            print(f"COLLECT split={a.split} seed={seed} mode={mode} "
                  f"steps={len(y)} brake_fraction={rec['brake_fraction']:.4f}",flush=True)
    x=np.concatenate(xs).astype(np.float32)
    y=np.concatenate(ys).astype(np.float32)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert len(records)==len(SEEDS[a.split])*len(MODES)
    assert np.all((y>=0)&(y<=1))
    assert any(r["brake_fraction"]>.05 for r in records if r["mode"]=="pulse")
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,observations=x,actions=y,
        episode_offsets=np.asarray(offsets,dtype=np.int64),
        episode_seeds=np.asarray(seeds,dtype=np.int32),
        episode_modes=np.asarray(modes,dtype=np.int8),
        event_steps=np.asarray(events,dtype=np.int16))
    metadata={
        "stage":"D4 privileged ACC teacher demonstration (student never sees teacher physics)",
        "split":a.split,"data_id":a.data_id,"n_episodes":len(records),"n_steps":len(y),
        "archive":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
        "modes":list(MODES),"teacher_brake_fraction":float((y[:,1]>.1).mean()),
        "seed_range":[min(SEEDS[a.split]),max(SEEDS[a.split])],"episodes":records,
    }
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print("DONE",json.dumps({k:v for k,v in metadata.items() if k!="episodes"}),flush=True)

if __name__=="__main__":
    main()
