#!/usr/bin/env python3
"""Heldout D4 feasibility oracle. Never used for student training or policy input."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_d4_following_scene import collect_teacher_episode

ROOT=Path("/mnt/datasets/lxb/dls")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-id",required=True)
    a=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    out=ROOT/"reports"/a.data_id/"holdout-privileged-feasibility.json"
    assert not out.exists()
    records=[]
    for seed in range(71000,71012):
        for mode in ("pulse","sham"):
            _,_,_,meta=collect_teacher_episode(seed,mode)
            records.append(meta)
            print("HOLDOUT_ORACLE",seed,mode,meta["success"],
                  meta["collision"],meta["min_surface_gap_m"],flush=True)
    result={
        "stage":"D4 heldout scene physical feasibility with privileged reference",
        "not_student_input":True,
        "scenes":24,
        "success":sum(r["success"] for r in records),
        "collisions":sum(r["collision"] for r in records),
        "event_exposure":sum(r["event_was_reached"] for r in records),
        "min_surface_gap_m":min(r["min_surface_gap_m"] for r in records),
        "records":records,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    assert result["success"]>=23 and result["collisions"]==0, result
    assert result["event_exposure"]>=23, result
    print("HOLDOUT_FEASIBLE",json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)

if __name__=="__main__":
    main()
