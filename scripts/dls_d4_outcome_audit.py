#!/usr/bin/env python3
"""Read-only D4 outcome and braking feasibility diagnostics for stored test trajectories."""
import argparse,json,os
from pathlib import Path
import numpy as np

ROOT=Path("/mnt/datasets/lxb/dls")
def average(rows,key):
    return float(np.mean([r[key] for r in rows])) if rows else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-id",required=True)
    p.add_argument("--eval-subdir",default="closed-loop")
    p.add_argument("--output-name",default="d4-outcome-meaning-audit.json")
    a=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    folder=ROOT/"reports"/a.data_id
    out=folder/a.output_name
    assert not out.exists()
    groups={}
    for variant in ("A","B","C"):
        rows=[]
        for seed in (42,43,44):
            report=json.loads((folder/a.eval_subdir/f"{variant}-seed{seed}.json").read_text())
            assert report["seed"]==seed and report["variant"]==variant
            rows.extend(report["episodes"])
        for condition in ("sham","pulse","pulse_no_peer"):
            if condition=="pulse_no_peer" and variant!="C":continue
            sample=[r for r in rows if
                    ((r["mode"]=="sham" and not r["block_peer_after_event"])
                    if condition=="sham" else
                    (r["mode"]=="pulse" and (r["block_peer_after_event"]==(condition=="pulse_no_peer"))))]
            assert len(sample)==36
            for r in sample:
                if r["event_reached"]:
                    assert r["steps"]>r["event_step"]
            event_rows=[]
            for r in sample:
                if not r["event_reached"]:continue
                moment=r["time_series"][r["event_step"]]
                event_rows.append({
                    "safe_success":bool(r["success"] and not r["collision"] and not r["out_of_road"]),
                    "collision":r["collision"],
                    "event_surface_gap_m":moment["surface_gap_m"],
                    "event_closing_speed_mps":moment["ego_speed_mps"]-moment["lead_speed_mps"],
                    "event_ego_speed_mps":moment["ego_speed_mps"],
                    "event_lead_speed_mps":moment["lead_speed_mps"],
                    "min_surface_gap_m":r["min_surface_gap_m"],
                })
            groups[variant+"-"+condition]={
                "n":len(sample),
                "arrived":sum(r["success"] for r in sample),
                "crashed":sum(r["collision"] for r in sample),
                "arrived_and_crashed":sum(r["success"] and r["collision"] for r in sample),
                "safe_arrival":sum(r["success"] and not r["collision"] and not r["out_of_road"] for r in sample),
                "event_seen":sum(r["event_reached"] for r in sample),
                "min_gap_negative":sum(r["min_surface_gap_m"]<0 for r in sample),
                "mean_event_surface_gap_m":average(event_rows,"event_surface_gap_m"),
                "mean_event_ego_speed_mps":average(event_rows,"event_ego_speed_mps"),
                "mean_event_lead_speed_mps":average(event_rows,"event_lead_speed_mps"),
                "mean_event_closing_speed_mps":average(event_rows,"event_closing_speed_mps"),
                "crash_mean_event_gap_m":average([r for r in event_rows if r["collision"]],"event_surface_gap_m"),
                "safe_mean_event_gap_m":average([r for r in event_rows if r["safe_success"]],"event_surface_gap_m"),
            }
    doc={
        "data_id":a.data_id,
        "stage":"Read-only D4 outcome semantics and physical gap/speed at brake trigger",
        "definition":"safe arrival = arrive_dest AND not crash_vehicle/object/building AND not out_of_road",
        "results":groups,
        "note":"MetaDrive arrival and collision flags can overlap; never treat both labels as disjoint without checking.",
    }
    out.write_text(json.dumps(doc,indent=2,ensure_ascii=False))
    print("OUTCOME_AUDIT",json.dumps(doc,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
