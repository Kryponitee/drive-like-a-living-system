#!/usr/bin/env python3
"""Aggregate the fully held-out, matched-scene D4 trial without erasing failures."""
import argparse, json, os
from pathlib import Path
import numpy as np

ROOT=Path("/mnt/datasets/lxb/dls")

def mean(items):
    return float(np.mean(items)) if items else None

def paired_response(sham,pulse,variant):
    assert sham["seed"]==pulse["seed"] and sham["event_step"]==pulse["event_step"]
    t0=sham["event_step"]
    stop=min(t0+22,sham["steps"],pulse["steps"])
    if stop<=t0+7:
        return None
    times=range(t0+7,stop)
    changes=[
        [pulse["time_series"][t][name]-sham["time_series"][t][name] for t in times]
        for name in ("throttle","brake","ego_speed_mps","lead_speed_mps")]
    summary={
        "throttle_pulse_minus_sham":mean(changes[0]),
        "brake_pulse_minus_sham":mean(changes[1]),
        "ego_speed_pulse_minus_sham_mps":mean(changes[2]),
        "lead_speed_pulse_minus_sham_mps":mean(changes[3]),
        "n_compare_steps":len(times),
    }
    if variant in ("B","C"):
        delta=np.stack([
            np.asarray(pulse["time_series"][t]["module_logits"],dtype=np.float64)
            -np.asarray(sham["time_series"][t]["module_logits"],dtype=np.float64)
            for t in times])
        assert delta.shape==(len(times),8,2)
        num=np.abs(delta.sum(axis=1)).sum(axis=0)
        den=np.abs(delta).sum(axis=(0,1))
        summary["module_cancellation_numerator"]=num.tolist()
        summary["module_cancellation_denominator"]=den.tolist()
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-id",required=True)
    a=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    directory=ROOT/"reports"/a.data_id
    out=directory/"d4-fullseq80-overview.json"
    assert not out.exists()
    train_meta=json.loads((directory/"teacher-train.json").read_text())
    val_meta=json.loads((directory/"teacher-val.json").read_text())
    oracle=json.loads((directory/"holdout-privileged-feasibility.json").read_text())
    assert train_meta["n_episodes"]==72 and val_meta["n_episodes"]==24
    assert train_meta["seed_range"]==[51000,51023]
    assert val_meta["seed_range"]==[61000,61007]
    assert oracle["scenes"]==24
    by_variant={}
    for variant in ("A","B","C"):
        training=[]
        episodes=[]
        paired=[]
        pair_diffs=[]
        for seed in (42,43,44):
            training_row=json.loads((directory/"full-seq80"/f"train-{variant}-seed{seed}.json").read_text())
            assert training_row["data_id"]==a.data_id
            assert training_row["variant"]==variant and training_row["seed"]==seed
            assert training_row["epochs"]==80
            assert training_row["train_episodes"]==72 and training_row["val_episodes"]==24
            training.append(training_row)
            evaluation=json.loads((directory/"closed-loop-fullseq80"/f"{variant}-seed{seed}.json").read_text())
            assert evaluation["data_id"]==a.data_id
            assert evaluation["variant"]==variant and evaluation["seed"]==seed
            assert evaluation["test_scene_seeds"]==list(range(71000,71012))
            assert len(evaluation["episodes"])==(36 if variant=="C" else 24)
            episodes.extend(evaluation["episodes"])
            for scene in range(71000,71012):
                sham=next(r for r in evaluation["episodes"]
                          if r["seed"]==scene and r["mode"]=="sham")
                pulse=next(r for r in evaluation["episodes"]
                           if r["seed"]==scene and r["mode"]=="pulse"
                           and not r["block_peer_after_event"])
                resp=paired_response(sham,pulse,variant)
                pair_diffs.append({
                    "model_seed":seed,"scene_seed":scene,
                    "event_step":pulse["event_step"],
                    "exposed_both":sham["event_reached"] and pulse["event_reached"],
                    "sham_collision":sham["collision"],
                    "pulse_collision":pulse["collision"],
                    "sham_success":sham["success"],
                    "pulse_success":pulse["success"],
                    "response":resp,
                })
        assert len(pair_diffs)==36
        split={}
        for mode,block in (("sham",False),("pulse",False),("pulse",True)):
            if block and variant!="C":
                continue
            group=[r for r in episodes if r["mode"]==mode and r["block_peer_after_event"]==block]
            assert len(group)==36
            split[mode+("-no_peer" if block else "")]={
                "episodes":len(group),
                "success_rate":mean([r["success"] for r in group]),
                "collision_rate":mean([r["collision"] for r in group]),
                "event_exposure_rate":mean([r["event_reached"] for r in group]),
                "mean_progress_m":mean([r["progress_m"] for r in group]),
                "mean_steps":mean([r["steps"] for r in group]),
                "mean_min_surface_gap_m":mean([r["min_surface_gap_m"] for r in group]),
                "mean_brake_fraction_after_event":mean([
                    r["brake_fraction_after_event"] for r in group
                    if r["brake_fraction_after_event"] is not None]),
                "mean_coactive_fraction":mean([r["mean_coactive_fraction"] for r in group]),
            }
        # Real paired treatment; do not count episodes that ended before the intervention.
        eligible=[r["response"] for r in pair_diffs if r["response"] is not None]
        change={
            "n_pairs":len(pair_diffs),"n_eligible":len(eligible),
            "mean_change_brake":mean([r["brake_pulse_minus_sham"] for r in eligible]),
            "mean_change_throttle":mean([r["throttle_pulse_minus_sham"] for r in eligible]),
            "mean_change_lead_speed_mps":mean([r["lead_speed_pulse_minus_sham_mps"] for r in eligible]),
        }
        if variant in ("B","C") and eligible:
            numerator=np.sum([r["module_cancellation_numerator"] for r in eligible],axis=0)
            denominator=np.sum([r["module_cancellation_denominator"] for r in eligible],axis=0)
            change["module_readout_cancellation_fraction"]=(
                1-numerator/np.maximum(denominator,1e-12)).tolist()
        by_variant[variant]={
            "training":{
                "val_huber_mean":mean([r["best_val_huber"] for r in training]),
                "val_huber_by_model_seed":{str(r["seed"]):r["best_val_huber"] for r in training},
                "best_epoch_by_seed":{str(r["seed"]):r["best_epoch"] for r in training},
            },
            "closed_loop":split,
            "paired_pulse_response":change,
            "paired_details":pair_diffs,
        }
    report={
        "stage":"DLS D4 80-epoch full-episode recurrent training, same data and A/B/C architectures",
        "data_id":a.data_id,
        "design":"same D4 data and heldout scenes, 80 epochs; teacher-forced GRU unroll across full episode instead of 40-step resets",
        "heldout_feasibility":{"success":oracle["success"],"collision":oracle["collisions"],
                               "event_exposure":oracle["event_exposure"],"episodes":oracle["scenes"]},
        "total_model_closed_loop_episodes":252,
        "comparison_is_same_student_observation":True,
        "teacher_privileged_but_not_student":True,
        "no_hand_specified_module_responsibilities":True,
        "results":by_variant,
        "limitations":[
            "supervised BC with privileged ACC teacher; difficult end-to-end generalization is not guaranteed",
            "same D4 observation sensor but teacher uses extra physics state, only training targets",
            "12 heldout scenario seeds x 3 training seeds, not 36 independent architectures",
            "readout cancellation is coordinate-relative, not proof of neural motor synergy",
            "any collisions or lack of post-event exposure must remain visible",
        ],
    }
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    brief={k:{
        "val":v["training"]["val_huber_mean"],
        "sham":v["closed_loop"]["sham"],
        "pulse":v["closed_loop"]["pulse"],
        "pulse_no_peer":v["closed_loop"].get("pulse-no_peer"),
        "paired":v["paired_pulse_response"],
    } for k,v in by_variant.items()}
    print("D4_AGGREGATE",json.dumps({"data_id":a.data_id,
           "heldout_oracle":report["heldout_feasibility"],
           "overview":brief},indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
