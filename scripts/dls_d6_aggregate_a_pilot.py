#!/usr/bin/env python3
"""D6 A-only shared on-policy state coverage versus pure data replay audit."""
import json,os
from pathlib import Path
import numpy as np

ROOT=Path("/mnt/datasets/lxb/dls")
DATA_ID="d4-dynamic-follow-v1-44481405397"
BASE=ROOT/"reports"/DATA_ID/"d6-a-pilot"
OUT=BASE/"coverage-vs-replay-summary.json"


def safe(x):
    return bool(x["success"] and not x["collision"] and not x["out_of_road"])


def summary():
    assert os.path.ismount("/mnt/datasets") and not OUT.exists()
    arms={}
    for arm in ("replay","onpolicy"):
        training=[];evaluations=[];paired=[]
        for seed in (42,43,44):
            train=json.loads((BASE/f"train-{arm}-seed{seed}.json").read_text())
            eval_data=json.loads((BASE/"evaluation"/f"{arm}-seed{seed}.json").read_text())
            assert train["seed"]==seed and train["arm"]==arm
            assert train["train_episodes"]==180 and train["optimizer_updates_per_epoch"]==12
            assert train["epochs"]==80
            assert eval_data["seed"]==seed and eval_data["arm"]==arm
            assert eval_data["matched_pulse_sham_prefixes"]==12
            assert len(eval_data["episodes"])==24
            training.append(train)
            evaluations.extend([{**e,"training_seed":seed} for e in eval_data["episodes"]])
            for scene in range(71000,71012):
                sham=next(z for z in eval_data["episodes"]
                    if z["seed"]==scene and z["mode"]=="sham")
                pulse=next(z for z in eval_data["episodes"]
                    if z["seed"]==scene and z["mode"]=="pulse")
                t0=pulse["event_step"]
                t1=min(t0+22,sham["steps"],pulse["steps"])
                post=range(t0+7,t1)
                diff=float(np.mean([
                    pulse["time_series"][t]["brake"]-sham["time_series"][t]["brake"]
                    for t in post])) if len(post)>=5 else None
                paired.append({
                    "training_seed":seed,"scene":scene,
                    "sham_safe":safe(sham),"pulse_safe":safe(pulse),
                    "pulse_collision":pulse["collision"],
                    "pulse_early_exposed":pulse["event_reached"],
                    "pulse_minus_sham_brake_7_22":diff,
                })
        d={}
        for mode in ("sham","pulse"):
            cases=[r for r in evaluations if r["mode"]==mode]
            assert len(cases)==36
            d[mode]={
                "n":len(cases),
                "safe_arrivals":sum(safe(x) for x in cases),
                "collisions":sum(x["collision"] for x in cases),
                "event_exposure":sum(x["event_reached"] for x in cases),
                "mean_steps":float(np.mean([x["steps"] for x in cases])),
                "mean_min_surface_gap_m":float(np.mean([x["min_surface_gap_m"] for x in cases])),
                "mean_coactive_fraction":float(np.mean([x["mean_coactive_fraction"] for x in cases])),
            }
        arms[arm]={
            "training":{
                "n_models":len(training),
                "n_episode_examples_each":180,
                "epochs_each":80,
                "updates_per_epoch_each":12,
                "input_steps_by_seed":{str(r["seed"]):r["train_time_steps"] for r in training},
                "val_huber_by_seed":{str(r["seed"]):r["best_val_huber"] for r in training},
            },
            "closed_loop":d,
            "mean_brake_response_pulse_minus_sham":float(np.mean([
                r["pulse_minus_sham_brake_7_22"] for r in paired
                if r["pulse_minus_sham_brake_7_22"] is not None])),
            "paired_results":paired,
        }
    replay={ (p["training_seed"],p["scene"]):p for p in arms["replay"]["paired_results"]}
    online={ (p["training_seed"],p["scene"]):p for p in arms["onpolicy"]["paired_results"]}
    assert len(replay)==len(online)==36 and set(replay)==set(online)
    improve=sum(not replay[k]["pulse_safe"] and online[k]["pulse_safe"] for k in replay)
    worsen=sum(replay[k]["pulse_safe"] and not online[k]["pulse_safe"] for k in replay)
    report={
        "stage":"D6 A-only same-model same-optimization replay vs online-label collection",
        "same_test_scenarios":True,
        "dataset":"d4-dynamic-follow-v1-44481405397",
        "input":"standard 259D observations and physical dual pedals; oracle physics only supplies action labels",
        "arms":arms,
        "paired_difference":{
            "total_matched_train-seed-by-physical-scene_cases":36,
            "onpolicy_saves_replay_failure":improve,
            "onpolicy_loses_replay_success":worsen,
            "net_extra_safe_arrivals":improve-worsen,
        },
        "limits":"A-only mechanism feasibility pilot; group effects need B/C and matched equal token budgets. Architectures and update counts match within this pilot, but sequence lengths and training token counts differ.",
    }
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("D6_A_PILOT_SUMMARY",json.dumps({
        "replay":{**arms["replay"]["closed_loop"],
                  "val":arms["replay"]["training"]["val_huber_by_seed"]},
        "onpolicy":{**arms["onpolicy"]["closed_loop"],
                    "val":arms["onpolicy"]["training"]["val_huber_by_seed"]},
        "paired":report["paired_difference"],
    },indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    summary()
