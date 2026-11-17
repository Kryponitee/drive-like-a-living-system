#!/usr/bin/env python3
"""D6b: 受损网络的跨模块信息接管检验；按训练seed和交通密度组织配对估计。"""

import json
from pathlib import Path

import numpy as np


ROOT=Path("/mnt/datasets/lxb/dls/reports")
SEEDS=(42,43,44)
DENSITIES=(0.0,0.10,0.25)
MODES=("local","cross","cross_blocked")
LESIONS=(-1,*range(8))
SCENES=list(range(40000,40024))


def summary_mode(rows):
    healthy=[r for r in rows if r["lesion"]==-1]
    damaged=[r for r in rows if r["lesion"]!=-1]
    return {
        "healthy_success_of_24":sum(int(r["success"]) for r in healthy),
        "damaged_success_of_192":sum(int(r["success"]) for r in damaged),
        "healthy_reward":float(np.mean([r["reward"] for r in healthy])),
        "damaged_reward":float(np.mean([r["reward"] for r in damaged])),
        "healthy_success_fraction":float(np.mean([r["success"] for r in healthy])),
        "damaged_success_fraction":float(np.mean([r["success"] for r in damaged])),
        "relative_damaged_success_drop_pp":float(100*(np.mean([r["success"] for r in damaged])-np.mean([r["success"] for r in healthy]))),
        "mean_abs_message":float(np.mean([np.mean(r["mean_abs_unit_messages"]) for r in rows])),
        "mean_abs_unit_activity":float(np.mean([np.mean(r["mean_abs_unit_activity"]) for r in rows])),
    }


def main():
    summary={
        "stage":"D6b 8 local sensory units with output lesion training: cross communication takeover",
        "trained_seeds":list(SEEDS),"densities":list(DENSITIES),
        "new_scene_seeds":SCENES,"eval_conditions":list(MODES),
        "lesions":list(LESIONS),"num_episode_records":5832,
        "paired_by_training_seed_scene_density_and_lesion":True,
        "not_independent_5832_episodes":True,
        "training":{},"results":{},
    }
    for seed in SEEDS:
        summary["training"][str(seed)]={}
        for arm in ("local","cross"):
            p=ROOT/f"d6b/train-{arm}-seed{seed}.json"
            r=json.loads(p.read_text())
            assert r["seed"]==seed and r["arm"]==arm
            assert r["params"]==114880 and r["optimizer_updates"]==408
            assert r["train_steps"]==91717 and r["val_steps"]==9822
            summary["training"][str(seed)][arm]={
                "healthy_val_huber":r["healthy_val_huber"],
                "mean_single_lesion_val_huber":float(np.mean(r["single_lesion_val_huber"])),
                "blocked_healthy_val_huber":r["blocked_healthy_val_huber"],
                "blocked_mean_single_lesion_val_huber":(
                    float(np.mean(r["blocked_lesion_val_huber"]))
                    if r["blocked_lesion_val_huber"] is not None else None
                ),
            }
        summary["results"][str(seed)]={}
        for density in DENSITIES:
            allrows=[]
            for shard in range(6):
                p=ROOT/f"d6b/eval/seed{seed}-density{density:.2f}-shard{shard}.json"
                r=json.loads(p.read_text())
                assert r["seed"]==seed and r["density"]==density
                assert r["scene_seeds"]==SCENES[shard*4:(shard+1)*4]
                assert r["num_episodes"]==108
                assert r["lesions"]==list(LESIONS) and r["modes"]==list(MODES)
                allrows.extend(r["episodes"])
            assert len(allrows)==24*9*3
            by_mode={}
            metrics={}
            for mode in MODES:
                group=[x for x in allrows if x["mode"]==mode]
                assert len(group)==24*9
                for lesion in LESIONS:
                    subgroup=[x for x in group if x["lesion"]==lesion]
                    assert [x["scene_seed"] for x in subgroup]==SCENES
                by_mode[mode]=group
                metrics[mode]=summary_mode(group)
            # 事前定义的“损伤相对健康损失差分中的差分”。
            damaged_did=(
                metrics["cross"]["relative_damaged_success_drop_pp"]-
                metrics["local"]["relative_damaged_success_drop_pp"]
            )
            blocked_did=(
                metrics["cross"]["relative_damaged_success_drop_pp"]-
                metrics["cross_blocked"]["relative_damaged_success_drop_pp"]
            )
            differences={}
            for first,second,name in (("cross","local","cross_minus_local"),
                                      ("cross","cross_blocked","cross_minus_blocked")):
                differences[name]={
                    "healthy_success_difference_of_24":
                        metrics[first]["healthy_success_of_24"]-metrics[second]["healthy_success_of_24"],
                    "damaged_success_difference_of_192":
                        metrics[first]["damaged_success_of_192"]-metrics[second]["damaged_success_of_192"],
                    "mean_lesion_reward_difference":
                        metrics[first]["damaged_reward"]-metrics[second]["damaged_reward"],
                    "relative_lesion_success_DiD_pp":
                        metrics[first]["relative_damaged_success_drop_pp"]-
                        metrics[second]["relative_damaged_success_drop_pp"],
                }
            summary["results"][str(seed)][f"{density:.2f}"]={
                "per_mode":metrics,"differences":differences,
                "cross_local_relative_damage_advantage_pp":damaged_did,
                "cross_vs_blocked_relative_damage_advantage_pp":blocked_did,
            }
            print(
                f"D6b seed{seed} density{density:.2f}: healthy L/X/B="
                f"{metrics['local']['healthy_success_of_24']}/"
                f"{metrics['cross']['healthy_success_of_24']}/"
                f"{metrics['cross_blocked']['healthy_success_of_24']}; "
                f"damaged L/X/B="
                f"{metrics['local']['damaged_success_of_192']}/"
                f"{metrics['cross']['damaged_success_of_192']}/"
                f"{metrics['cross_blocked']['damaged_success_of_192']}; "
                f"relative damage X−L={damaged_did:+.2f}pp, X−B={blocked_did:+.2f}pp",
                flush=True,
            )
    totals={}
    for mode in MODES:
        totals[mode]={
            "healthy_success_of_216":sum(
                summary["results"][str(seed)][f"{d:.2f}"]["per_mode"][mode]["healthy_success_of_24"]
                for seed in SEEDS for d in DENSITIES),
            "damaged_success_of_1728":sum(
                summary["results"][str(seed)][f"{d:.2f}"]["per_mode"][mode]["damaged_success_of_192"]
                for seed in SEEDS for d in DENSITIES),
        }
    summary["descriptive_totals_across_reused_scene_seeds"]=totals
    out=ROOT/"d6b/d6b-summary.json"
    assert not out.exists()
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("D6b complete:",json.dumps(totals),flush=True)


if __name__=="__main__":
    main()
