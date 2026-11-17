#!/usr/bin/env python3
"""D6: 按模型训练种子和交通密度汇总局部输入压力与跨模块通信的闭环因果干预。"""

import json
from pathlib import Path
import numpy as np


ROOT = Path("/mnt/datasets/lxb/dls/reports")
SEEDS = (42, 43, 44)
DENSITIES = (0.0, 0.10, 0.25)
MODES = ("local", "cross", "cross_blocked")
SCENES = list(range(38000, 38024))


def main():
    summary = {
        "stage": "D6: local information bottleneck cross vs self recurrent communication",
        "science_scope": "functional peer information necessity under imposed information bottleneck; not proof of natural motor synergy",
        "training_seeds": list(SEEDS),
        "scene_seeds": SCENES,
        "densities": list(DENSITIES),
        "modes": list(MODES),
        "new_episode_records": 3*3*24*3,
        "paired_at_scene_and_training_seed": True,
        "training": {},
        "results": {},
    }
    for seed in SEEDS:
        local = json.loads((ROOT/f"d6/train-local-seed{seed}.json").read_text())
        cross = json.loads((ROOT/f"d6/train-cross-seed{seed}.json").read_text())
        assert local["seed"] == cross["seed"] == seed
        assert local["arm"] == "local" and cross["arm"] == "cross"
        for x in (local,cross):
            assert x["params"] == 114880 and x["optimizer_updates"] == 408
            assert x["train_steps"] == 91717 and x["val_steps"] == 9822
            assert x["epochs"] == 8 and x["batch_size"] == 48
            assert x["projection_seed"] == 20261117
        summary["training"][str(seed)] = {
            "local_val_huber": local["healthy_val_huber"],
            "cross_val_huber": cross["healthy_val_huber"],
            "cross_blocked_val_huber": cross["cross_blocked_val_huber"],
            "params_each": 114880,
            "updates_each": 408,
        }
        summary["results"][str(seed)] = {}
        for density in DENSITIES:
            all_eps = []
            for shard in range(3):
                p = ROOT / f"d6/eval/seed{seed}-density{density:.2f}-shard{shard}.json"
                d = json.loads(p.read_text())
                assert d["seed"] == seed and d["density"] == density
                assert d["scene_seeds"] == SCENES[8*shard:8*(shard+1)]
                assert d["num_episodes"] == 24
                assert d["modes"] == list(MODES)
                all_eps.extend(d["episodes"])
            assert len(all_eps) == 72
            per = {}
            for mode in MODES:
                rows = [x for x in all_eps if x["mode"] == mode]
                assert [x["scene_seed"] for x in rows] == SCENES
                per[mode] = {
                    "success_of_24": sum(int(x["success"]) for x in rows),
                    "mean_reward": float(np.mean([x["reward"] for x in rows])),
                    "mean_completion": float(np.mean([x["route_completion"] for x in rows])),
                    "collisions": sum(int(x["collision"]) for x in rows),
                    "offroads": sum(int(x["offroad"]) for x in rows),
                    "mean_abs_message": float(np.mean([
                        np.mean(x["motor_mean_abs_messages"]) for x in rows
                    ])),
                    "mean_abs_activity": float(np.mean([
                        np.mean(x["motor_mean_abs_activity"]) for x in rows
                    ])),
                }
            by_mode = {
                mode: [ep for ep in all_eps if ep["mode"] == mode]
                for mode in MODES
            }
            diffs = {}
            for first, second, name in (
                ("cross","local","cross_minus_local"),
                ("cross","cross_blocked","cross_minus_blocked"),
            ):
                A,B = by_mode[first],by_mode[second]
                diffs[name] = {
                    "success_difference_of_24": per[first]["success_of_24"]-per[second]["success_of_24"],
                    "mean_reward_difference": per[first]["mean_reward"]-per[second]["mean_reward"],
                    "paired_reward_differences": [
                        float(a["reward"]-b["reward"]) for a,b in zip(A,B)
                    ],
                    "first_only_success": sum(bool(a["success"]) and not bool(b["success"]) for a,b in zip(A,B)),
                    "second_only_success": sum(bool(b["success"]) and not bool(a["success"]) for a,b in zip(A,B)),
                }
            summary["results"][str(seed)][f"{density:.2f}"] = {
                "per_mode": per,
                "paired_differences": diffs,
            }
            print(
                f"D6 seed{seed} density{density:.2f} local/cross/blocked "
                f"success={per['local']['success_of_24']}/"
                f"{per['cross']['success_of_24']}/"
                f"{per['cross_blocked']['success_of_24']}; "
                f"cross-local reward={diffs['cross_minus_local']['mean_reward_difference']:+.2f}; "
                f"cross-block reward={diffs['cross_minus_blocked']['mean_reward_difference']:+.2f}",
                flush=True,
            )
    summary["descriptive_totals_over_shared_scenes"] = {
        mode: sum(
            summary["results"][str(seed)][f"{density:.2f}"]["per_mode"][mode]["success_of_24"]
            for seed in SEEDS for density in DENSITIES
        )
        for mode in MODES
    }
    target = ROOT / "d6/d6-summary.json"
    assert not target.exists()
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("D6 completed:", json.dumps(summary["descriptive_totals_over_shared_scenes"]), flush=True)


if __name__ == "__main__":
    main()
