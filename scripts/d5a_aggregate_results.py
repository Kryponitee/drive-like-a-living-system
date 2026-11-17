#!/usr/bin/env python3
"""D5a: 在训练seed × traffic density层级做配对统计，不伪造独立样本。"""

import json
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/datasets/lxb/dls/reports")
SCENES = list(range(36000, 36024))
SEEDS = (42, 43, 44)
DENSITIES = (0.0, 0.1, 0.25)
ARMS = ("clean", "robust")
LESIONS = (-1, *range(8))


def main():
    summary = {
        "stage": "D5a redundant motor units: clean vs output-lesion BC training",
        "scientific_scope": "learnable redundancy / lesion tolerance, NOT proven online motor-synergy compensation",
        "train_seeds": list(SEEDS), "scene_seeds": SCENES,
        "traffic_densities": list(DENSITIES), "arms": list(ARMS),
        "lesion_units": list(LESIONS),
        "new_closed_loop_episodes": 3 * 3 * 2 * 24 * 9,
        "healthy_episodes": 3 * 3 * 2 * 24,
        "lesioned_episodes": 3 * 3 * 2 * 24 * 8,
        "training": {}, "comparisons": {},
        "no_scene_pseudoreplication": True,
        "old_C_checkpoints_unchanged": True,
    }
    for seed in SEEDS:
        summary["training"][str(seed)] = {}
        for arm in ARMS:
            p = ROOT / f"d5a/train-{arm}-seed{seed}.json"
            record = json.loads(p.read_text())
            assert record["arm"] == arm and record["seed"] == seed
            assert record["epochs"] == 8 and record["batch_size"] == 48
            assert record["optimizer_updates"] == 408
            assert record["train_steps"] == 91717 and record["val_steps"] == 9822
            assert record["params"] == 115568
            assert record["decoder_rank"] == 2
            summary["training"][str(seed)][arm] = {
                "healthy_val_huber": record["healthy_val_huber"],
                "single_lesion_val_huber": record["single_lesion_val_huber"],
                "decoder_rank": record["decoder_rank"],
                "decoder_columns": record["decoder_columns"],
            }
        summary["comparisons"][str(seed)] = {}
        for density in DENSITIES:
            series = {}
            for arm in ARMS:
                all_eps = []
                for shard in range(3):
                    p = ROOT / f"d5a/eval/{arm}-seed{seed}-density{density:.2f}-shard{shard}.json"
                    data = json.loads(p.read_text())
                    assert data["arm"] == arm and data["seed"] == seed
                    assert data["density"] == density and data["num_episodes"] == 72
                    assert data["scene_seeds"] == SCENES[shard*8:(shard+1)*8]
                    all_eps.extend(data["episodes"])
                assert len(all_eps) == 216
                for lesion in LESIONS:
                    group = [ep for ep in all_eps if ep["lesion_unit"] == lesion]
                    assert [ep["scene_seed"] for ep in group] == SCENES
                series[arm] = all_eps
            res = {"healthy": {}, "each_lesion": {}, "absolute_masked": {},
                   "healthy_abs_output_contributions": {}}
            healthy_score = {}
            lesion_succ = {}
            lesion_reward = {}
            for arm in ARMS:
                healthy = [ep for ep in series[arm] if ep["lesion_unit"] == -1]
                healthy_score[arm] = sum(int(ep["success"]) for ep in healthy)
                res["healthy"][arm] = {
                    "success_of_24": healthy_score[arm],
                    "mean_reward": float(np.mean([ep["reward"] for ep in healthy])),
                    "mean_completion": float(np.mean([ep["route_completion"] for ep in healthy])),
                }
                contributions = np.array([ep["unit_abs_contrib_pre_tanh"] for ep in healthy], dtype=float)
                assert contributions.shape == (24, 8, 2)
                res["healthy_abs_output_contributions"][arm] = contributions.mean(axis=0).tolist()
                lesion_succ[arm] = {}
                lesion_reward[arm] = {}
                for lesion in range(8):
                    eps = [ep for ep in series[arm] if ep["lesion_unit"] == lesion]
                    lesion_succ[arm][lesion] = sum(int(ep["success"]) for ep in eps)
                    lesion_reward[arm][lesion] = float(np.mean([ep["reward"] for ep in eps]))
                res["absolute_masked"][arm] = {
                    "success_of_192": int(sum(lesion_succ[arm].values())),
                    "mean_success_rate_over_8_masks": float(sum(lesion_succ[arm].values()) / 192),
                    "mean_reward_over_8_masks": float(np.mean(list(lesion_reward[arm].values()))),
                    "damage_vs_healthy_success_pp": float(
                        100 * (sum(lesion_succ[arm].values()) / 192 - healthy_score[arm] / 24)
                    ),
                }
            effects = []
            for lesion in range(8):
                diff_damage_success = (
                    (lesion_succ["robust"][lesion] - healthy_score["robust"])
                    - (lesion_succ["clean"][lesion] - healthy_score["clean"])
                )
                res["each_lesion"][str(lesion)] = {
                    "clean_success_of_24": lesion_succ["clean"][lesion],
                    "robust_success_of_24": lesion_succ["robust"][lesion],
                    "clean_reward": lesion_reward["clean"][lesion],
                    "robust_reward": lesion_reward["robust"][lesion],
                    "damage_DiD_success_pp": 100 * diff_damage_success / 24,
                }
                effects.append(100 * diff_damage_success / 24)
            res["mean_single_lesion_damage_DiD_success_pp"] = float(np.mean(effects))
            res["num_lesions_with_positive_damage_DiD"] = sum(x > 0 for x in effects)
            summary["comparisons"][str(seed)][f"{density:.2f}"] = res
            print(
                f"D5a seed{seed} density={density:.2f} healthy clean/robust "
                f"{healthy_score['clean']}/{healthy_score['robust']}; "
                f"damaged clean/robust "
                f"{res['absolute_masked']['clean']['success_of_192']}/"
                f"{res['absolute_masked']['robust']['success_of_192']}; "
                f"mean masked damage DiD={np.mean(effects):+.2f} pp",
                flush=True,
            )
    out = ROOT / "d5a/d5a-summary.json"
    assert not out.exists()
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"D5a aggregation done: {summary['new_closed_loop_episodes']} episodes; {out}")


if __name__ == "__main__":
    main()
