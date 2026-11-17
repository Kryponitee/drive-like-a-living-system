#!/usr/bin/env python3
"""D5b: 三种子×双训练条件，审计内部是否非平凡多维，保留原始完整谱。"""

import json
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/datasets/lxb/dls/reports/d5b")
SEEDS = (42,43,44)
ARMS = ("clean","robust")


def main():
    summary = {
        "stage": "D5b: is lesion robustness merely one common motor activity copied 8 times?",
        "scientific_limit": "PCA/correlation and output role profiles are observational; neither prove causal synergy nor feedback compensation",
        "seeds": list(SEEDS),
        "num_val_timesteps_per_model": 9822,
        "train_new_models": False,
        "models": {},
        "seed_level_descriptive_means": {},
    }
    for seed in SEEDS:
        p=ROOT/f"motor-diversity-seed{seed}.json"
        r=json.loads(p.read_text())
        assert r["seed"]==seed and r["num_val_timesteps"]==9822
        assert r["no_retraining"] is True
        summary["models"][str(seed)]={}
        for arm in ARMS:
            a=r["arms"][arm]
            assert a["decoder_2x8_rank"]==2
            assert len(a["raw_covariance_eigenvalues"])==8
            assert len(a["relative_abs_output_contribution_per_motor"])==8
            assert abs(sum(a["relative_abs_output_contribution_per_motor"])-1)<1e-6
            summary["models"][str(seed)][arm]=a
    for arm in ARMS:
        metrics=(
            "raw_first_pc_explained","standardized_first_pc_explained",
            "raw_effective_rank","standardized_effective_rank",
            "abs_pairwise_corr_mean","abs_pairwise_corr_gt_0_9",
            "contribution_max_unit_fraction","contribution_near_zero_units_below_1pct",
        )
        summary["seed_level_descriptive_means"][arm]={
            metric:float(np.mean([summary["models"][str(s)][arm][metric] for s in SEEDS]))
            for metric in metrics
        }
    for seed in SEEDS:
        a,b=summary["models"][str(seed)]["clean"],summary["models"][str(seed)]["robust"]
        print(
            f"D5b seed{seed} standardized activity effective-rank clean/robust "
            f"{a['standardized_effective_rank']:.2f}/{b['standardized_effective_rank']:.2f}; "
            f"firstPC {a['standardized_first_pc_explained']:.3f}/{b['standardized_first_pc_explained']:.3f}; "
            f"absCorr {a['abs_pairwise_corr_mean']:.3f}/{b['abs_pairwise_corr_mean']:.3f}; "
            f"max contribution {a['contribution_max_unit_fraction']:.3f}/{b['contribution_max_unit_fraction']:.3f}",
            flush=True,
        )
    print("D5b descriptive means:")
    for arm in ARMS:
        print(arm,json.dumps(summary["seed_level_descriptive_means"][arm],sort_keys=True),flush=True)
    out=ROOT/"d5b-summary.json"
    assert not out.exists()
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"D5b aggregate complete: {out}",flush=True)


if __name__=="__main__":
    main()
