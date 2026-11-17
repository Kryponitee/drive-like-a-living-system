#!/usr/bin/env python3
"""D5b: 无重新训练的8单元功能多样性诊断，区分一维复制与多维活动。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from d1_bc import ExpertSequences
from dls_motor_redundancy import MOTOR_UNITS, RedundantMotorController, motor_params


ROOT = Path("/mnt/datasets/lxb/dls")
EPS = 1e-12


def spectrum_metrics(activity):
    """活动矩阵: [valid_timesteps,8]，不把单时刻作为独立训练seed。"""
    assert activity.ndim == 2 and activity.shape[1] == MOTOR_UNITS
    assert np.isfinite(activity).all() and len(activity) >= 1000
    means = activity.mean(axis=0)
    centered = activity - means
    centered_std = centered.std(axis=0, ddof=1)
    cov = np.cov(centered, rowvar=False)
    vals = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)
    assert vals.shape == (8,) and vals.sum() > EPS
    weights = vals / vals.sum()
    erank = float(np.exp(-(weights[weights > 0] * np.log(weights[weights > 0])).sum()))
    first_frac = float(weights[0])
    # 以每个单元自身波动尺度做公平比较，排除单元幅度大小直接支配谱。
    z = centered / np.maximum(centered_std, EPS)
    zcov = (z.T @ z) / (len(z)-1)
    zvals = np.maximum(np.linalg.eigvalsh(zcov)[::-1], 0.0)
    assert zvals.sum() > EPS
    zw = zvals / zvals.sum()
    zrank = float(np.exp(-(zw[zw > 0] * np.log(zw[zw > 0])).sum()))
    correlation = zcov / np.maximum(np.sqrt(np.outer(np.diag(zcov), np.diag(zcov))), EPS)
    corr_vals = np.abs(correlation[np.triu_indices(MOTOR_UNITS, k=1)])
    assert len(corr_vals) == 28
    assert np.isfinite(corr_vals).all()
    return {
        "unit_mean": means.tolist(),
        "unit_std": centered_std.tolist(),
        "raw_covariance_eigenvalues": vals.tolist(),
        "raw_eigenvalue_ratio": weights.tolist(),
        "raw_first_pc_explained": first_frac,
        "raw_effective_rank": erank,
        "standardized_covariance_eigenvalues": zvals.tolist(),
        "standardized_first_pc_explained": float(zw[0]),
        "standardized_effective_rank": zrank,
        "abs_pairwise_corr_mean": float(corr_vals.mean()),
        "abs_pairwise_corr_median": float(np.median(corr_vals)),
        "abs_pairwise_corr_max": float(corr_vals.max()),
        "abs_pairwise_corr_gt_0_9": int((corr_vals > 0.9).sum()),
        "abs_pairwise_corr_gt_0_99": int((corr_vals > 0.99).sum()),
        "centered_one_pc_reconstruction_mse_fraction": float(1-weights[0]),
        "standardized_one_pc_reconstruction_mse_fraction": float(1-zw[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42,43,44), required=True)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(2)
    started = time.monotonic()
    train = ExpertSequences("train")
    val = ExpertSequences("val", train.mean, train.std)
    assert train.step_count == 91717 and val.step_count == 9822
    loader = DataLoader(val, batch_size=48, shuffle=False, num_workers=0, pin_memory=False)
    report = {
        "stage": "D5b original frozen motor activity functional diversity: duplicate or distinguishable",
        "seed": args.seed, "num_val_timesteps": val.step_count,
        "no_retraining": True, "heldout_validation_only": True,
        "arms": {},
    }
    for arm in ("clean", "robust"):
        checkpoint_path = ROOT / "checkpoints/d5a" / f"{arm}-seed{args.seed}.pt"
        original_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert ckpt["seed"] == args.seed and ckpt["arm"] == arm
        assert ckpt["variant"] == "D5a" and ckpt["data_id"] == "d1-v1"
        assert ckpt["num_params"] == 115568
        np.testing.assert_array_equal(ckpt["train_mean"], train.mean)
        np.testing.assert_array_equal(ckpt["train_std"], train.std)
        model = RedundantMotorController()
        assert motor_params(model) == ckpt["num_params"]
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        activities = []
        with torch.inference_mode():
            for x,y,mask in loader:
                assert x.shape[1:] == (40,259)
                state = model.initial_state(x.shape[0], torch.device("cpu"))
                for t in range(40):
                    _, state, activity = model.step(x[:,t],state,t,return_activities=True)
                    valid = mask[:,t].bool()
                    if bool(valid.any()):
                        activities.append(activity[valid].numpy().copy())
        all_act = np.concatenate(activities, axis=0).astype(np.float64)
        assert all_act.shape == (9822, MOTOR_UNITS)
        spec = spectrum_metrics(all_act)
        W = model.decoder.detach().cpu().numpy().astype(np.float64).T.copy()
        assert W.shape == (8,2) and np.linalg.matrix_rank(W)==2
        contrib = all_act[:,:,None] * W[None,:,:]
        abs_mean = np.abs(contrib).mean(axis=0)
        sizes = abs_mean.sum(axis=1)
        fraction = sizes / max(float(sizes.sum()),EPS)
        # Check whether W columns share the same output direction.
        wnorm=np.linalg.norm(W,axis=1)
        cos=np.abs((W@W.T)/(np.maximum(wnorm[:,None]*wnorm[None,:], EPS)))
        cos_edges=cos[np.triu_indices(8,k=1)]
        spec.update({
            "decoder_2x8_rank": 2,
            "decoder_columns_8x2": W.tolist(),
            "abs_control_contribution_per_motor_8x2": abs_mean.tolist(),
            "relative_abs_output_contribution_per_motor": fraction.tolist(),
            "contribution_near_zero_units_below_1pct": int((fraction < 0.01).sum()),
            "contribution_max_unit_fraction": float(fraction.max()),
            "decoder_abs_pairwise_cos_mean": float(cos_edges.mean()),
            "decoder_abs_pairwise_cos_gt_0_95": int((cos_edges > 0.95).sum()),
            "source_checkpoint_sha256": original_sha,
        })
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == original_sha
        report["arms"][arm] = spec
        print(
            f"D5b seed{args.seed} {arm}: raw_firstPC={spec['raw_first_pc_explained']:.3f} "
            f"std_firstPC={spec['standardized_first_pc_explained']:.3f} "
            f"std_erank={spec['standardized_effective_rank']:.2f} "
            f"meanAbsCorr={spec['abs_pairwise_corr_mean']:.3f} "
            f"maxUnitFrac={spec['contribution_max_unit_fraction']:.3f} "
            f"deadUnits(<1%)={spec['contribution_near_zero_units_below_1pct']}",
            flush=True,
        )
    report["wall_seconds"] = time.monotonic()-started
    dest = ROOT / "reports/d5b" / f"motor-diversity-seed{args.seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
