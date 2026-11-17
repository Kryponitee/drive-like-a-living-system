#!/usr/bin/env python3
"""只读审计D1′九个BC模型；不重新训练、不覆盖checkpoint或原始报告。"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


DATA_ROOT = Path("/mnt/datasets/lxb/dls")
DATA_ID = os.environ["DLS_DATA_ID"]
assert DATA_ID == "d1-longitudinal-v1-44117653285"
REPORT_DIR = DATA_ROOT / "reports" / DATA_ID
CHECKPOINT_DIR = DATA_ROOT / "checkpoints" / DATA_ID
OUT = REPORT_DIR / "d1-bc-audit.json"
assert not OUT.exists()

rows = []
for variant in ("A", "B", "C"):
    for seed in (42, 43, 44):
        report_path = REPORT_DIR / f"train-{variant}-seed{seed}.json"
        checkpoint_path = CHECKPOINT_DIR / f"{variant}-seed{seed}.pt"
        assert report_path.is_file() and checkpoint_path.is_file()
        report = json.loads(report_path.read_text())
        history = report["history"]
        best = min(history, key=lambda row: row["huber"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["best_epoch"] == best["epoch"]
        assert checkpoint["variant"] == variant and checkpoint["seed"] == seed
        assert checkpoint["data_id"] == DATA_ID
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "num_params": report["num_params"],
                "best_epoch": best["epoch"],
                "val_huber": best["huber"],
                "throttle_mae": best["throttle_mae"],
                "brake_mae": best["brake_mae"],
                "mean_pred_throttle": best["mean_pred_throttle"],
                "mean_pred_brake": best["mean_pred_brake"],
                "mean_target_throttle": best["mean_target_throttle"],
                "mean_target_brake": best["mean_target_brake"],
                "pred_coactive_fraction": best["pred_coactive_fraction"],
                "target_coactive_fraction": best["target_coactive_fraction"],
                "checkpoint": str(checkpoint_path),
            }
        )

summary = {}
for variant in ("A", "B", "C"):
    subset = [row for row in rows if row["variant"] == variant]
    summary[variant] = {}
    for key in (
        "val_huber",
        "throttle_mae",
        "brake_mae",
        "pred_coactive_fraction",
    ):
        values = np.asarray([row[key] for row in subset], dtype=np.float64)
        summary[variant][key] = {
            "mean": float(values.mean()),
            "std_across_three_training_seeds": float(values.std(ddof=1)),
            "per_seed": values.tolist(),
        }

result = {
    "stage": "D1-prime longitudinal dual-pedal BC read-only audit",
    "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    "data_id": DATA_ID,
    "models": rows,
    "descriptive_three_seed_summary": summary,
    "interpretation_limit": "teacher-forced validation only; closed-loop test is separate",
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
