#!/usr/bin/env python3
"""D3c：从头训练所有四模块配对，区分功能互补与任意通信拓扑。

只使用D3a已冻结的纵向状态代理数据，不宣称MetaDrive闭环结果。
每个data_seed由一个独立GPU进程执行；所有拓扑保持同初始化、训练预算。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d3_artificial_group_calibration import ModularPedalController, EPOCHS, BATCH

ROOT = Path("/mnt/datasets/lxb/dls")
DATA_ID = "d1-longitudinal-v1-44117653285"
SOURCE_DIR = ROOT / "reports" / DATA_ID / "d3-identifiability-v1"
REPORT_DIR = ROOT / "reports" / DATA_ID / "d3c-pairing-v1"
CHECKPOINT_DIR = ROOT / "checkpoints" / DATA_ID / "d3c-pairing-v1"
INIT_SEEDS = (501, 502, 503)
PAIRINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)
TASKS = ("interaction", "additive")
assert os.path.ismount("/mnt/datasets")


def partners_from_pairing(pairs):
    partner = [-1] * 4
    for a, b in pairs:
        assert partner[a] == partner[b] == -1
        partner[a] = b
        partner[b] = a
    return partner


def train_model(train, val, test, partner, init_seed, device):
    # 在同一个task/init下，所有候选拓扑的模型参数初始化与batch顺序相同。
    torch.manual_seed(130000 + init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(130000 + init_seed)
    model = ModularPedalController(partner).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best_val = float("inf")
    best_epoch = -1
    best_weights = None
    started = time.monotonic()

    train_x, train_y = train
    val_x, val_y = val
    test_x, test_y = test
    assert train_x.shape[1] == val_x.shape[1] == test_x.shape[1] == 4
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ordering = torch.randperm(train_x.shape[0], device=device)
        for indices in ordering.split(BATCH):
            pred = model(train_x[indices])
            loss = F.mse_loss(pred, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(val_x), val_y).item())
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_weights = {key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()}
    assert best_epoch > 0 and best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x)
        test_mse = float(F.mse_loss(test_pred, test_y).item())
        test_mae = float((test_pred - test_y).abs().mean().item())
    return best_weights, {
        "parameters": params,
        "best_epoch": best_epoch,
        "val_mse": best_val,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "train_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")
    if args.gpu >= 0:
        assert torch.cuda.is_available() and args.gpu < torch.cuda.device_count()

    data_path = SOURCE_DIR / f"calibration-data-seed{args.data_seed}.npz"
    meta_path = SOURCE_DIR / f"d3a-seed{args.data_seed}.json"
    meta = json.loads(meta_path.read_text())
    assert meta["seed"] == args.data_seed
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == meta["dataset_sha256"]
    truth = tuple(sorted(tuple(sorted(pair)) for pair in meta["ground_truth_pairs"]))
    assert truth in PAIRINGS
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"data-seed{args.data_seed}.json"
    assert not output_path.exists()

    with np.load(data_path, allow_pickle=False) as archive:
        x = {split: torch.tensor(np.array(archive[f"x_{split}"]), device=device)
             for split in ("train", "val", "test")}
        y = {
            "interaction": {split: torch.tensor(np.array(archive[f"y_{split}"]), device=device)
                            for split in x},
            "additive": {split: torch.tensor(np.array(archive[f"additive_{split}"]), device=device)
                         for split in x},
        }
    for split in x:
        assert x[split].dtype == torch.float32
        assert x[split].shape[1] == 4

    results = []
    for task in TASKS:
        datasets = {split: (x[split], y[task][split]) for split in x}
        for init_seed in INIT_SEEDS:
            runs = {}
            for pairing_id in ("P0", "P1", "P2", "NONE"):
                if pairing_id == "NONE":
                    partner = [-1] * 4
                    pairs = None
                else:
                    pairs = PAIRINGS[int(pairing_id[1])]
                    partner = partners_from_pairing(pairs)

                path = CHECKPOINT_DIR / f"{task}-d{args.data_seed}-i{init_seed}-{pairing_id}.pt"
                assert not path.exists(), path
                weights, metrics = train_model(
                    datasets["train"], datasets["val"], datasets["test"],
                    partner, init_seed, device
                )
                torch.save({
                    "state_dict": weights,
                    "task": task,
                    "data_seed": args.data_seed,
                    "init_seed": init_seed,
                    "pairing_id": pairing_id,
                    "partners": partner,
                    "data_sha256": meta["dataset_sha256"],
                    "model_class": "ModularPedalController",
                }, path)
                metrics["checkpoint"] = str(path)
                metrics["partners"] = partner
                runs[pairing_id] = metrics
                print(
                    f"D3c data_seed={args.data_seed} init={init_seed} task={task} "
                    f"topology={pairing_id} val={metrics['val_mse']:.7f} "
                    f"test={metrics['test_mse']:.7f} epoch={metrics['best_epoch']}",
                    flush=True,
                )

            names = ("P0", "P1", "P2")
            val_order = sorted(names, key=lambda n: (runs[n]["val_mse"], n))
            test_order = sorted(names, key=lambda n: (runs[n]["test_mse"], n))
            actual_id = next(f"P{i}" for i, p in enumerate(PAIRINGS) if p == truth)
            advantage_val = min(runs[n]["val_mse"] for n in names if n != actual_id) - runs[actual_id]["val_mse"]
            advantage_test = min(runs[n]["test_mse"] for n in names if n != actual_id) - runs[actual_id]["test_mse"]

            row = {
                "task": task,
                "data_seed": args.data_seed,
                "init_seed": init_seed,
                "all_metrics": runs,
                "validation_selected_topology": val_order[0],
                "validation_topology_ranking": val_order,
                "test_topology_ranking_audit_only": test_order,
                "validation_selected_test_mse": runs[val_order[0]]["test_mse"],
                "true_topology_id_unsealed_for_scoring": actual_id,
                "true_topology_best_by_validation": val_order[0] == actual_id,
                "true_topology_best_by_test": test_order[0] == actual_id,
                "true_topology_margin_vs_best_other_validation": advantage_val,
                "true_topology_margin_vs_best_other_test": advantage_test,
                "validation_best_paired_vs_no_comm": runs["NONE"]["val_mse"] - runs[val_order[0]]["val_mse"],
                "test_mse_no_comm": runs["NONE"]["test_mse"],
            }
            results.append(row)

    manifest = {
        "stage": "D3c train-time topology counterfactual",
        "data_seed": args.data_seed,
        "data_archive": str(data_path),
        "data_archive_sha256": meta["dataset_sha256"],
        "scope": "longitudinal driving-state SURROGATE; not MetaDrive closed loop",
        "training": {
            "init_seeds": INIT_SEEDS,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "optimizer": "Adam",
            "learning_rate": 0.003,
            "checkpoints_per_data_seed": len(results) * 4,
        },
        "candidate_pairings": {f"P{i}": PAIRINGS[i] for i in range(3)},
        "ground_truth_pairs_for_scoring_only": truth,
        "tasks": {
            "interaction": "risk and speed pressure with physically informed nonlinear complementary variables",
            "additive": "independent scalar contributions in pre-sigmoid output logits",
        },
        "num_runs": len(results) * 4,
        "rows": results,
    }
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({
        "stage": manifest["stage"],
        "data_seed": args.data_seed,
        "num_runs": manifest["num_runs"],
        "true_pairing": truth,
        "positive_correct_wins_val": sum(r["true_topology_best_by_validation"] for r in results if r["task"] == "interaction"),
        "negative_placebo_correct_wins_val": sum(r["true_topology_best_by_validation"] for r in results if r["task"] == "additive"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
