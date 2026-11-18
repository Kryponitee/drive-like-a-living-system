#!/usr/bin/env python3
"""D3a/D3b: DLS双踏板控制器中的人工协调组与盲识别校准。

本实验是基于MetaDrive真实自车速度分布的纵向状态代理任务，
不是MetaDrive闭环成功率实验。阳性真值来自受控通信拓扑和互补计算。
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path("/mnt/datasets/lxb/dls")
DATA_ID = "d1-longitudinal-v1-44117653285"
OUT = ROOT / "reports" / DATA_ID / "d3-identifiability-v1"
CKPTS = ROOT / "checkpoints" / DATA_ID / "d3-identifiability-v1"
BASE_SPEED = ROOT / "reports" / DATA_ID / "closed-loop" / "A-seed42-trajectories.npz"
VARIANTS = ("paired", "no_comm", "wrong_pair", "null_unwired", "null_wired")
SIZES = (8192, 2048, 2048)
EPOCHS = 70
BATCH = 512
assert os.path.ismount("/mnt/datasets")


class ModularPedalController(nn.Module):
    """4个等构模块；每个模块接收自己的标量输入和最多一个同伴消息。"""

    def __init__(self, partner, hidden=24, message_dim=6):
        super().__init__()
        assert len(partner) == 4
        assert all(j == -1 or (0 <= j < 4 and j != i) for i, j in enumerate(partner))
        self.partner = tuple(int(j) for j in partner)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, message_dim))
            for _ in range(4)
        ])
        self.receivers = nn.ModuleList([
            nn.Sequential(nn.Linear(message_dim + 1, hidden), nn.Tanh(),
                          nn.Linear(hidden, 1), nn.Tanh())
            for _ in range(4)
        ])
        # 只有线性跨模块读出；互补非线性必须在组内局部消息处理时完成。
        self.decoder = nn.Linear(4, 2)

    def forward(self, x, *, swap_sender=None, permutation=None, diagnostics=False):
        assert x.ndim == 2 and x.shape[1] == 4
        messages = [encoder(x[:, i:i+1]) for i, encoder in enumerate(self.encoders)]
        activities = []
        for j, recv in enumerate(self.receivers):
            sender = self.partner[j]
            if sender < 0:
                m = torch.zeros_like(messages[j])
            else:
                m = messages[sender]
                if swap_sender == sender:
                    assert permutation is not None and permutation.shape == (len(x),)
                    m = m[permutation]
            activities.append(recv(torch.cat([x[:, j:j+1], m], dim=-1)))
        a = torch.cat(activities, dim=-1)
        output = torch.sigmoid(self.decoder(a))
        return (output, a) if diagnostics else output


def partners_for(variant, p):
    assert sorted(p) == list(range(4))
    if variant in ("no_comm", "null_unwired"):
        return [-1] * 4
    pair_roles = ((0, 2), (1, 3)) if variant == "wrong_pair" else ((0, 1), (2, 3))
    links = [-1] * 4
    for role_a, role_b in pair_roles:
        a, b = p[role_a], p[role_b]
        links[a] = b
        links[b] = a
    return links


def targets_for(distance, closing, ego_speed, target_speed):
    """仅用于目标生成，不进入网络内部。均为量纲明确的驾驶状态。"""
    risk = 1.15 * (np.maximum(closing, 0.0) / 8.0)**2 * 26.0 / (distance + 10.0)
    speed_pressure = (target_speed - ego_speed) / (5.0 + 0.25 * ego_speed)
    signed_demand = np.clip(0.68 * speed_pressure - 0.85 * risk, -1.0, 1.0)
    positive = np.stack((np.maximum(signed_demand, 0.0),
                         np.maximum(-signed_demand, 0.0)), axis=-1)
    # 独立两路踏板目标；没有将控制器输出限制为互斥或tanh单轴。
    additive_logit_t = (-1.0 + 0.9 * (distance - 44.0) / 36.0
                        + 0.6 * closing / 9.0 + 1.1 * ego_speed / 14.0
                        - 0.8 * (target_speed - 8.0) / 8.0)
    additive_logit_b = (-0.7 - 0.65 * (distance - 44.0) / 36.0
                        - 0.5 * closing / 9.0 - 0.8 * ego_speed / 14.0
                        + 0.95 * (target_speed - 8.0) / 8.0)
    additive = 1.0 / (1.0 + np.exp(
        -np.stack((additive_logit_t, additive_logit_b), axis=-1)))
    return positive.astype(np.float32), additive.astype(np.float32)


def build_data(seed, speed_pool, role_positions):
    rng = np.random.default_rng(880000 + seed)
    rows = []
    for n in SIZES:
        # 自车速度由已存储的MetaDrive闭环轨迹采样，其他变量是明确标记的反事实条件。
        v = rng.choice(speed_pool, n, replace=True).astype(np.float32)
        v = np.clip(v, 0.0, 19.0)
        d = rng.uniform(8.0, 80.0, n).astype(np.float32)
        closing = (-4.0 + rng.uniform(0.0, 1.0, n) * (4.0 + np.minimum(v, 9.0))).astype(np.float32)
        vt = rng.uniform(1.0, 17.0, n).astype(np.float32)
        assert np.all(v - closing >= -1e-5)  # 前车速度不为负
        y, additive = targets_for(d, closing, v, vt)
        roles = np.stack(((d - 44.0) / 36.0, closing / 9.0,
                          v / 14.0, (vt - 8.0) / 8.0), axis=-1)
        x = np.empty_like(roles, dtype=np.float32)
        x[:, role_positions] = roles
        assert np.isfinite(x).all() and np.isfinite(y).all()
        rows.append((x, y, additive))
    return rows


def evaluate(model, x, y, device):
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x, device=device)
        pred = model(xt).cpu().numpy()
    return {
        "mse": float(np.mean((pred - y)**2)),
        "mae": float(np.mean(np.abs(pred - y))),
        "throttle_mae": float(np.mean(np.abs(pred[:, 0] - y[:, 0]))),
        "brake_mae": float(np.mean(np.abs(pred[:, 1] - y[:, 1]))),
        "pred_coactive_fraction_0_1": float(np.mean((pred[:,0] > .1) & (pred[:,1] > .1))),
        "target_coactive_fraction_0_1": float(np.mean((y[:,0] > .1) & (y[:,1] > .1))),
    }


def training(seed, device):
    assert BASE_SPEED.is_file()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPTS.mkdir(parents=True, exist_ok=True)
    report_path = OUT / f"d3a-seed{seed}.json"
    data_path = OUT / f"calibration-data-seed{seed}.npz"
    assert not report_path.exists() and not data_path.exists()
    for variant in VARIANTS:
        assert not (CKPTS / f"{variant}-seed{seed}.pt").exists()
    with np.load(BASE_SPEED, allow_pickle=False) as f:
        speed_pool = np.asarray(f["speed"], dtype=np.float32)
    speed_pool = speed_pool[np.isfinite(speed_pool)]
    assert len(speed_pool) > 1000
    p = np.random.default_rng(2200+seed).permutation(4).tolist()
    rows = build_data(seed, speed_pool, p)
    np.savez_compressed(
        data_path,
        x_train=rows[0][0], y_train=rows[0][1], additive_train=rows[0][2],
        x_val=rows[1][0], y_val=rows[1][1], additive_val=rows[1][2],
        x_test=rows[2][0], y_test=rows[2][1], additive_test=rows[2][2],
    )
    stats = {}
    for variant in VARIANTS:
        null_task = variant.startswith("null_")
        torch.manual_seed(5000 + seed)
        model = ModularPedalController(partners_for(variant, p)).to(device)
        train_x, train_y = rows[0][0], rows[0][2] if null_task else rows[0][1]
        val_x, val_y = rows[1][0], rows[1][2] if null_task else rows[1][1]
        test_x, test_y = rows[2][0], rows[2][2] if null_task else rows[2][1]
        train_x = torch.as_tensor(train_x, device=device)
        train_y = torch.as_tensor(train_y, device=device)
        val_x = torch.as_tensor(val_x, device=device)
        val_y = torch.as_tensor(val_y, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        best = float("inf")
        best_epoch = 0
        state_best = None
        started = time.monotonic()
        for epoch in range(1, EPOCHS+1):
            model.train()
            order = torch.randperm(len(train_x), device=device)
            for batch in order.split(BATCH):
                pred = model(train_x[batch])
                loss = F.mse_loss(pred, train_y[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(model(val_x), val_y).item()
            if val_loss < best:
                best, best_epoch = val_loss, epoch
                state_best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state_best)
        stats[variant] = {
            "val_mse": best,
            "best_epoch": best_epoch,
            "test": evaluate(model, test_x, test_y, device),
            "train_seconds": time.monotonic()-started,
            "parameters": sum(q.numel() for q in model.parameters()),
        }
        ckpt = CKPTS / f"{variant}-seed{seed}.pt"
        torch.save({
            "state_dict": state_best,
            "variant": variant,
            "seed": seed,
            "partners": partners_for(variant, p),
            "role_positions": p,  # Ground truth is stored for audit, not passed to blind identifier.
            "architecture": "4 identical MLP motor modules, six-dimensional messages, linear shared 2-channel readout",
            "data_archive": str(data_path),
            "null_task": null_task,
        }, ckpt)
        print(f"D3a seed={seed} {variant} val_mse={best:.6f} test_mse={stats[variant]['test']['mse']:.6f} epoch={best_epoch}", flush=True)
    result = {
        "stage": "D3a artificial coordination-group formation and functional fit",
        "seed": seed,
        "input_variables": ["front_distance_m", "closing_speed_mps", "ego_speed_mps", "target_speed_mps"],
        "speed_source_archive": str(BASE_SPEED),
        "speed_source_sha256": hashlib.sha256(BASE_SPEED.read_bytes()).hexdigest(),
        "counterfactual_inputs": ["front_distance_m", "closing_speed_mps", "target_speed_mps"],
        "scope": "longitudinal driving-state surrogate, NOT MetaDrive closed-loop",
        "role_positions_ground_truth_not_available_to_blind_identifier": p,
        "ground_truth_pairs": [sorted([p[0],p[1]]), sorted([p[2],p[3]])],
        "train_val_test_sizes": SIZES,
        "results": stats,
        "positive_better_than_no_comm": stats["paired"]["test"]["mse"] < stats["no_comm"]["test"]["mse"],
        "positive_better_than_wrong_pair": stats["paired"]["test"]["mse"] < stats["wrong_pair"]["test"]["mse"],
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def derive_pairs(scores):
    """只使用干预后的活动/输出变化，不读取partner或真实标签。"""
    possible = [
        [(0,1),(2,3)],
        [(0,2),(1,3)],
        [(0,3),(1,2)],
    ]
    weighted = [(sum(scores[a,b] + scores[b,a] for a,b in pairs), pairs)
                for pairs in possible]
    strength, found = max(weighted, key=lambda item: item[0])
    # 接口没有访问拓扑标签。每条候选边必须出现非零的因果传递。
    is_detected = bool(all(scores[a,b]+scores[b,a] > 0.02 for a,b in found))
    return sorted([sorted(list(p)) for p in found]) if is_detected else [], float(strength)


def identify(model, x, y, device, seed):
    rng = np.random.default_rng(99000 + seed)
    n = min(2048,len(x))
    xt = torch.as_tensor(x[:n], device=device)
    yt = torch.as_tensor(y[:n], device=device)
    shuffle = torch.as_tensor(rng.permutation(n), device=device, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        base_pred, base_activities = model(xt, diagnostics=True)
        base_mse = float(F.mse_loss(base_pred, yt).item())
        variance = base_activities.var(dim=0, unbiased=False).clamp_min(1e-6)
        changes = np.zeros((4,4), dtype=np.float64)
        impacts = []
        for sender in range(4):
            modified_pred, modified_activities = model(
                xt, swap_sender=sender, permutation=shuffle, diagnostics=True)
            diff = ((modified_activities-base_activities)**2).mean(dim=0)/variance
            changes[sender] = diff.cpu().numpy()
            impacts.append(float(F.mse_loss(modified_pred, yt).item()-base_mse))
    # 活动因果传递是主要指标；损失变化独立报告，避免以相关性推断功能。
    scores = changes.copy()
    np.fill_diagonal(scores, 0.0)
    pairs, total = derive_pairs(scores)
    return {
        "baseline_mse": base_mse,
        "message_permutation_loss_increase_by_sender": impacts,
        "normalized_receiver_activity_effect": scores.tolist(),
        "predicted_pairs": pairs,
        "max_pair_score": total,
        "detected": bool(pairs),
        "positive_task_relevance": bool(np.mean(impacts) > 1e-4),
    }


def partition_ari(truth, predicted):
    if not predicted:
        return 0.0
    assert len(truth) == len(predicted) == 2
    truth_labels = [0]*4
    found_labels = [0]*4
    for idx, group in enumerate(truth):
        for m in group:
            truth_labels[m] = idx
    for idx, group in enumerate(predicted):
        for m in group:
            found_labels[m] = idx
    def c2(n):
        return n*(n-1)/2
    matrix = [[sum(truth_labels[k]==i and found_labels[k]==j for k in range(4))
               for j in range(2)] for i in range(2)]
    sum_ij = sum(c2(v) for row in matrix for v in row)
    row_sum = sum(c2(sum(row)) for row in matrix)
    col_sum = sum(c2(sum(matrix[i][j] for i in range(2))) for j in range(2))
    expectation = row_sum*col_sum/c2(4)
    denominator = 0.5*(row_sum+col_sum)-expectation
    return float((sum_ij-expectation)/denominator)


def discovery(seed, device):
    data_path = OUT / f"calibration-data-seed{seed}.npz"
    train_report = OUT / f"d3a-seed{seed}.json"
    report_path = OUT / f"d3b-seed{seed}.json"
    assert train_report.is_file() and data_path.is_file() and not report_path.exists()
    data_meta = json.loads(train_report.read_text())
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == data_meta["dataset_sha256"]
    with np.load(data_path, allow_pickle=False) as data:
        test_x = np.array(data["x_test"])
        test_y = np.array(data["y_test"])
        additive_y = np.array(data["additive_test"])
    blind_results = {}
    for variant in VARIANTS:
        ckpt = torch.load(CKPTS / f"{variant}-seed{seed}.pt", map_location="cpu", weights_only=False)
        assert ckpt["variant"] == variant and ckpt["seed"] == seed
        model = ModularPedalController(ckpt["partners"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        # Blind identifier sees only an intervention-capable forward(x, sender_id).
        # It never reads ckpt["partners"] or ckpt["role_positions"].
        data_target = additive_y if variant.startswith("null_") else test_y
        blind_results[variant] = identify(model, test_x, data_target, device, seed)
        print(f"D3b seed={seed} {variant} pairs={blind_results[variant]['predicted_pairs']} impact={blind_results[variant]['message_permutation_loss_increase_by_sender']}", flush=True)
    # 真值解封只在盲识别完成之后进行，与预测结果作比较。
    truth = sorted([sorted(p) for p in data_meta["ground_truth_pairs"]])
    predicted = blind_results["paired"]["predicted_pairs"]
    recovered = predicted == truth
    null_no_fp = not blind_results["null_unwired"]["detected"]
    positive_relevant = blind_results["paired"]["positive_task_relevance"]
    paired_mse = data_meta["results"]["paired"]["test"]["mse"]
    no_comm_mse = data_meta["results"]["no_comm"]["test"]["mse"]
    wrong_mse = data_meta["results"]["wrong_pair"]["test"]["mse"]
    out = {
        "stage": "D3b blind identification of artificial coordination groups",
        "seed": seed,
        "test_set_sha256": data_meta["dataset_sha256"],
        "blind_identifier_only_uses_intervention_response": True,
        "ground_truth_pairs_revealed_for_final_scoring": truth,
        "predicted_pairs": predicted,
        "positive_pair_exact_recovery": recovered,
        "positive_adjusted_rand_index": partition_ari(truth, predicted),
        "null_unwired_no_false_positive": null_no_fp,
        "functional_fit_positive_mse": paired_mse,
        "functional_fit_no_comm_mse": no_comm_mse,
        "functional_fit_wrong_pair_mse": wrong_mse,
        "functional_comm_advantage_over_no_comm": no_comm_mse-paired_mse,
        "functional_comm_advantage_over_wrong_pair": wrong_mse-paired_mse,
        "positive_message_task_relevance": positive_relevant,
        "positive_calibration_criteria": {
            "paired_better_than_no_comm": paired_mse < no_comm_mse,
            "paired_better_than_wrong_pair": paired_mse < wrong_mse,
            "exact_recovery": recovered,
            "no_unwired_false_positive": null_no_fp,
            "paired_message_shuffle_increases_loss": positive_relevant,
        },
        "controls": blind_results,
    }
    report_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("d3a","d3b"), required=True)
    parser.add_argument("--seed", type=int, choices=(42,43,44), required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")
    if args.gpu >= 0:
        assert torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
    if args.stage == "d3a":
        training(args.seed, device)
    else:
        discovery(args.seed, device)


if __name__ == "__main__":
    main()
