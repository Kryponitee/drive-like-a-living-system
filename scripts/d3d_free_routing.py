#!/usr/bin/env python3
"""D3d：模块自主选择通信对象，检验从最终双踏板监督中是否形成任务匹配的协调组。

- 不给网络提供真实分组、连接掩码或各模块的计算公式。
- 每个模块独立在「不通信 + 其余3个模块」中选择一个消息来源；
  不强制相互配对、不强制通信，也不把三个完整匹配硬编码为候选。
- 先软路由预热，再用直通Gumbel离散路由；验证/测试始终使用确定性硬路由。
- 数据、隐藏层结构、输出接口均复用D3a/D3c；尚非MetaDrive闭环实验。
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
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d3_artificial_group_calibration import EPOCHS, BATCH

ROOT = Path("/mnt/datasets/lxb/dls")
DATA_ID = "d1-longitudinal-v1-44117653285"
SOURCE_DIR = ROOT / "reports" / DATA_ID / "d3-identifiability-v1"
D3C_REPORT_DIR = ROOT / "reports" / DATA_ID / "d3c-pairing-v1"
REPORT_DIR = ROOT / "reports" / DATA_ID / "d3d-free-routing-v1"
CHECKPOINT_DIR = ROOT / "checkpoints" / DATA_ID / "d3d-free-routing-v1"
INIT_SEEDS = (501, 502, 503)
TASKS = ("interaction", "additive")
BUDGET = 0.001
WARMUP_EPOCHS = 8
assert os.path.ismount("/mnt/datasets")


class FreeRoutingPedalController(nn.Module):
    """任一模块可选任一其他模块或NONE；4个模块均独立选择，没有预设伙伴。"""

    def __init__(self, hidden=24, message_dim=6):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, message_dim))
            for _ in range(4)
        ])
        self.receivers = nn.ModuleList([
            nn.Sequential(nn.Linear(message_dim + 1, hidden), nn.Tanh(),
                          nn.Linear(hidden, 1), nn.Tanh())
            for _ in range(4)
        ])
        self.decoder = nn.Linear(4, 2)
        # column 0 => NONE; columns 1,2,3 => remaining possible senders,
        # ascending module index order; NOT a matrix of preset correct pairings.
        self.options = tuple(tuple([-1] + [j for j in range(4) if j != i])
                             for i in range(4))
        self.router_logits = nn.Parameter(torch.zeros(4, 4))
        with torch.no_grad():
            self.router_logits.add_(0.01 * torch.randn_like(self.router_logits))

    def routing_probabilities(self):
        return torch.softmax(self.router_logits, dim=-1)

    def active_rate(self):
        return 1.0 - self.routing_probabilities()[:, 0].mean()

    def chosen_senders(self):
        indices = self.router_logits.detach().argmax(dim=1).tolist()
        return [self.options[i][j] for i, j in enumerate(indices)]

    def forward(self, x, *, mode="hard", tau=1.0):
        assert mode in ("soft", "gumbel_st", "hard")
        assert x.ndim == 2 and x.shape[1] == 4
        messages = [encoder(x[:, i:i + 1]) for i, encoder in enumerate(self.encoders)]
        if mode == "soft":
            weights = F.softmax(self.router_logits / tau, dim=-1)
        elif mode == "gumbel_st":
            weights = F.gumbel_softmax(self.router_logits, tau=tau, hard=True, dim=-1)
        else:
            indices = self.router_logits.argmax(dim=-1)
            weights = F.one_hot(indices, num_classes=4).to(self.router_logits.dtype)
        activities = []
        for receiver in range(4):
            candidates = torch.stack([
                torch.zeros_like(messages[receiver]) if sender == -1 else messages[sender]
                for sender in self.options[receiver]
            ], dim=1)
            routed = torch.einsum("bkm,k->bm", candidates, weights[receiver])
            activities.append(
                self.receivers[receiver](
                    torch.cat((x[:, receiver:receiver + 1], routed), dim=-1)
                )
            )
        a = torch.cat(activities, dim=-1)
        return torch.sigmoid(self.decoder(a))


def read_splits(data_path, device):
    with np.load(data_path, allow_pickle=False) as archive:
        return {
            task: {
                split: (
                    torch.tensor(np.array(archive[f"x_{split}"]), device=device),
                    torch.tensor(np.array(archive[f"{'y' if task == 'interaction' else 'additive'}_{split}"]), device=device),
                )
                for split in ("train", "val", "test")
            }
            for task in TASKS
        }


def eval_frozen(model, split):
    model.eval()
    with torch.no_grad():
        pred = model(split[0], mode="hard")
        return {
            "mse": float(F.mse_loss(pred, split[1]).item()),
            "mae": float((pred - split[1]).abs().mean().item()),
        }


def graph_metrics(model, truth_pairs):
    senders = model.chosen_senders()
    allowed = {tuple(sorted(pair)) for pair in truth_pairs}
    directed_edges = [(i, j) for i, j in enumerate(senders) if j != -1]
    correct_edges = sum(tuple(sorted((a, b))) in allowed for a, b in directed_edges)
    reciprocals = sorted([
        [i, j] for i in range(4) for j in range(i + 1, 4)
        if senders[i] == j and senders[j] == i
    ])
    return {
        "chosen_senders_by_receiver": senders,
        "num_active_edges": len(directed_edges),
        "correct_edges_against_positive_task_ground_truth": correct_edges,
        "correct_edge_precision": (correct_edges / len(directed_edges)) if directed_edges else None,
        "complete_true_pairing": all(senders[i] == partner
                                     for group in truth_pairs for i, partner
                                     in ((group[0], group[1]), (group[1], group[0]))),
        "reciprocal_pairs": reciprocals,
        "two_disjoint_reciprocal_pairs": len(reciprocals) == 2,
        "routing_probabilities": model.routing_probabilities().detach().cpu().tolist(),
        "routing_soft_active_rate": float(model.active_rate().detach().item()),
    }


def train_one(task, data_seed, init_seed, data, truth, device):
    torch.manual_seed(130000 + init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(130000 + init_seed)
    model = FreeRoutingPedalController().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    started = time.monotonic()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        phase = "soft" if epoch <= WARMUP_EPOCHS else "gumbel_st"
        tau = 1.4 if epoch <= WARMUP_EPOCHS else max(0.45, 1.4 * (0.96 ** (epoch - WARMUP_EPOCHS)))
        order = torch.randperm(len(data["train"][0]), device=device)
        for index in order.split(BATCH):
            xb, yb = (tensor[index] for tensor in data["train"])
            pred = model(xb, mode=phase, tau=tau)
            loss = F.mse_loss(pred, yb) + BUDGET * model.active_rate()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val = eval_frozen(model, data["val"])["mse"]
        if val < best_val:
            best_val, best_epoch = val, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch in (1, 4, 8, 12, 20, 30, 40, 50, 60, 70):
            history.append({
                "epoch": epoch,
                "phase": phase,
                "tau": tau,
                "val_hard_mse": val,
                "selected_senders": model.chosen_senders(),
                "soft_active_rate": float(model.active_rate().detach().item()),
                "router_top_probability_by_receiver": model.routing_probabilities().detach().amax(dim=1).cpu().tolist(),
            })

    assert best_state is not None and best_epoch > 0
    model.load_state_dict(best_state)
    metrics = {
        "task": task, "data_seed": data_seed, "init_seed": init_seed,
        "best_epoch": best_epoch,
        "val_best_hard_mse": best_val,
        "validation": eval_frozen(model, data["val"]),
        "test": eval_frozen(model, data["test"]),
        "graph": graph_metrics(model, truth),
        "router_num_parameters": model.router_logits.numel(),
        "total_parameters": param_count,
        "training_seconds": time.monotonic() - started,
        "history": history,
    }
    path = CHECKPOINT_DIR / f"{task}-d{data_seed}-i{init_seed}.pt"
    assert not path.exists(), path
    torch.save({
        "state_dict": best_state,
        "task": task,
        "data_seed": data_seed,
        "init_seed": init_seed,
        "model": "FreeRoutingPedalController",
        "router_options": model.options,
        "train_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": 0.003,
            "message_penalty": BUDGET,
            "soft_warmup_epochs": WARMUP_EPOCHS,
            "post_warmup": "Gumbel softmax straight through with temperature annealing",
            "eval": "deterministic hard argmax route",
        }
    }, path)
    metrics["checkpoint"] = str(path)
    print(
        f"D3d data={data_seed} init={init_seed} task={task} "
        f"val={metrics['validation']['mse']:.7f} test={metrics['test']['mse']:.7f} "
        f"senders={metrics['graph']['chosen_senders_by_receiver']} "
        f"correct={metrics['graph']['complete_true_pairing']} "
        f"selected_epoch={best_epoch}",
        flush=True,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, choices=(42,43,44), required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")
    if args.gpu >= 0:
        assert torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"data-seed{args.data_seed}.json"
    assert not report_path.exists()

    meta_path = SOURCE_DIR / f"d3a-seed{args.data_seed}.json"
    data_path = SOURCE_DIR / f"calibration-data-seed{args.data_seed}.npz"
    d3c_path = D3C_REPORT_DIR / f"data-seed{args.data_seed}.json"
    assert d3c_path.is_file()
    meta = json.loads(meta_path.read_text())
    d3c = json.loads(d3c_path.read_text())
    assert meta["seed"] == d3c["data_seed"] == args.data_seed
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == meta["dataset_sha256"] == d3c["data_archive_sha256"]

    truth = sorted([sorted(pair) for pair in meta["ground_truth_pairs"]])
    data = read_splits(data_path, device)

    rows = []
    for task in TASKS:
        for init_seed in INIT_SEEDS:
            metrics = train_one(task, args.data_seed, init_seed, data[task], truth, device)
            reference = next(row for row in d3c["rows"]
                             if row["task"] == task and row["init_seed"] == init_seed)
            fixed_validation_selected = reference["validation_selected_topology"]
            fixed_best = reference["all_metrics"][fixed_validation_selected]
            metrics["fixed_pairing_val_selected_reference"] = {
                "selection": fixed_validation_selected,
                "val_mse": fixed_best["val_mse"],
                "test_mse": fixed_best["test_mse"],
            }
            metrics["fixed_no_comm_test_mse"] = reference["all_metrics"]["NONE"]["test_mse"]
            metrics["fixed_true_pairing_test_mse"] = reference["all_metrics"][reference["true_topology_id_unsealed_for_scoring"]]["test_mse"]
            rows.append(metrics)

    report = {
        "stage": "D3d free learned directed module communication",
        "scope": "DLS longitudinal-state surrogate, NOT simulator closed loop",
        "data_seed": args.data_seed,
        "input_data_sha256": meta["dataset_sha256"],
        "true_pairs_for_scoring_only": truth,
        "module_routing": "Each receiver chooses NONE or one of 3 other senders; asymmetric and incomplete connectivity allowed",
        "training": {
            "epochs": EPOCHS, "init_seeds": INIT_SEEDS, "routing": "warmup + straight-through Gumbel",
            "message_penalty": BUDGET, "soft_warmup_epochs": WARMUP_EPOCHS,
            "topology_selection": "best deterministic hard validation MSE epoch",
        },
        "rows": rows,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({
        "stage": report["stage"],
        "data_seed": args.data_seed,
        "positive_full_group_matches": sum(r["graph"]["complete_true_pairing"] for r in rows if r["task"] == "interaction"),
        "negative_placebo_full_group_matches": sum(r["graph"]["complete_true_pairing"] for r in rows if r["task"] == "additive"),
        "positive_mean_test_mse": float(np.mean([r["test"]["mse"] for r in rows if r["task"] == "interaction"])),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
