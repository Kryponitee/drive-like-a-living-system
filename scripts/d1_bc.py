#!/usr/bin/env python3
"""D1 行为克隆：同批训练/验证数据，同序列截断与预算，A/B/C 可在不同 GPU 并行训练。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dls_p0 import model_info, nparams, run_sequence, ModularController


DATA_ROOT = Path("/mnt/datasets/lxb/dls")
SEQ = 40


class ExpertSequences(Dataset):
    def __init__(self, split, mean=None, std=None):
        assert split in ("train", "val")
        root = DATA_ROOT / "demonstrations" / "d1-v1" / split
        manifests = sorted(root.glob("worker-*.json"))
        assert len(manifests) == (24 if split == "train" else 3), len(manifests)
        self.episodes = []
        self.seed_set = set()
        self.density_counts = {}
        for meta_path in manifests:
            manifest = json.loads(meta_path.read_text())
            assert manifest["split"] == split
            archive = Path(manifest["archive"])
            assert archive.parent == root
            assert archive.is_file()
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
            with np.load(archive, allow_pickle=False) as shard:
                observations = np.asarray(shard["observations"], dtype=np.float32)
                actions = np.asarray(shard["actions"], dtype=np.float32)
                offsets = np.asarray(shard["episode_offsets"], dtype=np.int64)
                seeds = np.asarray(shard["episode_seeds"], dtype=np.int32)
                densities = np.asarray(shard["traffic_density"], dtype=np.float32)
            assert observations.ndim == 2 and observations.shape[1] == 259
            assert actions.shape == (len(observations), 2)
            assert offsets.shape == (len(seeds) + 1,) and offsets[0] == 0
            assert offsets[-1] == len(observations)
            assert len(densities) == len(seeds)
            assert np.isfinite(observations).all() and np.isfinite(actions).all()
            assert not (self.seed_set & set(seeds.tolist()))
            self.seed_set.update(seeds.tolist())
            for i, seed in enumerate(seeds.tolist()):
                start, end = map(int, offsets[i:i+2])
                assert start < end
                density = float(densities[i])
                self.density_counts[str(round(density, 2))] = self.density_counts.get(str(round(density, 2)), 0) + (end - start)
                self.episodes.append((observations[start:end].copy(), actions[start:end].copy(), seed, density))
        assert len(self.episodes) > 5, "No valid successful episodes"
        if mean is None:
            assert split == "train"
            xs = np.concatenate([e[0] for e in self.episodes], axis=0)
            self.mean = xs.mean(axis=0, dtype=np.float64).astype(np.float32)
            self.std = np.maximum(xs.std(axis=0, dtype=np.float64).astype(np.float32), 0.01)
        else:
            assert mean is not None and std is not None
            self.mean = mean
            self.std = std

        # 剪成以 40 步为单位的连续序列；40 是 N 慢环周期 5 的整数倍。
        self.chunks = []
        for epi, (obs, act, seed, density) in enumerate(self.episodes):
            for start in range(0, len(obs), SEQ):
                self.chunks.append((epi, start))
        self.step_count = int(sum(len(x[0]) for x in self.episodes))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        epi, start = self.chunks[index]
        obs, act, seed, density = self.episodes[epi]
        length = min(SEQ, len(obs) - start)
        x = np.zeros((SEQ, 259), dtype=np.float32)
        y = np.zeros((SEQ, 2), dtype=np.float32)
        mask = np.zeros(SEQ, dtype=np.float32)
        x[:length] = np.clip((obs[start:start+length] - self.mean) / self.std, -10.0, 10.0)
        y[:length] = act[start:start+length]
        mask[:length] = 1
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def evaluate(model, loader, device):
    model.eval()
    accum = {"loss": 0.0, "mae_steer": 0.0, "mae_pedal": 0.0, "steps": 0.0}
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            predicted = run_sequence(model, x)
            err = predicted - y
            loss = F.smooth_l1_loss(predicted, y, reduction="none", beta=0.1)
            accum["loss"] += float((loss * mask.unsqueeze(-1)).sum().item())
            accum["mae_steer"] += float((err[..., 0].abs() * mask).sum().item())
            accum["mae_pedal"] += float((err[..., 1].abs() * mask).sum().item())
            accum["steps"] += float(mask.sum().item())
    return {
        "huber": accum["loss"] / max(1.0, 2 * accum["steps"]),
        "steer_mae": accum["mae_steer"] / max(1.0, accum["steps"]),
        "pedal_mae": accum["mae_pedal"] / max(1.0, accum["steps"]),
        "steps": int(accum["steps"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert int(os.environ.get("DLS_D1_TRAINING", "0")) == 1
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    train = ExpertSequences("train")
    val = ExpertSequences("val", train.mean, train.std)
    assert train.seed_set.isdisjoint(val.seed_set)
    assert min(train.seed_set) >= 1000 and max(train.seed_set) < 1600
    assert min(val.seed_set) >= 2000 and max(val.seed_set) < 2200

    model, arch = model_info(args.variant)
    b_count = nparams(ModularController("B"))
    c_count = nparams(ModularController("C"))
    assert b_count == c_count
    param_count = nparams(model)
    if args.variant == "A":
        assert abs(param_count / c_count - 1.0) < 0.02, ("Monolithic capacity mismatch", param_count, c_count)
    model.to(device)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-5)
    checkpoint_dir = DATA_ROOT / "checkpoints" / "d1-v1"
    report_dir = DATA_ROOT / "reports" / "d1"
    tensorboard_dir = DATA_ROOT / "tensorboard" / "d1-v1" / f"{args.variant}-seed{args.seed}"
    for p in (checkpoint_dir, report_dir, tensorboard_dir):
        p.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{args.variant}-seed{args.seed}.pt"
    report_path = report_dir / f"train-{args.variant}-seed{args.seed}.json"
    assert not checkpoint.exists() and not report_path.exists(), "Do not overwrite earlier training result"

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    best_val = float("inf")
    history = []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        seen_steps = 0.0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            predicted = run_sequence(model, x)
            per_element = F.smooth_l1_loss(predicted, y, reduction="none", beta=0.1)
            loss = (per_element * mask.unsqueeze(-1)).sum() / (2 * mask.sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = float(mask.sum().item())
            seen_steps += count
            loss_sum += float(loss.detach().item()) * count
        val_metrics = evaluate(model, val_loader, device)
        train_loss = loss_sum / seen_steps
        history.append({"epoch": epoch+1, "train_huber": train_loss, **val_metrics})
        writer.add_scalar("bc/train_huber", train_loss, epoch+1)
        writer.add_scalar("bc/val_huber", val_metrics["huber"], epoch+1)
        writer.add_scalar("bc/val_steer_mae", val_metrics["steer_mae"], epoch+1)
        writer.add_scalar("bc/val_pedal_mae", val_metrics["pedal_mae"], epoch+1)
        print(f"{args.variant}: epoch={epoch+1}/{args.epochs} train={train_loss:.6f} "
              f"val={val_metrics['huber']:.6f} steer={val_metrics['steer_mae']:.4f} "
              f"pedal={val_metrics['pedal_mae']:.4f}", flush=True)
        if val_metrics["huber"] < best_val:
            best_val = val_metrics["huber"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": arch,
                    "variant": args.variant,
                    "seed": args.seed,
                    "train_mean": train.mean,
                    "train_std": train.std,
                    "num_params": param_count,
                    "best_epoch": epoch+1,
                    "val_metrics": val_metrics,
                    "sequence_length": SEQ,
                    "slow_period": 5,
                    "data_id": "d1-v1",
                },
                checkpoint
            )
    writer.close()
    report = {
        "stage": "D1 first BC pilot, teacher-forced scores only",
        "variant": args.variant, "seed": args.seed, "gpu": args.gpu,
        "device_name": torch.cuda.get_device_name(args.gpu),
        "architecture": arch, "num_params": param_count,
        "reference_params_C": c_count,
        "relative_params_to_C": param_count/c_count,
        "train_steps": train.step_count, "val_steps": val.step_count,
        "train_episodes": len(train.episodes), "val_episodes": len(val.episodes),
        "train_density_steps": train.density_counts, "val_density_steps": val.density_counts,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "loss": "elementwise smooth L1 beta=0.1 on two actions, mask padded steps",
        "hidden": "zero-initialized every 40-step sequence, 5-step slow updates",
        "best_val_huber": best_val,
        "checkpoint": str(checkpoint),
        "wall_seconds": time.monotonic()-started,
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
