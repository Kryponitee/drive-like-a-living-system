#!/usr/bin/env python3
"""D1′：A/B/C容量匹配模型的纵向双踏板行为克隆。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dls_longitudinal_motor import model_info, num_params, run_sequence


parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--data-id", required=True)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=48)
args = parser.parse_args()

assert args.epochs == 8 and args.batch_size == 48
assert args.data_id.startswith("d1-longitudinal-v1-")
DATA_ROOT = Path("/mnt/datasets/lxb/dls")
assert os.path.ismount("/mnt/datasets")
assert os.path.isdir(DATA_ROOT / "demonstrations" / args.data_id)

SEQ = 40
DENSITIES = (0.0, 0.1, 0.25)


class LongitudinalEpisodes(Dataset):
    def __init__(self, split, mean=None, std=None):
        assert split in ("train", "val")
        self.episodes = []
        self.seed_set = set()
        self.density_steps = {f"{d:.2f}": 0 for d in DENSITIES}
        data_dir = DATA_ROOT / "demonstrations" / args.data_id / split

        for density in DENSITIES:
            stem = f"density-{density:.2f}"
            manifest_path = data_dir / f"{stem}.json"
            archive_path = data_dir / f"{stem}.npz"
            meta = json.loads(manifest_path.read_text())
            assert meta["split"] == split and meta["density"] == density
            assert archive_path.is_file()
            assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == meta["archive_sha256"]
            with np.load(archive_path, allow_pickle=False) as z:
                observations = np.asarray(z["observations"], dtype=np.float32)
                actions = np.asarray(z["actions"], dtype=np.float32)
                offsets = np.asarray(z["episode_offsets"], dtype=np.int64)
                seeds = np.asarray(z["episode_seeds"], dtype=np.int32)
                success = np.asarray(z["episode_success"], dtype=np.bool_)
                densities = np.asarray(z["traffic_density"], dtype=np.float32)

            assert observations.ndim == 2 and observations.shape[1] == 259
            assert actions.shape == (len(observations), 2)
            assert offsets.shape == (len(seeds) + 1,) and offsets[0] == 0
            assert offsets[-1] == len(observations)
            assert len(success) == len(seeds) == len(densities)
            assert np.isfinite(observations).all() and np.isfinite(actions).all()
            assert np.all(actions >= 0.0) and np.all(actions <= 1.0)
            assert self.seed_set.isdisjoint(seeds.tolist())

            for i, seed in enumerate(seeds.tolist()):
                start, end = map(int, offsets[i:i + 2])
                assert start < end
                if not bool(success[i]):
                    continue
                episode_obs = observations[start:end].copy()
                episode_actions = actions[start:end].copy()
                self.episodes.append((episode_obs, episode_actions, seed, density))
                self.seed_set.add(seed)
                self.density_steps[f"{density:.2f}"] += end - start

        assert len(self.episodes) >= 20, (split, len(self.episodes))
        if mean is None:
            assert split == "train"
            xs = np.concatenate([episode[0] for episode in self.episodes], axis=0)
            self.mean = xs.mean(axis=0, dtype=np.float64).astype(np.float32)
            self.std = np.maximum(
                xs.std(axis=0, dtype=np.float64).astype(np.float32), 0.01
            )
        else:
            assert split == "val" and mean is not None and std is not None
            self.mean = mean
            self.std = std

        self.chunks = []
        for episode_index, (obs, action, seed, density) in enumerate(self.episodes):
            for start in range(0, len(obs), SEQ):
                self.chunks.append((episode_index, start))
        self.step_count = int(sum(len(episode[0]) for episode in self.episodes))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        episode_index, start = self.chunks[index]
        obs, actions, seed, density = self.episodes[episode_index]
        length = min(SEQ, len(obs) - start)
        x = np.zeros((SEQ, 259), dtype=np.float32)
        y = np.zeros((SEQ, 2), dtype=np.float32)
        mask = np.zeros(SEQ, dtype=np.float32)
        x[:length] = np.clip((obs[start:start + length] - self.mean) / self.std, -10.0, 10.0)
        y[:length] = actions[start:start + length]
        mask[:length] = 1.0
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def evaluate(model, loader, device):
    model.eval()
    totals = {
        "loss": 0.0,
        "throttle_abs": 0.0,
        "brake_abs": 0.0,
        "pred_throttle": 0.0,
        "pred_brake": 0.0,
        "target_throttle": 0.0,
        "target_brake": 0.0,
        "pred_coactive": 0.0,
        "target_coactive": 0.0,
        "steps": 0.0,
    }
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = run_sequence(model, x)
            per = F.smooth_l1_loss(pred, y, reduction="none", beta=0.1)
            valid = mask.unsqueeze(-1)
            totals["loss"] += float((per * valid).sum().item())
            totals["throttle_abs"] += float(((pred[..., 0] - y[..., 0]).abs() * mask).sum().item())
            totals["brake_abs"] += float(((pred[..., 1] - y[..., 1]).abs() * mask).sum().item())
            totals["pred_throttle"] += float((pred[..., 0] * mask).sum().item())
            totals["pred_brake"] += float((pred[..., 1] * mask).sum().item())
            totals["target_throttle"] += float((y[..., 0] * mask).sum().item())
            totals["target_brake"] += float((y[..., 1] * mask).sum().item())
            totals["pred_coactive"] += float(
                (((pred[..., 0] > 0.1) & (pred[..., 1] > 0.1)).float() * mask).sum().item()
            )
            totals["target_coactive"] += float(
                (((y[..., 0] > 0.1) & (y[..., 1] > 0.1)).float() * mask).sum().item()
            )
            totals["steps"] += float(mask.sum().item())

    steps = max(1.0, totals["steps"])
    return {
        "huber": totals["loss"] / (2.0 * steps),
        "throttle_mae": totals["throttle_abs"] / steps,
        "brake_mae": totals["brake_abs"] / steps,
        "mean_pred_throttle": totals["pred_throttle"] / steps,
        "mean_pred_brake": totals["pred_brake"] / steps,
        "mean_target_throttle": totals["target_throttle"] / steps,
        "mean_target_brake": totals["target_brake"] / steps,
        "pred_coactive_fraction": totals["pred_coactive"] / steps,
        "target_coactive_fraction": totals["target_coactive"] / steps,
        "steps": int(totals["steps"]),
    }


torch.set_num_threads(1)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
assert torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)

train = LongitudinalEpisodes("train")
val = LongitudinalEpisodes("val", train.mean, train.std)
assert train.seed_set.isdisjoint(val.seed_set)
assert min(train.seed_set) >= 10000 and max(train.seed_set) < 10300
assert min(val.seed_set) >= 20000 and max(val.seed_set) < 20300

model, architecture = model_info(args.variant)
param_count = num_params(model)
assert param_count == architecture["num_params"]
model.to(device)

train_loader = DataLoader(
    train, batch_size=args.batch_size, shuffle=True, num_workers=0,
    pin_memory=True, drop_last=False,
)
val_loader = DataLoader(
    val, batch_size=args.batch_size, shuffle=False, num_workers=0,
    pin_memory=True, drop_last=False,
)

checkpoint_dir = DATA_ROOT / "checkpoints" / args.data_id
report_dir = DATA_ROOT / "reports" / args.data_id
tensorboard_dir = DATA_ROOT / "tensorboard" / args.data_id / f"{args.variant}-seed{args.seed}"
for directory in (checkpoint_dir, report_dir, tensorboard_dir):
    directory.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / f"{args.variant}-seed{args.seed}.pt"
report_path = report_dir / f"train-{args.variant}-seed{args.seed}.json"
assert not checkpoint_path.exists() and not report_path.exists()

writer = SummaryWriter(log_dir=str(tensorboard_dir))
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-5)
best_val = float("inf")
history = []
started = time.monotonic()

for epoch in range(args.epochs):
    model.train()
    loss_sum = 0.0
    seen_steps = 0.0
    for x, y, mask in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = run_sequence(model, x)
        per = F.smooth_l1_loss(pred, y, reduction="none", beta=0.1)
        loss = (per * mask.unsqueeze(-1)).sum() / (2.0 * mask.sum())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        count = float(mask.sum().item())
        seen_steps += count
        loss_sum += float(loss.detach().item()) * count

    val_metrics = evaluate(model, val_loader, device)
    train_loss = loss_sum / max(1.0, seen_steps)
    history.append({"epoch": epoch + 1, "train_huber": train_loss, **val_metrics})
    writer.add_scalar("bc/train_huber", train_loss, epoch + 1)
    writer.add_scalar("bc/val_huber", val_metrics["huber"], epoch + 1)
    writer.add_scalar("bc/val_throttle_mae", val_metrics["throttle_mae"], epoch + 1)
    writer.add_scalar("bc/val_brake_mae", val_metrics["brake_mae"], epoch + 1)
    writer.add_scalar("bc/val_pred_coactive_fraction", val_metrics["pred_coactive_fraction"], epoch + 1)
    print(
        f"{args.variant} seed={args.seed} epoch={epoch + 1}/{args.epochs} "
        f"train={train_loss:.6f} val={val_metrics['huber']:.6f} "
        f"T_MAE={val_metrics['throttle_mae']:.4f} B_MAE={val_metrics['brake_mae']:.4f}",
        flush=True,
    )
    if val_metrics["huber"] < best_val:
        best_val = val_metrics["huber"]
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture": architecture,
                "variant": args.variant,
                "seed": args.seed,
                "train_mean": train.mean,
                "train_std": train.std,
                "num_params": param_count,
                "best_epoch": epoch + 1,
                "val_metrics": val_metrics,
                "sequence_length": SEQ,
                "action_interface": "[absolute_throttle, absolute_brake] in [0,1]^2",
                "data_id": args.data_id,
            },
            checkpoint_path,
        )

writer.close()
report = {
    "stage": "D1-prime longitudinal dual-pedal BC",
    "variant": args.variant,
    "seed": args.seed,
    "gpu": args.gpu,
    "device_name": torch.cuda.get_device_name(args.gpu),
    "architecture": architecture,
    "num_params": param_count,
    "data_id": args.data_id,
    "train_episodes": len(train.episodes),
    "val_episodes": len(val.episodes),
    "train_steps": train.step_count,
    "val_steps": val.step_count,
    "train_density_steps": train.density_steps,
    "val_density_steps": val.density_steps,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "sequence_length": SEQ,
    "loss": "elementwise smooth L1 beta=0.1 over independent [T,B] targets",
    "output_activation": "two independent sigmoid channels",
    "best_val_huber": best_val,
    "checkpoint": str(checkpoint_path),
    "wall_seconds": time.monotonic() - started,
    "history": history,
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "history"}, ensure_ascii=False, indent=2))
