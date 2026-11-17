#!/usr/bin/env python3
"""D6: 同局部信息、同网络参数、同D1行为克隆数据；比较local self vs cross peer信息。"""

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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from d1_bc import ExpertSequences, SEQ
from dls_local_motor import (
    LocalMotorController, MOTORS, OBS_DIM, PROJECTION_SEED, num_params, run_sequence,
)


ROOT = Path("/mnt/datasets/lxb/dls")


def validate(model, dataloader, device, block_messages=False):
    model.eval()
    total_loss, valid_steps = 0.0, 0.0
    with torch.no_grad():
        for x, target, time_mask in dataloader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            time_mask = time_mask.to(device, non_blocking=True)
            predicted = run_sequence(model, x, block_messages=block_messages)
            loss = F.smooth_l1_loss(predicted, target, reduction="none", beta=0.1)
            total_loss += float((loss * time_mask.unsqueeze(-1)).sum())
            valid_steps += float(time_mask.sum())
    assert int(valid_steps) == 9822
    return total_loss / (2 * valid_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("local", "cross"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert os.environ.get("DLS_D6_TRAINING") == "1"
    assert torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
    assert SEQ == 40
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    train = ExpertSequences("train")
    val = ExpertSequences("val", train.mean, train.std)
    assert train.step_count == 91717 and val.step_count == 9822
    assert len(train.episodes) == 219 and len(val.episodes) == 23
    assert train.seed_set.isdisjoint(val.seed_set)

    model = LocalMotorController(args.arm)
    count = num_params(model)
    assert count == 114880, count
    assert model.projection.shape == (OBS_DIM, MOTORS * 33)
    model.to(device)
    loader = DataLoader(train, batch_size=48, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=False)
    validation_loader = DataLoader(val, batch_size=48, shuffle=False, num_workers=0,
                                   pin_memory=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-5)

    ckpt_dir = ROOT / "checkpoints/d6"
    report_dir = ROOT / "reports/d6"
    tensorboard = ROOT / "tensorboard/d6" / f"{args.arm}-seed{args.seed}"
    for directory in (ckpt_dir, report_dir, tensorboard):
        directory.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.arm}-seed{args.seed}.pt"
    report_path = report_dir / f"train-{args.arm}-seed{args.seed}.json"
    assert not ckpt_path.exists() and not report_path.exists()
    original_c = ROOT / "checkpoints/d1-v1" / f"C-seed{args.seed}.pt"
    assert original_c.is_file()
    original_sha = hashlib.sha256(original_c.read_bytes()).hexdigest()

    writer = SummaryWriter(log_dir=str(tensorboard))
    history = []
    updates = 0
    started = time.monotonic()
    for epoch in range(8):
        model.train()
        total_train_loss, valid_steps = 0.0, 0.0
        for x, targets, time_mask in loader:
            x = x.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            time_mask = time_mask.to(device, non_blocking=True)
            predicted = run_sequence(model, x)
            loss_elements = F.smooth_l1_loss(predicted, targets, beta=0.1, reduction="none")
            loss = (loss_elements * time_mask.unsqueeze(-1)).sum() / (2 * time_mask.sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            updates += 1
            n = float(time_mask.sum())
            total_train_loss += float(loss.detach()) * n
            valid_steps += n
        assert int(valid_steps) == train.step_count
        healthy_val = validate(model, validation_loader, device)
        epoch_entry = {
            "epoch": epoch + 1,
            "train_huber": total_train_loss / valid_steps,
            "healthy_val_huber": healthy_val,
        }
        history.append(epoch_entry)
        writer.add_scalar("bc/train_huber", epoch_entry["train_huber"], epoch + 1)
        writer.add_scalar("bc/val_huber", healthy_val, epoch + 1)
        print(f"D6 {args.arm} seed{args.seed} epoch{epoch+1}/8 "
              f"train={epoch_entry['train_huber']:.6f} val={healthy_val:.6f}", flush=True)
    assert updates == 408, (updates, len(loader))
    blocked_val = validate(model, validation_loader, device, block_messages=True) if args.arm == "cross" else None
    writer.close()
    assert hashlib.sha256(original_c.read_bytes()).hexdigest() == original_sha
    torch.save({
        "state_dict": model.state_dict(),
        "variant": "D6", "arm": args.arm, "seed": args.seed,
        "num_params": count, "num_motors": MOTORS,
        "projection_seed": PROJECTION_SEED,
        "projection_type": "fixed row orthogonal 259-to-264 split8x33",
        "architecture": "8 independent GRUCell(41,49), learned 8d self/other old hidden state messages; 8 scalar motors; 2x8 decoder",
        "data_id": "d1-v1", "train_mean": train.mean,
        "train_std": train.std, "epochs": 8, "batch_size": 48,
        "optimizer_updates": updates,
        "healthy_val_huber": history[-1]["healthy_val_huber"],
        "cross_blocked_val_huber": blocked_val,
        "original_C_sha256": original_sha,
    }, ckpt_path)
    report = {
        "stage": "D6 local-information bottleneck, cross vs self messages",
        "arm": args.arm, "seed": args.seed, "gpu": args.gpu,
        "params": count, "train_steps": train.step_count,
        "val_steps": val.step_count, "train_episodes": len(train.episodes),
        "val_episodes": len(val.episodes), "epochs": 8, "batch_size": 48,
        "optimizer_updates": updates, "final_epoch_checkpoint": True,
        "training": "same D1 BC objective, no lesion training",
        "projection_seed": PROJECTION_SEED,
        "healthy_val_huber": history[-1]["healthy_val_huber"],
        "cross_blocked_val_huber": blocked_val,
        "original_C_sha256": original_sha,
        "checkpoint": str(ckpt_path),
        "wall_seconds": time.monotonic()-started,
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"D6 {args.arm} seed{args.seed} complete: params={count} "
          f"val={history[-1]['healthy_val_huber']:.6f} blocked_val={blocked_val} "
          f"updates={updates}", flush=True)


if __name__ == "__main__":
    main()
