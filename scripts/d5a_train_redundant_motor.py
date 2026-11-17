#!/usr/bin/env python3
"""D5a: 相同八单元网络、数据、优化预算；只改变训练期间的单元失活分布。"""

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
from dls_motor_redundancy import (
    MOTOR_UNITS, RedundantMotorController, motor_params, run_motor_sequence,
)


ROOT = Path("/mnt/datasets/lxb/dls")
SEEDS = (42, 43, 44)


def sample_unit_masks(batch, device):
    """每条40步序列：25%健康、45%失活一个、30%失活两个，损伤ID均匀。"""
    modes = torch.rand(batch, device=device)
    first = torch.randint(MOTOR_UNITS, (batch,), device=device)
    other = torch.randint(MOTOR_UNITS - 1, (batch,), device=device)
    second = other + (other >= first).long()
    masks = torch.ones(batch, MOTOR_UNITS, device=device)
    row = torch.arange(batch, device=device)
    one_or_two = modes >= 0.25
    two = modes >= 0.70
    masks[row[one_or_two], first[one_or_two]] = 0.0
    masks[row[two], second[two]] = 0.0
    assert masks.shape == (batch, MOTOR_UNITS)
    assert bool((masks.sum(dim=1) >= MOTOR_UNITS - 2).all())
    return masks


def validate(model, loader, device, lesion_unit=None):
    model.eval()
    total = 0.0
    steps = 0.0
    with torch.no_grad():
        for x, y, time_mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            time_mask = time_mask.to(device, non_blocking=True)
            unit_mask = None
            if lesion_unit is not None:
                unit_mask = torch.ones(x.shape[0], MOTOR_UNITS, device=device)
                unit_mask[:, lesion_unit] = 0.0
            predicted = run_motor_sequence(model, x, unit_mask=unit_mask)
            per = F.smooth_l1_loss(predicted, y, reduction="none", beta=0.1)
            total += float((per * time_mask.unsqueeze(-1)).sum().item())
            steps += float(time_mask.sum().item())
    assert int(steps) == 9822
    return total / (2 * steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("clean", "robust"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert int(os.environ.get("DLS_D5A_TRAINING", "0")) == 1
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
    assert train.seed_set.isdisjoint(val.seed_set)
    assert train.step_count == 91717 and val.step_count == 9822
    assert len(train.episodes) == 219 and len(val.episodes) == 23
    model = RedundantMotorController()
    params = motor_params(model)
    assert params == 115568, params
    model.to(device)

    train_loader = DataLoader(train, batch_size=48, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=48, shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-5)
    outdir = ROOT / "reports/d5a"
    ckptdir = ROOT / "checkpoints/d5a"
    logdir = ROOT / "tensorboard/d5a" / f"{args.arm}-seed{args.seed}"
    for d in (outdir, ckptdir, logdir):
        d.mkdir(parents=True, exist_ok=True)
    ckptpath = ckptdir / f"{args.arm}-seed{args.seed}.pt"
    reportpath = outdir / f"train-{args.arm}-seed{args.seed}.json"
    assert not ckptpath.exists() and not reportpath.exists()
    # 只读审计旧C权重，保证与其他实验互不污染。
    oldc = ROOT / "checkpoints/d1-v1" / f"C-seed{args.seed}.pt"
    assert oldc.is_file()
    old_sha = hashlib.sha256(oldc.read_bytes()).hexdigest()
    writer = SummaryWriter(log_dir=str(logdir))
    history = []
    updates = 0
    started = time.monotonic()
    for epoch in range(8):
        model.train()
        train_loss_sum, seen = 0.0, 0.0
        for x, y, time_mask in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            time_mask = time_mask.to(device, non_blocking=True)
            unit_mask = sample_unit_masks(x.shape[0], device) if args.arm == "robust" else None
            predicted = run_motor_sequence(model, x, unit_mask=unit_mask)
            loss_elem = F.smooth_l1_loss(predicted, y, reduction="none", beta=0.1)
            loss = (loss_elem * time_mask.unsqueeze(-1)).sum() / (2 * time_mask.sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            updates += 1
            count = float(time_mask.sum().item())
            seen += count
            train_loss_sum += float(loss.detach().item()) * count
        assert int(seen) == train.step_count
        normal_val = validate(model, val_loader, device)
        train_loss = train_loss_sum / seen
        history.append({"epoch": epoch+1, "train_masked_huber": train_loss,
                        "val_healthy_huber": normal_val})
        writer.add_scalar("bc/train_masked_huber", train_loss, epoch+1)
        writer.add_scalar("bc/val_healthy_huber", normal_val, epoch+1)
        print(f"D5a {args.arm} seed{args.seed} epoch{epoch+1}/8 "
              f"train={train_loss:.6f} val_healthy={normal_val:.6f}", flush=True)
    lesion_val = [validate(model, val_loader, device, i) for i in range(MOTOR_UNITS)]
    writer.add_scalar("bc/val_mean_single_lesion_huber", float(np.mean(lesion_val)), 8)
    writer.close()
    assert updates == 8 * len(train_loader) == 408, (updates, len(train_loader))
    assert hashlib.sha256(oldc.read_bytes()).hexdigest() == old_sha
    healthy_val = history[-1]["val_healthy_huber"]
    torch.save({
        "state_dict": model.state_dict(),
        "variant": "D5a", "arm": args.arm, "seed": args.seed,
        "num_params": params, "motor_units": MOTOR_UNITS, "motor_hidden": 29,
        "architecture": "8 independent GRUCell motor units; shared P0 slow/fast features; learned W(2x8)",
        "train_mean": train.mean, "train_std": train.std,
        "data_id": "d1-v1", "epochs": 8, "batch_size": 48,
        "optimizer_updates": updates, "healthy_val_huber": healthy_val,
        "single_lesion_val_huber": lesion_val,
        "original_C_sha256": old_sha,
    }, ckptpath)
    rank = int(torch.linalg.matrix_rank(model.decoder.detach()).item())
    report = {
        "stage": "D5a independent learned eight-unit redundancy, robust vs clean BC",
        "arm": args.arm, "seed": args.seed, "gpu": args.gpu,
        "params": params, "train_steps": train.step_count,
        "val_steps": val.step_count, "train_episodes": len(train.episodes),
        "val_episodes": len(val.episodes), "epochs": 8, "batch_size": 48,
        "optimizer_updates": updates, "final_epoch_checkpoint": True,
        "optimizer": "AdamW lr6e-4 wd1e-5 clip1",
        "robust_mask_distribution": "p(0)=0.25, p(1)=0.45, p(2)=0.30; static within SEQ40",
        "healthy_val_huber": healthy_val,
        "single_lesion_val_huber": lesion_val,
        "decoder_rank": rank,
        "decoder_columns": model.decoder.detach().cpu().tolist(),
        "original_C_sha256": old_sha, "checkpoint": str(ckptpath),
        "wall_seconds": time.monotonic()-started,
        "history": history,
    }
    reportpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"D5a {args.arm} seed{args.seed} complete: params={params} "
          f"val={healthy_val:.6f} lesion_val_mean={float(np.mean(lesion_val)):.6f} "
          f"decoder_rank={rank} updates={updates}", flush=True)


if __name__ == "__main__":
    main()
