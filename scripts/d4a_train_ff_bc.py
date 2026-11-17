#!/usr/bin/env python3
"""D4a 独立纯前馈 BC 训练：完全复用 D1 数据与损失，控制数据/epoch/更新次数。"""
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

from d1_bc import ExpertSequences, DATA_ROOT, SEQ
from dls_p0 import model_info, nparams
from dls_feedforward import FeedforwardController, num_params


def evaluate(model, loader, device):
    model.eval()
    loss_sum, steer_sum, pedal_sum, steps = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            err = pred - y
            point = F.smooth_l1_loss(pred, y, beta=0.1, reduction="none")
            loss_sum += float((point * mask.unsqueeze(-1)).sum())
            steer_sum += float((err[..., 0].abs() * mask).sum())
            pedal_sum += float((err[..., 1].abs() * mask).sum())
            steps += float(mask.sum())
    return {
        "huber": loss_sum / (2 * steps),
        "steer_mae": steer_sum / steps,
        "pedal_mae": pedal_sum / steps,
        "steps": int(steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert os.environ.get("DLS_D4A_TRAINING") == "1"
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train = ExpertSequences("train")
    val = ExpertSequences("val", train.mean, train.std)
    assert train.step_count == 91717 and val.step_count == 9822
    assert len(train.episodes) == 219 and len(val.episodes) == 23
    assert train.seed_set.isdisjoint(val.seed_set)
    assert SEQ == 40

    # 重建 D1 C 初始化之后的 CPU RNG 状态，以复用相同训练数据 shuffle 顺序。
    reference, _ = model_info("C")
    reference_params = nparams(reference)
    assert reference_params == 114390
    d1_loader_rng_state = torch.get_rng_state().clone()
    del reference
    torch.manual_seed(args.seed + 100000)
    model = FeedforwardController()
    params = num_params(model)
    assert params == 114477 and abs(params/reference_params - 1) < 0.001
    # 恢复 D1 C 的 DataLoader 随机数状态。
    torch.set_rng_state(d1_loader_rng_state)
    model.to(device)

    original_path = DATA_ROOT / "checkpoints/d1-v1" / f"C-seed{args.seed}.pt"
    assert original_path.is_file()
    original_sha = hashlib.sha256(original_path.read_bytes()).hexdigest()
    original = torch.load(original_path, map_location="cpu", weights_only=False)
    assert original["variant"] == "C" and original["seed"] == args.seed
    np.testing.assert_array_equal(train.mean, np.asarray(original["train_mean"], np.float32))
    np.testing.assert_array_equal(train.std, np.asarray(original["train_std"], np.float32))

    train_loader = DataLoader(train, batch_size=48, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=48, shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-5)
    out_dir = DATA_ROOT / "checkpoints/d4a"
    reports = DATA_ROOT / "reports/d4a"
    tbdir = DATA_ROOT / "tensorboard/d4a" / f"F-seed{args.seed}"
    for directory in (out_dir, reports, tbdir):
        directory.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"F-seed{args.seed}.pt"
    report_path = reports / f"train-F-seed{args.seed}.json"
    assert not ckpt_path.exists() and not report_path.exists()
    writer = SummaryWriter(log_dir=str(tbdir))
    best = float("inf")
    history = []
    updates = 0
    started = time.monotonic()
    for epoch in range(8):
        model.train()
        sum_loss, seen = 0.0, 0.0
        for x, y, mask in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred = model(x)
            per = F.smooth_l1_loss(pred, y, beta=0.1, reduction="none")
            loss = (per * mask.unsqueeze(-1)).sum() / (2 * mask.sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            updates += 1
            count = float(mask.sum())
            seen += count
            sum_loss += float(loss.detach()) * count
        metrics = evaluate(model, val_loader, device)
        train_loss = sum_loss / seen
        assert seen == train.step_count
        history.append({"epoch": epoch + 1, "train_huber": train_loss, **metrics})
        writer.add_scalar("bc/train_huber", train_loss, epoch + 1)
        writer.add_scalar("bc/val_huber", metrics["huber"], epoch + 1)
        print(f"F seed{args.seed} epoch{epoch+1}/8 train={train_loss:.6f} val={metrics['huber']:.6f}", flush=True)
        if metrics["huber"] < best:
            best = metrics["huber"]
            torch.save({
                "state_dict": model.state_dict(),
                "variant": "F",
                "seed": args.seed,
                "architecture": {"type": "stateless_feedforward", "widths": [259, 256, 185, 2]},
                "num_params": params,
                "train_mean": train.mean,
                "train_std": train.std,
                "data_id": "d1-v1",
                "epochs": 8,
                "batch_size": 48,
                "best_epoch": epoch + 1,
                "val_metrics": metrics,
                "original_C_sha256": original_sha,
            }, ckpt_path)
    writer.close()
    assert updates == 8 * len(train_loader)
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == original_sha
    report = {
        "stage": "D4a separately trained feedforward BC; no hidden memory",
        "seed": args.seed, "gpu": args.gpu,
        "train_steps": train.step_count, "val_steps": val.step_count,
        "train_episodes": len(train.episodes), "val_episodes": len(val.episodes),
        "params_F": params, "params_C": reference_params, "ratio": params/reference_params,
        "epochs": 8, "batch_size": 48, "sequence_length_for_data": SEQ,
        "optimizer_updates": updates,
        "optimizer": "AdamW lr6e-4 wd1e-5, grad clip norm1",
        "loss": "elementwise Huber beta 0.1 on two actions masked",
        "d1_C_sampler_rng_reconstructed": True,
        "validation_best_huber": best,
        "original_C_sha256": original_sha,
        "checkpoint": str(ckpt_path),
        "wall_seconds": time.monotonic()-started,
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"D4a F seed{args.seed} complete: params={params} best_val={best:.6f} updates={updates}", flush=True)


if __name__ == "__main__":
    main()
