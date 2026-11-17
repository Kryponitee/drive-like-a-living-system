#!/usr/bin/env python3
"""D2h: train ONLY the existing feed-forward Critic on observed full-episode returns.

Actor checkpoints are never read or modified here; original D2f critic serves as
an initialization and a frozen baseline. Training and held-out evaluation use
disjoint scenario seeds. Loss is raw-return MSE, no reward rescaling.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dls_recurrent_ppo import IdenticalCritic


DATA = Path("/mnt/datasets/lxb/dls")
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.001
GAMMA = 0.99


def read_dataset(variant, split):
    root = DATA / "rollouts" / "d2h-v1"
    manifest = json.loads((root / f"{split}-{variant}.json").read_text())
    assert manifest["variant"] == variant and manifest["split"] == split
    assert manifest["gamma"] == GAMMA and manifest["sigma_pre_tanh"] == 0.05
    archive = Path(manifest["output"])
    assert archive == root / f"{split}-{variant}.npz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    with np.load(archive, allow_pickle=False) as z:
        x = np.asarray(z["normalized_observations"], dtype=np.float32)
        y = np.asarray(z["realized_discounted_return_to_horizon"], dtype=np.float32)
        old = np.asarray(z["original_d2f_value"], dtype=np.float32)
        offsets = np.asarray(z["episode_offsets"], dtype=np.int64)
        scenes = np.asarray(z["scene_seed"], dtype=np.int32)
        valid = np.asarray(z["uncensored_label_mask"], dtype=bool)
    assert x.shape == (len(y), 259)
    assert old.shape == y.shape == valid.shape == scenes.shape
    assert offsets[-1] == len(y) and len(offsets) == len(manifest["scene_seeds"]) + 1
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(old).all()
    assert all(s >= 4600 and s <= 4623 for s in manifest["scene_seeds"]) if split == "train" else all(
        4700 <= s <= 4715 for s in manifest["scene_seeds"]
    )
    assert manifest["uncensored_supervised_steps"] == int(valid.sum())
    assert valid.sum() >= 1000
    return {"x": x, "y": y, "old": old, "offsets": offsets, "valid": valid,
            "scenes": scenes, "manifest": manifest}


def correlation(a, b):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(prediction, data, constant):
    target = data["y"]
    valid = data["valid"]
    pred = np.asarray(prediction, dtype=np.float64)
    assert pred.shape == target.shape
    p = pred[valid]
    y = target[valid].astype(np.float64)
    residual = y - p
    assert np.isfinite(residual).all()
    variance = float(np.var(y))
    assert variance > 1e-8
    per_episode_mae = []
    per_episode_first_v = []
    per_episode_first_target = []
    for start, stop in zip(data["offsets"][:-1], data["offsets"][1:]):
        ix = slice(int(start), int(stop))
        if np.all(valid[ix]):
            per_episode_mae.append(float(np.mean(np.abs(target[ix] - pred[ix]))))
            per_episode_first_v.append(float(pred[int(start)]))
            per_episode_first_target.append(float(target[int(start)]))
        else:
            assert not np.any(valid[ix]), "No partial-episode censoring expected"
    assert len(per_episode_mae) >= 8
    return {
        "valid_steps": int(valid.sum()),
        "uncensored_episodes": len(per_episode_mae),
        "censored_episodes": int(len(data["offsets"]) - 1 - len(per_episode_mae)),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "explained_variance": 1.0 - float(np.var(residual)) / variance,
        "pearson": correlation(p, y),
        "pred_std": float(np.std(p)),
        "return_std": float(np.std(y)),
        "mean_prediction": float(np.mean(p)),
        "mean_return": float(np.mean(y)),
        "mean_episode_MAE": float(np.mean(per_episode_mae)),
        "first_state_episode_pearson": correlation(per_episode_first_v, per_episode_first_target),
        "constant_train_mean_RMSE": float(np.sqrt(np.mean((y - constant) ** 2))),
    }


def evaluate_model(model, x, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(x), 4096):
            batch = torch.from_numpy(x[offset:offset + 4096]).to(device)
            outputs.append(model(batch).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count()
    torch.set_num_threads(1)
    torch.manual_seed(7200)  # critic starting state is loaded identically to original model per variant
    np.random.seed(7200)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    train = read_dataset(args.variant, "train")
    validation = read_dataset(args.variant, "validation")
    assert set(train["manifest"]["scene_seeds"]).isdisjoint(validation["manifest"]["scene_seeds"])
    source = DATA / "checkpoints" / "d2f-v1" / f"{args.variant}-seed42.pt"
    original = torch.load(source, map_location="cpu", weights_only=False)
    assert original["variant"] == args.variant and original["training_environment_steps"] == 4096
    critic = IdenticalCritic()
    critic.load_state_dict(original["critic_state_dict"], strict=True)
    initial_state = {name: p.detach().clone() for name, p in critic.state_dict().items()}
    critic.to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=LEARNING_RATE, weight_decay=0)
    assert all("actor" not in name for name, _ in critic.named_parameters())

    x = torch.from_numpy(train["x"][train["valid"]].copy()).to(device)
    y = torch.from_numpy(train["y"][train["valid"]].copy()).to(device)
    constant = float(train["y"][train["valid"]].mean())
    old_train = metrics(train["old"], train, constant)
    old_val = metrics(validation["old"], validation, constant)
    report_history = []
    t0 = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        critic.train()
        order = torch.randperm(len(x), device=device)
        objective_sum = 0.0
        for j in range(0, len(x), BATCH_SIZE):
            indices = order[j:j + BATCH_SIZE]
            p = critic(x[indices])
            loss = F.mse_loss(p, y[indices])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100)
            assert torch.isfinite(grad_norm)
            optimizer.step()
            objective_sum += float(loss.detach().item()) * len(indices)
        train_mse = objective_sum / len(x)
        if epoch in (1, 5, 10, 20, 35, EPOCHS):
            val_predictions = evaluate_model(critic, validation["x"], device)
            val_result = metrics(val_predictions, validation, constant)
            report_history.append({"epoch": epoch, "train_MSE": train_mse, "val": val_result})
            print(
                f"{args.variant}: epoch={epoch}/{EPOCHS} train_MSE={train_mse:.2f} "
                f"val_EV={val_result['explained_variance']:.4f} "
                f"val_RMSE={val_result['RMSE']:.2f} val_Pearson={val_result['pearson']}",
                flush=True,
            )

    new_train = metrics(evaluate_model(critic, train["x"], device), train, constant)
    new_val = metrics(evaluate_model(critic, validation["x"], device), validation, constant)
    assert np.isfinite(new_val["RMSE"]) and np.isfinite(new_train["RMSE"])
    param_change = sum(
        float((p.detach().cpu() - initial_state[n]).abs().sum())
        for n, p in critic.state_dict().items()
    )
    assert param_change > 0

    checkpoint_path = DATA / "checkpoints" / "d2h-v1" / f"critic-{args.variant}-seed42.pt"
    report_path = DATA / "reports" / "d2h" / f"critic-{args.variant}-seed42.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    assert not checkpoint_path.exists() and not report_path.exists(), "Do not overwrite previous value-only result"
    torch.save({
        "stage": "D2h frozen-BC actor return-supervised critic only",
        "variant": args.variant,
        "original_critic_source": str(source),
        "critic_state_dict": {k: p.detach().cpu() for k, p in critic.state_dict().items()},
        "epochs": EPOCHS,
        "train_scenes": train["manifest"]["scene_seeds"],
        "validation_scenes": validation["manifest"]["scene_seeds"],
        "gamma": GAMMA, "sigma": 0.05,
        "actor_untouched": True,
    }, checkpoint_path)
    result = {
        "stage": "D2h critic-only regression on full-episode returns; NO actor update",
        "variant": args.variant, "model_seed": 42, "gpu": args.gpu,
        "epochs_fixed_predeclared": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "loss": "unscaled raw-return MSE",
        "old_critic_source": str(source),
        "checkpoint": str(checkpoint_path),
        "train_scene_seeds": train["manifest"]["scene_seeds"],
        "validation_scene_seeds": validation["manifest"]["scene_seeds"],
        "train_episode_count": len(train["manifest"]["scene_seeds"]),
        "validation_episode_count": len(validation["manifest"]["scene_seeds"]),
        "train_valid_steps": old_train["valid_steps"],
        "validation_valid_steps": old_val["valid_steps"],
        "old_train": old_train, "new_train": new_train,
        "old_validation": old_val, "new_validation": new_val,
        "initial_critic_parameter_change_L1": param_change,
        "history": report_history,
        "training_wall_seconds": time.monotonic() - t0,
        "caveat": "Single Monte Carlo future per state; time-censored episodes excluded. Existing D2f critic is off-policy baseline under fixed BC occupancy; no actor/PPO performance claim.",
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FINAL {args.variant}: old EV={old_val['explained_variance']:.4f}, "
          f"new EV={new_val['explained_variance']:.4f}, "
          f"old RMSE={old_val['RMSE']:.3f}, new RMSE={new_val['RMSE']:.3f}, "
          f"constant RMSE={new_val['constant_train_mean_RMSE']:.3f}", flush=True)


if __name__ == "__main__":
    main()
