#!/usr/bin/env python3
"""D2i-H: Critic history ON/OFF control, equal GRU weights, identical return samples.

Read-only to D2h datasets. No new MetaDrive rollout, no policy optimization.
Uses one GRU (equivalent to repeated GRUCell operations), with or without
carrying the recurrent state across the 64-step window.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("/mnt/datasets/lxb/dls")
WINDOW = 64
HIDDEN = 24
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001


def load_episode_windows(variant, split):
    root = ROOT / "rollouts" / "d2h-v1"
    manifest = json.loads((root / f"{split}-{variant}.json").read_text())
    assert manifest["split"] == split and manifest["variant"] == variant
    archive = root / f"{split}-{variant}.npz"
    assert manifest["output"] == str(archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    with np.load(archive, allow_pickle=False) as data:
        x = np.asarray(data["normalized_observations"], dtype=np.float32)
        returns = np.asarray(data["realized_discounted_return_to_horizon"], dtype=np.float32)
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        seeds = np.asarray(data["scene_seed"], dtype=np.int32)
        valid = np.asarray(data["uncensored_label_mask"], dtype=np.bool_)
    assert x.shape == (len(returns), 259)
    assert offsets[0] == 0 and offsets[-1] == len(returns)
    assert offsets.shape == (len(manifest["scene_seeds"]) + 1,)
    assert returns.shape == seeds.shape == valid.shape
    assert valid.sum() == manifest["uncensored_supervised_steps"]
    assert np.isfinite(x).all() and np.isfinite(returns).all()
    expected = range(4600, 4624) if split == "train" else range(4700, 4716)
    assert manifest["scene_seeds"] == list(expected)

    observations, targets, masks, episode_ids = [], [], [], []
    used_episodes = []
    censored = 0
    for episode_index, (left, right) in enumerate(zip(offsets[:-1], offsets[1:])):
        start, end = int(left), int(right)
        assert end > start
        seed = int(manifest["scene_seeds"][episode_index])
        assert np.all(seeds[start:end] == seed)
        if not np.all(valid[start:end]):
            assert not np.any(valid[start:end])
            censored += 1
            continue
        used_episodes.append(seed)
        for chunk_start in range(start, end, WINDOW):
            n = min(WINDOW, end - chunk_start)
            obs = np.zeros((WINDOW, 259), dtype=np.float32)
            yy = np.zeros((WINDOW,), dtype=np.float32)
            mask = np.zeros((WINDOW,), dtype=np.float32)
            obs[:n] = x[chunk_start:chunk_start+n]
            yy[:n] = returns[chunk_start:chunk_start+n]
            mask[:n] = 1.0
            observations.append(obs)
            targets.append(yy)
            masks.append(mask)
            episode_ids.append(np.full((WINDOW,), seed, dtype=np.int32))
    assert len(used_episodes) >= 8
    stacked = {
        "x": np.stack(observations),
        "y": np.stack(targets),
        "mask": np.stack(masks),
        "episode_ids": np.stack(episode_ids),
        "valid_episode_seeds": used_episodes,
        "censored_episodes": censored,
    }
    assert int(stacked["mask"].sum()) == int(valid.sum())
    return stacked


class HistorySwitchValue(nn.Module):
    def __init__(self, carry_history, train_target_mean, train_target_std):
        super().__init__()
        self.carry_history = carry_history
        self.recurrent = nn.GRU(259, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)
        self.register_buffer("target_mean", torch.tensor(float(train_target_mean)))
        self.register_buffer("target_std", torch.tensor(float(train_target_std)))
        # identical parameters, fixed affine scaling in raw reward units
        assert sum(p.numel() for p in self.parameters()) == 20545

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1:] == (WINDOW, 259)
        batch, steps, _ = x.shape
        if self.carry_history:
            r, _ = self.recurrent(x)
        else:
            # Shape [B*T, 1, F]: one GRU step at a time with zero recurrent state.
            # Both branches use exactly the same GRU weights, bias and output head.
            r, _ = self.recurrent(x.reshape(batch * steps, 1, 259))
            r = r.reshape(batch, steps, HIDDEN)
        return (self.head(r).squeeze(-1) * self.target_std) + self.target_mean


def score(model, data, device, mean_train):
    model.eval()
    predictions = []
    with torch.no_grad():
        for j in range(0, len(data["x"]), 64):
            batch = torch.from_numpy(data["x"][j:j+64]).to(device)
            predictions.append(model(batch).cpu().numpy())
    p = np.concatenate(predictions)[data["mask"].astype(bool)].astype(np.float64)
    y = data["y"][data["mask"].astype(bool)].astype(np.float64)
    epis = data["episode_ids"][data["mask"].astype(bool)]
    assert len(p) == len(y) == int(data["mask"].sum())
    assert np.isfinite(p).all()
    resid = y-p
    variance = float(np.var(y))
    assert variance > 1e-8
    pearson = None if np.std(p) < 1e-12 else float(np.corrcoef(p,y)[0,1])
    ep_mae = [float(np.mean(np.abs(resid[epis == seed]))) for seed in data["valid_episode_seeds"]]
    ep_rmse = [float(np.sqrt(np.mean(resid[epis == seed] ** 2))) for seed in data["valid_episode_seeds"]]
    return {
        "valid_steps": int(len(y)),
        "valid_episodes": len(ep_mae),
        "MAE": float(np.mean(np.abs(resid))),
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "explained_variance": float(1.0 - np.var(resid)/variance),
        "pearson": pearson,
        "constant_train_mean_RMSE": float(np.sqrt(np.mean((y-mean_train)**2))),
        "mean_episode_MAE": float(np.mean(ep_mae)),
        "mean_episode_RMSE": float(np.mean(ep_rmse)),
        "episode_MAE_by_seed": dict(zip(map(str,data["valid_episode_seeds"]),ep_mae)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("A", "B", "C"), required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count()
    torch.set_num_threads(1)
    torch.manual_seed(31007)
    np.random.seed(31007)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    train = load_episode_windows(args.variant, "train")
    val = load_episode_windows(args.variant, "validation")
    assert set(train["valid_episode_seeds"]).isdisjoint(val["valid_episode_seeds"])

    train_values = train["y"][train["mask"].astype(bool)]
    train_mean = float(train_values.mean(dtype=np.float64))
    train_std = float(train_values.std(dtype=np.float64))
    assert np.isfinite(train_mean) and train_std > 1.0
    on = HistorySwitchValue(True, train_mean, train_std).to(device)
    off = HistorySwitchValue(False, train_mean, train_std).to(device)
    off.load_state_dict(copy.deepcopy(on.state_dict()), strict=True)
    assert all(torch.equal(a,b) for a,b in zip(on.parameters(),off.parameters()))
    models = {"on": on, "off": off}
    optimizers = {
        key: torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        for key,model in models.items()
    }
    all_x = torch.from_numpy(train["x"]).to(device)
    all_y = torch.from_numpy(train["y"]).to(device)
    all_m = torch.from_numpy(train["mask"]).to(device)
    assert all_x.shape[1:] == (WINDOW,259)
    n_windows = len(all_x)
    history = []
    start_time = time.monotonic()
    rng = np.random.default_rng(32001)
    for epoch in range(EPOCHS):
        permutation = rng.permutation(n_windows)
        loss_by_condition = {"on":0.0, "off":0.0}
        observed_by_condition = {"on":0, "off":0}
        for j in range(0, n_windows, BATCH_SIZE):
            idx = torch.from_numpy(permutation[j:j+BATCH_SIZE]).to(device)
            xs, ys, mask = all_x[idx], all_y[idx], all_m[idx]
            count = float(mask.sum().item())
            for key, model in models.items():
                model.train()
                prediction = model(xs)
                loss = ((prediction - ys).square() * mask).sum() / mask.sum()
                assert torch.isfinite(loss)
                optimizer = optimizers[key]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                assert torch.isfinite(grad_norm)
                optimizer.step()
                loss_by_condition[key] += float(loss.detach().item()) * count
                observed_by_condition[key] += int(count)
        if epoch+1 in (1,10,20,35,EPOCHS):
            val_results = {name:score(model,val,device,train_mean) for name,model in models.items()}
            row = {
                "epoch":epoch+1,
                "train_MSE":{k:loss_by_condition[k]/observed_by_condition[k] for k in models},
                "validation":val_results,
            }
            history.append(row)
            print(f"{args.variant}: epoch={epoch+1} "
                  f"historyON val_RMSE={val_results['on']['RMSE']:.3f} EV={val_results['on']['explained_variance']:.3f}; "
                  f"OFF val_RMSE={val_results['off']['RMSE']:.3f} EV={val_results['off']['explained_variance']:.3f}",
                  flush=True)

    val_results = history[-1]["validation"]
    train_results = {name:score(model,train,device,train_mean) for name,model in models.items()}
    out_dir = ROOT / "reports" / "d2i-history"
    weight_dir = ROOT / "checkpoints" / "d2i-history"
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"history-ablation-{args.variant}-seed42.json"
    checkpoints = {key:weight_dir / f"critic-{args.variant}-{key}-seed42.pt" for key in models}
    assert not report_path.exists()
    assert all(not path.exists() for path in checkpoints.values())
    for name,model in models.items():
        torch.save({
            "critic_state_dict": {k: v.cpu() for k,v in model.state_dict().items()},
            "variant": args.variant,
            "condition":name,
            "window":WINDOW,
            "target_mean":train_mean,
            "target_std":train_std,
            "train_episode_seeds":train["valid_episode_seeds"],
            "validation_episode_seeds":val["valid_episode_seeds"],
        },checkpoints[name])
    summary = {
        "stage":"D2i-H controlled critic recurrent HISTORY-ON versus per-step RESET-OFF",
        "variant":args.variant,
        "actor_used":"unchanged BC seed42 on D2h observations",
        "no_actor_update":True,
        "no_new_simulation":True,
        "same_initial_critic_weights":True,
        "same_model_architecture_and_parameter_count":20545,
        "history_window":WINDOW,"epochs":EPOCHS,"batch_size":BATCH_SIZE,
        "lr":LR,"loss":"raw-return masked MSE",
        "fixed_output_scaling":"train-only mean/std affine; identical for ON and OFF",
        "train_windows":n_windows,
        "train_valid_steps":int(train["mask"].sum()),
        "validation_valid_steps":int(val["mask"].sum()),
        "train_episodes":train["valid_episode_seeds"],
        "validation_episodes":val["valid_episode_seeds"],
        "train_metrics":train_results,
        "validation_metrics":val_results,
        "validation_relative_RMSE_improvement_ON_over_OFF":
            (val_results["off"]["RMSE"]-val_results["on"]["RMSE"])/val_results["off"]["RMSE"],
        "history":history,
        "checkpoints":{key:str(path) for key,path in checkpoints.items()},
        "wall_seconds":time.monotonic()-start_time,
        "caveat":"One stochastic observed future per state, 64-step truncated history, different A/B/C actor occupancy, 24/16 original scene seeds.",
    }
    report_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"FINAL {args.variant}: val RMSE history ON={val_results['on']['RMSE']:.3f} "
          f"OFF={val_results['off']['RMSE']:.3f}; "
          f"relative improvement={summary['validation_relative_RMSE_improvement_ON_over_OFF']:.2%}",flush=True)

if __name__ == "__main__":
    main()
