#!/usr/bin/env python3
"""D4a：F vs 原始 C，完全相同的新 MetaDrive 场景、训练种子、观测与动作接口。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv

from dls_p0 import model_info, nparams
from dls_feedforward import FeedforwardController, num_params


ROOT = Path("/mnt/datasets/lxb/dls")
BASE_SCENE = 34000
SHARD_SIZE = 8
DENSITY_CHOICES = (0.0, 0.10, 0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("F", "C"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--density", type=float, choices=DENSITY_CHOICES, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    start_seed = BASE_SCENE + SHARD_SIZE * args.shard
    seeds = list(range(start_seed, start_seed + SHARD_SIZE))
    assert seeds[-1] <= 34023
    checkpoint = (ROOT / ("checkpoints/d4a" if args.variant == "F" else "checkpoints/d1-v1")
                  / f"{args.variant}-seed{args.seed}.pt")
    assert checkpoint.is_file()
    source_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["variant"] == args.variant and ckpt["seed"] == args.seed
    assert ckpt["data_id"] == "d1-v1"
    if args.variant == "F":
        model = FeedforwardController()
        assert num_params(model) == ckpt["num_params"] == 114477
    else:
        model, _ = model_info("C")
        assert nparams(model) == ckpt["num_params"] == 114390
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    mean, std = (np.asarray(ckpt["train_mean"], np.float32),
                 np.asarray(ckpt["train_std"], np.float32))
    assert mean.shape == std.shape == (259,) and np.isfinite(mean).all() and (std > 0).all()

    # 逐个训练种子审计 F 与 C 的输入归一化必须完全相同。
    original_C = ROOT / "checkpoints/d1-v1" / f"C-seed{args.seed}.pt"
    c_ckpt = torch.load(original_C, map_location="cpu", weights_only=False)
    np.testing.assert_array_equal(mean, np.asarray(c_ckpt["train_mean"], np.float32))
    np.testing.assert_array_equal(std, np.asarray(c_ckpt["train_std"], np.float32))
    if args.variant == "F":
        assert hashlib.sha256(original_C.read_bytes()).hexdigest() == ckpt["original_C_sha256"]

    density_tag = f"{args.density:.2f}"
    output = ROOT / "reports/d4a/eval" / f"{args.variant}-seed{args.seed}-density{density_tag}-shard{args.shard}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists()
    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": args.density,
        "map": 3,
        "start_seed": start_seed,
        "num_scenarios": SHARD_SIZE,
        "horizon": 800,
        "log_level": 50,
    })
    episodes = []
    started = time.monotonic()
    with torch.no_grad():
        for scene in seeds:
            obs, _ = env.reset(seed=scene)
            state = model.initial_state(1, torch.device("cpu")) if args.variant == "C" else None
            reward_total, info = 0.0, {}
            for t in range(800):
                x = np.asarray(obs, np.float32)
                assert x.shape == (259,) and np.isfinite(x).all()
                x = np.clip((x - mean) / std, -10.0, 10.0).copy()
                inp = torch.from_numpy(x).unsqueeze(0)
                if args.variant == "C":
                    action, state = model.step(inp, state, t)
                else:
                    action = model(inp)
                assert action.shape == (1, 2) and torch.isfinite(action).all()
                obs, reward, terminated, truncated, info = env.step(
                    np.clip(action[0].numpy().astype(np.float32), -1.0, 1.0))
                reward_total += float(reward)
                if terminated or truncated:
                    break
            arrived = bool(info.get("arrive_dest", False))
            collision = bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
            offroad = bool(info.get("out_of_road", False))
            episodes.append({
                "seed": scene, "steps": t + 1,
                "success": arrived, "collision": collision, "offroad": offroad,
                "timeout": not(arrived or collision or offroad),
                "reward": reward_total,
                "route_completion": float(info.get("route_completion", 0.0)),
            })
    env.close()
    assert [x["seed"] for x in episodes] == seeds
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_sha
    summary = {
        "stage": "D4a paired independent training baseline, C frozen",
        "variant": args.variant, "trained_seed": args.seed, "checkpoint_sha256": source_sha,
        "density": args.density, "scene_seeds": seeds,
        "num_episodes": len(episodes), "map": 3, "horizon": 800,
        "success": sum(x["success"] for x in episodes),
        "collision": sum(x["collision"] for x in episodes),
        "offroad": sum(x["offroad"] for x in episodes),
        "mean_reward": float(np.mean([x["reward"] for x in episodes])),
        "mean_route_completion": float(np.mean([x["route_completion"] for x in episodes])),
        "wall_seconds": time.monotonic() - started,
        "episodes": episodes,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"D4a {args.variant} seed{args.seed} density={density_tag} shard{args.shard} "
          f"success={summary['success']}/8 reward={summary['mean_reward']:.2f}", flush=True)


if __name__ == "__main__":
    main()
