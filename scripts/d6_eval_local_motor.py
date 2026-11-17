#!/usr/bin/env python3
"""D6: 同场景内严格配对local/cross/cross-blocked，感知瓶颈保持不变。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv

from dls_local_motor import (
    LocalMotorController, num_params, MOTORS, OBS_DIM,
)


ROOT = Path("/mnt/datasets/lxb/dls")
BASE_SEED = 38000
SHARD_SIZE = 8
DENSITIES = (0.0, 0.10, 0.25)
MODES = ("local", "cross", "cross_blocked")


def load_model(arm, seed):
    path = ROOT / "checkpoints/d6" / f"{arm}-seed{seed}.pt"
    assert path.is_file()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["arm"] == arm and ckpt["seed"] == seed
    assert ckpt["variant"] == "D6" and ckpt["data_id"] == "d1-v1"
    model = LocalMotorController(arm)
    assert num_params(model) == ckpt["num_params"] == 114880
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    mean, std = np.asarray(ckpt["train_mean"], np.float32), np.asarray(ckpt["train_std"], np.float32)
    assert mean.shape == std.shape == (OBS_DIM,) and (std > 0).all()
    original_c = ROOT / "checkpoints/d1-v1" / f"C-seed{seed}.pt"
    assert hashlib.sha256(original_c.read_bytes()).hexdigest() == ckpt["original_C_sha256"]
    return model, (mean, std), sha, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42,43,44), required=True)
    parser.add_argument("--density", type=float, choices=DENSITIES, required=True)
    parser.add_argument("--shard", type=int, choices=(0,1,2), required=True)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    first = BASE_SEED + args.shard * SHARD_SIZE
    seeds = list(range(first, first + SHARD_SIZE))
    assert seeds[-1] <= BASE_SEED + 23

    L, (mean_l, std_l), sha_l, path_l = load_model("local", args.seed)
    X, (mean_x, std_x), sha_x, path_x = load_model("cross", args.seed)
    np.testing.assert_array_equal(mean_l, mean_x)
    np.testing.assert_array_equal(std_l, std_x)
    torch.testing.assert_close(L.projection, X.projection, rtol=0, atol=0)

    outdir = ROOT / "reports/d6/eval"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"seed{args.seed}-density{args.density:.2f}-shard{args.shard}.json"
    assert not outfile.exists()

    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": args.density,
        "map": 3,
        "start_seed": first,
        "num_scenarios": SHARD_SIZE,
        "horizon": 800,
        "log_level": 50,
    })
    records = []
    started = time.monotonic()
    with torch.no_grad():
        for scene in seeds:
            for mode in MODES:
                model = L if mode == "local" else X
                blocked = mode == "cross_blocked"
                obs, _ = env.reset(seed=scene)
                state = model.initial_state(1, torch.device("cpu"))
                rewards = 0.0
                total_message = np.zeros(MOTORS, dtype=np.float64)
                total_activity = np.zeros(MOTORS, dtype=np.float64)
                info = {}
                for t in range(800):
                    x = np.asarray(obs, np.float32)
                    assert x.shape == (OBS_DIM,) and np.isfinite(x).all()
                    x = np.clip((x - mean_l)/std_l, -10., 10.).copy()
                    action, state, diag = model.step(
                        torch.from_numpy(x).unsqueeze(0), state, t,
                        block_messages=blocked, return_diagnostics=True,
                    )
                    total_message += diag["mean_abs_messages"].numpy()
                    total_activity += np.abs(diag["activities"].squeeze(0).numpy())
                    action_np = np.clip(action.squeeze(0).numpy(), -1., 1.).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(action_np)
                    rewards += float(reward)
                    if terminated or truncated:
                        break
                records.append({
                    "scene_seed": scene, "mode": mode,
                    "steps": t+1,
                    "success": bool(info.get("arrive_dest", False)),
                    "collision": bool(info.get("crash_vehicle", False) or info.get("crash_object", False)),
                    "offroad": bool(info.get("out_of_road", False)),
                    "reward": rewards,
                    "route_completion": float(info.get("route_completion", 0.0)),
                    "motor_mean_abs_messages": (total_message/(t+1)).tolist(),
                    "motor_mean_abs_activity": (total_activity/(t+1)).tolist(),
                })
    env.close()
    assert len(records) == 24
    for mode in MODES:
        assert [r["scene_seed"] for r in records if r["mode"] == mode] == seeds
    assert hashlib.sha256(path_l.read_bytes()).hexdigest() == sha_l
    assert hashlib.sha256(path_x.read_bytes()).hexdigest() == sha_x
    aggregate = {}
    for mode in MODES:
        group = [r for r in records if r["mode"] == mode]
        aggregate[mode] = {
            "success_of_8": sum(int(r["success"]) for r in group),
            "mean_reward": float(np.mean([r["reward"] for r in group])),
            "mean_completion": float(np.mean([r["route_completion"] for r in group])),
            "collision": sum(int(r["collision"]) for r in group),
            "offroad": sum(int(r["offroad"]) for r in group),
            "mean_abs_message": float(np.mean([np.mean(r["motor_mean_abs_messages"]) for r in group])),
        }
    result = {
        "stage": "D6 paired local information pressure",
        "seed": args.seed, "density": args.density, "shard": args.shard,
        "scene_seeds": seeds, "modes": list(MODES),
        "trained_sha256": {"local": sha_l, "cross": sha_x},
        "num_episodes": len(records),
        "map": 3, "horizon": 800,
        "aggregate": aggregate, "episodes": records,
        "wall_seconds": time.monotonic()-started,
    }
    outfile.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"D6 seed{args.seed} density{args.density:.2f} shard{args.shard}: "
        f"local/cross/blocked success="
        f"{aggregate['local']['success_of_8']}/"
        f"{aggregate['cross']['success_of_8']}/"
        f"{aggregate['cross_blocked']['success_of_8']}; "
        f"cross-message={aggregate['cross']['mean_abs_message']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
