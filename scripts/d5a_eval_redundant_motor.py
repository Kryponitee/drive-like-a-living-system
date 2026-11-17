#!/usr/bin/env python3
"""D5a: 全新场景、全八个单单元永久损伤，按训练seed和密度的严格配对评估。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv

from dls_motor_redundancy import MOTOR_UNITS, RedundantMotorController, motor_params


ROOT = Path("/mnt/datasets/lxb/dls")
SCENE_BASE = 36000
SHARD_SIZE = 8
DENSITIES = (0.0, 0.10, 0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("clean", "robust"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--density", type=float, choices=DENSITIES, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    first_seed = SCENE_BASE + args.shard * SHARD_SIZE
    scene_seeds = list(range(first_seed, first_seed + SHARD_SIZE))
    assert scene_seeds[-1] <= SCENE_BASE + 23
    path = ROOT / "checkpoints/d5a" / f"{args.arm}-seed{args.seed}.pt"
    assert path.is_file()
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["arm"] == args.arm and ckpt["seed"] == args.seed
    assert ckpt["variant"] == "D5a" and ckpt["data_id"] == "d1-v1"
    model = RedundantMotorController()
    assert ckpt["num_params"] == motor_params(model) == 115568
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    mean = np.asarray(ckpt["train_mean"], np.float32)
    std = np.asarray(ckpt["train_std"], np.float32)
    assert mean.shape == std.shape == (259,) and (std > 0).all()
    oldc = ROOT / "checkpoints/d1-v1" / f"C-seed{args.seed}.pt"
    assert hashlib.sha256(oldc.read_bytes()).hexdigest() == ckpt["original_C_sha256"]
    if args.arm == "robust":
        paired = ROOT / "checkpoints/d5a" / f"clean-seed{args.seed}.pt"
        clean = torch.load(paired, map_location="cpu", weights_only=False)
        np.testing.assert_array_equal(mean, np.asarray(clean["train_mean"], np.float32))
        np.testing.assert_array_equal(std, np.asarray(clean["train_std"], np.float32))
        assert clean["num_params"] == ckpt["num_params"]
    decoder = model.decoder.detach().numpy().T.copy()  # 每个单元二维控制贡献
    assert decoder.shape == (MOTOR_UNITS, 2)

    dest = (ROOT / "reports/d5a/eval"
            / f"{args.arm}-seed{args.seed}-density{args.density:.2f}-shard{args.shard}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": args.density,
        "map": 3,
        "start_seed": first_seed,
        "num_scenarios": SHARD_SIZE,
        "horizon": 800,
        "log_level": 50,
    })
    episodes = []
    started = time.monotonic()
    with torch.no_grad():
        for seed in scene_seeds:
            # 每个场景都从完全相同的seed重新开始，不从前一次受损的末状态继续。
            for lesion in (-1, *range(MOTOR_UNITS)):
                obs, _ = env.reset(seed=seed)
                state = model.initial_state(1, torch.device("cpu"))
                unit_mask = torch.ones(MOTOR_UNITS, dtype=torch.float32)
                if lesion >= 0:
                    unit_mask[lesion] = 0.0
                reward_total = 0.0
                unit_abs = np.zeros((MOTOR_UNITS, 2), dtype=np.float64)
                info = {}
                for t in range(800):
                    x = np.asarray(obs, np.float32)
                    assert x.shape == (259,) and np.isfinite(x).all()
                    x = np.clip((x - mean) / std, -10.0, 10.0).copy()
                    inp = torch.from_numpy(x).unsqueeze(0)
                    action, state, activities = model.step(
                        inp, state, t, unit_mask=unit_mask, return_activities=True
                    )
                    assert action.shape == (1, 2) and torch.isfinite(action).all()
                    contribution = activities.squeeze(0).numpy()[:, None] * decoder
                    contribution *= unit_mask.numpy()[:, None]
                    unit_abs += np.abs(contribution)
                    command = np.clip(action.squeeze(0).numpy(), -1.0, 1.0).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(command)
                    reward_total += float(reward)
                    if terminated or truncated:
                        break
                success = bool(info.get("arrive_dest", False))
                collision = bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
                offroad = bool(info.get("out_of_road", False))
                episodes.append({
                    "scene_seed": seed,
                    "lesion_unit": lesion,
                    "steps": t+1,
                    "success": success, "collision": collision, "offroad": offroad,
                    "timeout": not (success or collision or offroad),
                    "reward": reward_total,
                    "route_completion": float(info.get("route_completion", 0.0)),
                    "unit_abs_contrib_pre_tanh": (unit_abs / (t+1)).tolist(),
                })
    env.close()
    assert len(episodes) == SHARD_SIZE * (1 + MOTOR_UNITS)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_sha
    assert hashlib.sha256(oldc.read_bytes()).hexdigest() == ckpt["original_C_sha256"]
    by_lesion = {}
    for lesion in (-1, *range(MOTOR_UNITS)):
        sub = [ep for ep in episodes if ep["lesion_unit"] == lesion]
        assert [ep["scene_seed"] for ep in sub] == scene_seeds
        by_lesion[str(lesion)] = {
            "success_out_of_8": int(sum(ep["success"] for ep in sub)),
            "mean_reward": float(np.mean([ep["reward"] for ep in sub])),
            "mean_route_completion": float(np.mean([ep["route_completion"] for ep in sub])),
            "collision": int(sum(ep["collision"] for ep in sub)),
            "offroad": int(sum(ep["offroad"] for ep in sub)),
        }
    result = {
        "stage": "D5a closed-loop single motor lesion generalization",
        "arm": args.arm, "seed": args.seed, "density": args.density,
        "trained_checkpoint_sha256": source_sha,
        "scene_seeds": scene_seeds, "map": 3, "horizon": 800,
        "num_episodes": len(episodes),
        "nine_test_conditions": [-1, *range(MOTOR_UNITS)],
        "by_lesion": by_lesion,
        "episodes": episodes, "wall_seconds": time.monotonic()-started,
    }
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"D5a {args.arm} seed{args.seed} density={args.density:.2f} shard{args.shard} "
          f"healthy={by_lesion['-1']['success_out_of_8']}/8 "
          f"all_single_lesion_success={sum(by_lesion[str(i)]['success_out_of_8'] for i in range(MOTOR_UNITS))}/64",
          flush=True)


if __name__ == "__main__":
    main()
