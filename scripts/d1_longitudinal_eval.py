#!/usr/bin/env python3
"""D1′：在完全留出的场景seed上做闭环双踏板评估。"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dls_dual_pedal_adapter import install_dual_pedal_adapter, wrap_dual_pedal_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dls_longitudinal_motor import model_info


parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
parser.add_argument("--data-id", required=True)
args = parser.parse_args()

assert args.data_id == "d1-longitudinal-v1-44117653285"
DATA_ROOT = Path("/mnt/datasets/lxb/dls")
assert os.path.ismount("/mnt/datasets")
CHECKPOINT = DATA_ROOT / "checkpoints" / args.data_id / f"{args.variant}-seed{args.seed}.pt"
REPORT_DIR = DATA_ROOT / "reports" / args.data_id / "closed-loop"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / f"{args.variant}-seed{args.seed}.json"
ARCHIVE_PATH = REPORT_DIR / f"{args.variant}-seed{args.seed}-trajectories.npz"
assert CHECKPOINT.is_file()
assert not REPORT_PATH.exists() and not ARCHIVE_PATH.exists()

torch.set_num_threads(1)
device = torch.device("cpu")
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model, architecture = model_info(args.variant)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()
mean = np.asarray(checkpoint["train_mean"], dtype=np.float32)
std = np.asarray(checkpoint["train_std"], dtype=np.float32)
assert mean.shape == (259,) and std.shape == (259,)
assert checkpoint["data_id"] == args.data_id

DENSITIES = (0.0, 0.1, 0.25)
BASE_SEEDS = (30000, 30100, 30200)
EPISODES_PER_DENSITY = 10
HORIZON = 800
episode_records = []
all_throttle = []
all_brake = []
all_speed = []
all_rewards = []
offsets = [0]
episode_seeds = []
episode_densities = []
episode_success = []
started = time.monotonic()

for density, base_seed in zip(DENSITIES, BASE_SEEDS):
    env = wrap_dual_pedal_env(
        MetaDriveEnv(
            {
                "use_render": False,
                "image_observation": False,
                "map": "S",
                "traffic_density": density,
                "num_scenarios": EPISODES_PER_DENSITY,
                "start_seed": base_seed,
                "horizon": HORIZON,
                "log_level": 50,
            }
        )
    )
    assert env.action_space.shape == (2,)
    assert np.allclose(env.action_space.low, [0.0, 0.0])
    assert np.allclose(env.action_space.high, [1.0, 1.0])

    for seed in range(base_seed, base_seed + EPISODES_PER_DENSITY):
        obs, reset_info = env.reset(seed=seed)
        vehicle = env.agent
        if not hasattr(vehicle, "_dls_original_set_action"):
            install_dual_pedal_adapter(vehicle, steering_controller=lambda _: 0.0)
        state = model.initial_state(1, device)
        throttle_values = []
        brake_values = []
        speed_values = []
        reward_values = []
        last_info = reset_info
        terminated = False
        truncated = False

        with torch.no_grad():
            for step_idx in range(HORIZON):
                observation = np.asarray(obs, dtype=np.float32)
                assert observation.shape == (259,) and np.isfinite(observation).all()
                normalized = np.clip((observation - mean) / std, -10.0, 10.0)
                x = torch.from_numpy(normalized).unsqueeze(0)
                action_tensor, state = model.step(x, state, step_idx)
                action = np.clip(action_tensor.cpu().numpy()[0], 0.0, 1.0).astype(np.float32)
                assert action.shape == (2,) and env.action_space.contains(action)

                next_obs, reward, terminated, truncated, info = env.step(action)
                assert np.isclose(vehicle.steering, 0.0, atol=1e-8)
                assert np.allclose(
                    [vehicle.dls_throttle, vehicle.dls_brake], action, atol=1e-6
                )
                throttle_values.append(float(action[0]))
                brake_values.append(float(action[1]))
                speed_values.append(float(vehicle.speed))
                reward_values.append(float(reward))
                obs = next_obs
                last_info = info
                if terminated or truncated:
                    break

        success = bool(last_info.get("arrive_dest", False))
        collision = bool(
            last_info.get("crash_vehicle", False)
            or last_info.get("crash_object", False)
            or last_info.get("crash_building", False)
        )
        out_of_road = bool(last_info.get("out_of_road", False))
        t = np.asarray(throttle_values, dtype=np.float32)
        b = np.asarray(brake_values, dtype=np.float32)
        v = np.asarray(speed_values, dtype=np.float32)
        r = np.asarray(reward_values, dtype=np.float32)
        assert len(t) > 0 and len(t) == len(b) == len(v) == len(r)
        coactive = (t > 0.1) & (b > 0.1)

        all_throttle.extend(t.tolist())
        all_brake.extend(b.tolist())
        all_speed.extend(v.tolist())
        all_rewards.extend(r.tolist())
        offsets.append(len(all_throttle))
        episode_seeds.append(seed)
        episode_densities.append(density)
        episode_success.append(success)

        episode_records.append(
            {
                "seed": seed,
                "density": density,
                "steps": int(len(t)),
                "success": success,
                "collision": collision,
                "out_of_road": out_of_road,
                "timeout": not success and not collision and not out_of_road,
                "reward_sum": float(r.sum()),
                "route_completion": float(last_info.get("route_completion", 0.0)),
                "mean_speed_mps": float(v.mean()),
                "mean_throttle": float(t.mean()),
                "mean_brake": float(b.mean()),
                "coactive_fraction_threshold_0_1": float(coactive.mean()),
                "mean_min_throttle_brake": float(np.minimum(t, b).mean()),
            }
        )
        print(
            f"variant={args.variant} model_seed={args.seed} density={density:.2f} "
            f"scene={seed} steps={len(t)} success={success} collision={collision} "
            f"coactive={coactive.mean():.3f}",
            flush=True,
        )
    env.close()

throttle_array = np.asarray(all_throttle, dtype=np.float32)
brake_array = np.asarray(all_brake, dtype=np.float32)
speed_array = np.asarray(all_speed, dtype=np.float32)
reward_array = np.asarray(all_rewards, dtype=np.float32)
offset_array = np.asarray(offsets, dtype=np.int64)
success_array = np.asarray(episode_success, dtype=np.bool_)
assert len(episode_records) == 30 and offset_array.shape == (31,)
assert offset_array[-1] == len(throttle_array)
assert len(brake_array) == len(speed_array) == len(reward_array) == len(throttle_array)

np.savez_compressed(
    ARCHIVE_PATH,
    throttle=throttle_array,
    brake=brake_array,
    speed=speed_array,
    reward=reward_array,
    episode_offsets=offset_array,
    episode_seeds=np.asarray(episode_seeds, dtype=np.int32),
    traffic_density=np.asarray(episode_densities, dtype=np.float32),
    episode_success=success_array,
)
archive_sha256 = hashlib.sha256(ARCHIVE_PATH.read_bytes()).hexdigest()
success_by_density = {
    f"{density:.2f}": float(np.mean([
        e["success"] for e in episode_records if e["density"] == density
    ]))
    for density in DENSITIES
}
coactive_values = [e["coactive_fraction_threshold_0_1"] for e in episode_records]
report = {
    "stage": "D1-prime longitudinal dual-pedal closed-loop evaluation",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "data_id": args.data_id,
    "variant": args.variant,
    "training_seed": args.seed,
    "architecture": architecture,
    "checkpoint": str(CHECKPOINT),
    "test_seeds": [30000, 30009, 30100, 30109, 30200, 30209],
    "num_episodes": len(episode_records),
    "num_transitions": int(len(throttle_array)),
    "success_rate": float(success_array.mean()),
    "success_rate_by_density": success_by_density,
    "collision_rate": float(np.mean([e["collision"] for e in episode_records])),
    "out_of_road_rate": float(np.mean([e["out_of_road"] for e in episode_records])),
    "mean_reward_per_episode": float(np.mean([e["reward_sum"] for e in episode_records])),
    "mean_route_completion": float(np.mean([e["route_completion"] for e in episode_records])),
    "mean_speed_mps": float(speed_array.mean()),
    "mean_throttle": float(throttle_array.mean()),
    "mean_brake": float(brake_array.mean()),
    "mean_coactive_fraction_threshold_0_1": float(np.mean(coactive_values)),
    "mean_min_throttle_brake": float(np.minimum(throttle_array, brake_array).mean()),
    "trajectory_archive": str(ARCHIVE_PATH),
    "trajectory_archive_sha256": archive_sha256,
    "wall_seconds": time.monotonic() - started,
    "episodes": episode_records,
}
REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, ensure_ascii=False, indent=2), flush=True)
