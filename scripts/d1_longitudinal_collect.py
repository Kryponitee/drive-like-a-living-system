#!/usr/bin/env python3
"""D1′：采集固定转向下的纵向双踏板专家示范。"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dls_dual_pedal_adapter import install_dual_pedal_adapter, wrap_dual_pedal_env


parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=("train", "val", "test"), required=True)
parser.add_argument("--density", type=float, choices=(0.0, 0.1, 0.25), required=True)
parser.add_argument("--base-seed", type=int, required=True)
parser.add_argument("--episodes", type=int, required=True)
args = parser.parse_args()

expected = {"train": 30, "val": 10, "test": 10}
assert args.episodes == expected[args.split]
density_index = {0.0: 0, 0.1: 1, 0.25: 2}[args.density]
expected_base = {"train": 10000, "val": 20000, "test": 30000}[args.split] + 100 * density_index
assert args.base_seed == expected_base

DATA_ROOT = Path(os.environ["DLS_DATA_ROOT"]).resolve()
assert DATA_ROOT == Path("/mnt/datasets/lxb/dls")
assert os.path.ismount("/mnt/datasets")
RUN_ID = os.environ["GITHUB_RUN_ID"]
DATA_ID = f"d1-longitudinal-v1-{RUN_ID}"
DATA_DIR = DATA_ROOT / "demonstrations" / DATA_ID / args.split
REPORT_DIR = DATA_ROOT / "reports" / DATA_ID
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
STEM = f"density-{args.density:.2f}"
ARCHIVE_PATH = DATA_DIR / f"{STEM}.npz"
MANIFEST_PATH = DATA_DIR / f"{STEM}.json"
REPORT_PATH = REPORT_DIR / f"{args.split}-{STEM}.json"
assert not ARCHIVE_PATH.exists() and not MANIFEST_PATH.exists() and not REPORT_PATH.exists()

env = wrap_dual_pedal_env(
    MetaDriveEnv(
        {
            "use_render": False,
            "image_observation": False,
            "map": "S",
            "traffic_density": args.density,
            "num_scenarios": args.episodes,
            "start_seed": args.base_seed,
            "horizon": 800,
            "log_level": 50,
        }
    )
)
assert env.action_space.shape == (2,)
assert np.allclose(env.action_space.low, [0.0, 0.0])
assert np.allclose(env.action_space.high, [1.0, 1.0])

all_obs = []
all_actions = []
all_next_obs = []
all_dones = []
all_teacher_steering = []
offsets = [0]
seeds = []
success_flags = []
episode_records = []
started = time.monotonic()

for seed in range(args.base_seed, args.base_seed + args.episodes):
    obs, reset_info = env.reset(seed=seed)
    vehicle = env.agent
    if not hasattr(vehicle, "_dls_original_set_action"):
        install_dual_pedal_adapter(vehicle, steering_controller=lambda _: 0.0)

    episode_obs = []
    episode_actions = []
    episode_next_obs = []
    episode_dones = []
    teacher_steering = []
    rewards = 0.0
    last_info = reset_info
    terminated = False
    truncated = False

    for step_idx in range(800):
        state = np.asarray(obs, dtype=np.float32)
        teacher_action = np.asarray(expert(vehicle, deterministic=True), dtype=np.float32)
        assert state.shape == (259,) and np.isfinite(state).all()
        assert teacher_action.shape == (2,) and np.isfinite(teacher_action).all()

        teacher_steering.append(float(teacher_action[0]))
        pedal = float(np.clip(teacher_action[1], -1.0, 1.0))
        action = np.asarray([max(pedal, 0.0), max(-pedal, 0.0)], dtype=np.float32)
        assert env.action_space.contains(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = np.asarray(next_obs, dtype=np.float32)
        assert next_state.shape == (259,) and np.isfinite(next_state).all()
        assert np.isclose(vehicle.steering, 0.0, atol=1e-8)
        assert np.allclose([vehicle.dls_throttle, vehicle.dls_brake], action, atol=1e-6)

        episode_obs.append(state)
        episode_actions.append(action)
        episode_next_obs.append(next_state)
        episode_dones.append(bool(terminated or truncated))
        rewards += float(reward)
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
    steps = len(episode_actions)
    assert steps > 0

    all_obs.extend(episode_obs)
    all_actions.extend(episode_actions)
    all_next_obs.extend(episode_next_obs)
    all_dones.extend(episode_dones)
    all_teacher_steering.extend(teacher_steering)
    offsets.append(len(all_obs))
    seeds.append(seed)
    success_flags.append(success)
    episode_records.append(
        {
            "seed": seed,
            "density": args.density,
            "steps": steps,
            "success": success,
            "collision": collision,
            "out_of_road": out_of_road,
            "timeout": not success and not collision and not out_of_road,
            "reward_sum": rewards,
            "throttle_active_steps": int(sum(a[0] > 1e-3 for a in episode_actions)),
            "brake_active_steps": int(sum(a[1] > 1e-3 for a in episode_actions)),
            "simultaneous_active_steps": int(sum(a[0] > 1e-3 and a[1] > 1e-3 for a in episode_actions)),
            "mean_abs_teacher_steering_ignored": float(np.mean(np.abs(teacher_steering))),
        }
    )
    print(
        f"split={args.split} density={args.density:.2f} seed={seed} steps={steps} "
        f"success={success} collision={collision} brake_steps={episode_records[-1]['brake_active_steps']}",
        flush=True,
    )

env.close()

obs_array = np.asarray(all_obs, dtype=np.float32)
action_array = np.asarray(all_actions, dtype=np.float32)
next_array = np.asarray(all_next_obs, dtype=np.float32)
done_array = np.asarray(all_dones, dtype=np.bool_)
offset_array = np.asarray(offsets, dtype=np.int64)
seed_array = np.asarray(seeds, dtype=np.int32)
success_array = np.asarray(success_flags, dtype=np.bool_)
steering_array = np.asarray(all_teacher_steering, dtype=np.float32)
assert obs_array.shape == (len(action_array), 259)
assert next_array.shape == obs_array.shape
assert action_array.shape == (len(obs_array), 2)
assert done_array.shape == (len(obs_array),)
assert steering_array.shape == (len(obs_array),)
assert np.isfinite(obs_array).all() and np.isfinite(action_array).all()
assert bool((action_array >= 0.0).all() and (action_array <= 1.0).all())
assert offset_array.shape == (args.episodes + 1,) and offset_array[-1] == len(obs_array)

np.savez_compressed(
    ARCHIVE_PATH,
    observations=obs_array,
    actions=action_array,
    next_observations=next_array,
    done=done_array,
    episode_offsets=offset_array,
    episode_seeds=seed_array,
    episode_success=success_array,
    traffic_density=np.full(args.episodes, args.density, dtype=np.float32),
    teacher_steering=steering_array,
)
archive_sha256 = hashlib.sha256(ARCHIVE_PATH.read_bytes()).hexdigest()
success_rate = float(success_array.mean())
report = {
    "stage": "D1-prime longitudinal expert data shard",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "data_id": DATA_ID,
    "split": args.split,
    "density": args.density,
    "seed_range": [args.base_seed, args.base_seed + args.episodes - 1],
    "num_attempted": args.episodes,
    "num_successful": int(success_array.sum()),
    "success_rate": success_rate,
    "num_transitions": int(len(obs_array)),
    "throttle_active_steps": int(np.sum(action_array[:, 0] > 1e-3)),
    "brake_active_steps": int(np.sum(action_array[:, 1] > 1e-3)),
    "simultaneous_active_steps": int(np.sum((action_array[:, 0] > 1e-3) & (action_array[:, 1] > 1e-3))),
    "mean_abs_teacher_steering_ignored": float(np.mean(np.abs(steering_array))),
    "archive": str(ARCHIVE_PATH),
    "archive_sha256": archive_sha256,
    "archive_bytes": ARCHIVE_PATH.stat().st_size,
    "episodes": episode_records,
    "wall_seconds": time.monotonic() - started,
}
MANIFEST_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, ensure_ascii=False, indent=2), flush=True)
