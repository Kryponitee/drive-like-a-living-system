#!/usr/bin/env python3
"""D0'-T: 审计现有MetaDrive PPO专家能否为纵向双踏板任务提供示范。"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dls_dual_pedal_adapter import install_dual_pedal_adapter, wrap_dual_pedal_env


DATA_ROOT = Path(os.environ["DLS_DATA_ROOT"]).resolve()
assert DATA_ROOT == Path("/mnt/datasets/lxb/dls")
assert os.path.ismount("/mnt/datasets")

RUN_ID = os.environ.get("GITHUB_RUN_ID", "manual")
REPORT_DIR = DATA_ROOT / "reports" / "d0-dual-pedal"
DATA_DIR = DATA_ROOT / "demonstrations" / "d0-dual-pedal"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / f"teacher-audit-{RUN_ID}.json"
DATA_PATH = DATA_DIR / f"expert-probe-{RUN_ID}.npz"
assert not REPORT_PATH.exists() and not DATA_PATH.exists()

DENSITIES = (0.0, 0.1, 0.25)
SEEDS_PER_DENSITY = 4
BASE_SEEDS = (7200, 7210, 7220)
HORIZON = 800

all_obs = []
all_actions = []
all_next_obs = []
all_dones = []
episode_offsets = [0]
episode_seeds = []
episode_densities = []
episode_success = []
episodes = []

for density, base_seed in zip(DENSITIES, BASE_SEEDS):
    env = wrap_dual_pedal_env(
        MetaDriveEnv(
            {
                "use_render": False,
                "image_observation": False,
                "map": "S",
                "traffic_density": density,
                "num_scenarios": SEEDS_PER_DENSITY,
                "start_seed": base_seed,
                "horizon": HORIZON,
                "log_level": 50,
            }
        )
    )
    assert env.action_space.shape == (2,)
    assert np.allclose(env.action_space.low, [0.0, 0.0])
    assert np.allclose(env.action_space.high, [1.0, 1.0])

    for seed in range(base_seed, base_seed + SEEDS_PER_DENSITY):
        obs, reset_info = env.reset(seed=seed)
        vehicle = env.agent
        if not hasattr(vehicle, "_dls_original_set_action"):
            install_dual_pedal_adapter(vehicle, steering_controller=lambda _: 0.0)

        episode_obs = []
        episode_actions = []
        episode_next_obs = []
        episode_dones = []
        pedal_values = []
        expert_steer_values = []
        speed_values = []
        last_info = reset_info
        episode_reward = 0.0
        terminated = False
        truncated = False

        for step_idx in range(HORIZON):
            state = np.asarray(obs, dtype=np.float32)
            assert state.ndim == 1 and np.isfinite(state).all()
            teacher_action = np.asarray(expert(vehicle, deterministic=True), dtype=np.float32)
            assert teacher_action.shape == (2,), teacher_action.shape
            assert np.isfinite(teacher_action).all()

            expert_steer_values.append(float(teacher_action[0]))
            pedal = float(np.clip(teacher_action[1], -1.0, 1.0))
            throttle = max(pedal, 0.0)
            brake = max(-pedal, 0.0)
            action = np.asarray([throttle, brake], dtype=np.float32)
            assert env.action_space.contains(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = np.asarray(next_obs, dtype=np.float32)
            assert next_state.shape == state.shape and np.isfinite(next_state).all()
            assert np.isclose(vehicle.steering, 0.0, atol=1e-8)
            assert np.isclose(vehicle.dls_throttle, throttle, atol=1e-6)
            assert np.isclose(vehicle.dls_brake, brake, atol=1e-6)

            episode_obs.append(state)
            episode_actions.append(action)
            episode_next_obs.append(next_state)
            episode_dones.append(bool(terminated or truncated))
            pedal_values.append(pedal)
            speed_values.append(float(vehicle.speed))
            episode_reward += float(reward)
            obs = next_obs
            last_info = info
            if terminated or truncated:
                break

        steps = len(episode_actions)
        assert steps > 0
        success = bool(last_info.get("arrive_dest", False))
        collision = bool(
            last_info.get("crash_vehicle", False)
            or last_info.get("crash_object", False)
            or last_info.get("crash_building", False)
        )
        out_of_road = bool(last_info.get("out_of_road", False))
        brake_count = sum(a[1] > 1e-3 for a in episode_actions)
        throttle_count = sum(a[0] > 1e-3 for a in episode_actions)
        simultaneous_count = sum(a[0] > 1e-3 and a[1] > 1e-3 for a in episode_actions)

        all_obs.extend(episode_obs)
        all_actions.extend(episode_actions)
        all_next_obs.extend(episode_next_obs)
        all_dones.extend(episode_dones)
        episode_offsets.append(len(all_obs))
        episode_seeds.append(seed)
        episode_densities.append(density)
        episode_success.append(success)

        episodes.append(
            {
                "seed": seed,
                "density": density,
                "steps": steps,
                "success": success,
                "collision": collision,
                "out_of_road": out_of_road,
                "timeout": not success and not collision and not out_of_road,
                "reward_sum": episode_reward,
                "mean_speed_mps": float(np.mean(speed_values)),
                "max_speed_mps": float(np.max(speed_values)),
                "throttle_active_steps": int(throttle_count),
                "brake_active_steps": int(brake_count),
                "simultaneous_active_steps": int(simultaneous_count),
                "mean_abs_teacher_steering_ignored": float(np.mean(np.abs(expert_steer_values))),
                "pedal_min": float(np.min(pedal_values)),
                "pedal_max": float(np.max(pedal_values)),
            }
        )
        print(
            f"seed={seed} density={density:.2f} steps={steps} success={success} "
            f"collision={collision} brake_steps={brake_count} throttle_steps={throttle_count}",
            flush=True,
        )
    env.close()

obs_array = np.asarray(all_obs, dtype=np.float32)
action_array = np.asarray(all_actions, dtype=np.float32)
next_array = np.asarray(all_next_obs, dtype=np.float32)
done_array = np.asarray(all_dones, dtype=np.bool_)
offset_array = np.asarray(episode_offsets, dtype=np.int64)
assert obs_array.ndim == 2 and len(obs_array) == len(action_array) == len(next_array)
assert action_array.shape[1] == 2 and np.all(action_array >= 0.0) and np.all(action_array <= 1.0)
assert offset_array.shape == (len(episodes) + 1,) and offset_array[-1] == len(obs_array)

np.savez_compressed(
    DATA_PATH,
    observations=obs_array,
    actions=action_array,
    next_observations=next_array,
    done=done_array,
    episode_offsets=offset_array,
    episode_seeds=np.asarray(episode_seeds, dtype=np.int32),
    traffic_density=np.asarray(episode_densities, dtype=np.float32),
    episode_success=np.asarray(episode_success, dtype=np.bool_),
)

success_rate = float(np.mean(episode_success))
success_by_density = {
    f"{density:.2f}": float(np.mean([
        e["success"] for e in episodes if e["density"] == density
    ]))
    for density in DENSITIES
}
total_steps = len(action_array)
brake_steps = int(np.sum(action_array[:, 1] > 1e-3))
throttle_steps = int(np.sum(action_array[:, 0] > 1e-3))
simultaneous_steps = int(np.sum(
    (action_array[:, 0] > 1e-3) & (action_array[:, 1] > 1e-3)
))
brake_fraction = brake_steps / total_steps
teacher_gate_pass = (
    success_rate >= 0.75
    and min(success_by_density.values()) >= 0.5
    and brake_steps >= 20
)

report = {
    "stage": "D0'-T longitudinal teacher audit under independent absolute pedals",
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "teacher coverage audit only; no DLS training",
    "map": "S",
    "steering": "fixed zero; PPO expert steering output intentionally ignored",
    "teacher": "MetaDrive deterministic PPO expert; signed pedal split into T=max(p,0), B=max(-p,0)",
    "action_interface": "[absolute_throttle, absolute_brake] in [0,1]^2",
    "seeds_per_density": SEEDS_PER_DENSITY,
    "episodes": episodes,
    "num_episodes": len(episodes),
    "num_transitions": total_steps,
    "success_rate": success_rate,
    "success_rate_by_density": success_by_density,
    "throttle_active_steps": throttle_steps,
    "brake_active_steps": brake_steps,
    "brake_step_fraction": brake_fraction,
    "simultaneous_active_steps": simultaneous_steps,
    "teacher_gate": {
        "criterion": "overall success >=0.75; each density success >=0.50; at least 20 brake-active transitions",
        "passed": bool(teacher_gate_pass),
    },
    "dataset_path": str(DATA_PATH),
    "dataset_bytes": DATA_PATH.stat().st_size,
    "report_path": str(REPORT_PATH),
}
REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
