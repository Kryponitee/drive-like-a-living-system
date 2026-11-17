#!/usr/bin/env python3
"""D0c：只改变交通密度，审计专家表现与动作覆盖；成功轨迹另存为候选，不训练。"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert


data_root = Path(os.environ["DLS_DATA_ROOT"]).resolve()
assert data_root == Path("/mnt/datasets/lxb/dls")
assert os.path.ismount("/mnt/datasets")  # 严格确认数据盘已挂载

report_dir = data_root / "reports" / "d0"
demo_dir = data_root / "demonstrations" / "d0c_success_only"
report_dir.mkdir(parents=True, exist_ok=True)
demo_dir.mkdir(parents=True, exist_ok=True)

max_steps = 800
densities = (0.0, 0.1, 0.25)
scenario_seeds = tuple(range(12))
all_groups = []

for density in densities:
    env = MetaDriveEnv(
        {
            "use_render": False,
            "image_observation": False,
            "traffic_density": density,
            "map": 3,
            "start_seed": 0,
            "num_scenarios": 32,
            "horizon": max_steps,
            "log_level": 50,
        }
    )
    episodes = []
    success_records = []
    success_count = 0

    for seed in scenario_seeds:
        obs, _ = env.reset(seed=seed)
        observations = []
        actions = []
        next_observations = []
        total_reward = 0.0
        final_info = {}
        ended = False

        for t in range(max_steps):
            obs = np.asarray(obs, dtype=np.float32)
            action = np.clip(np.asarray(expert(env.agent, deterministic=True), dtype=np.float32), -1, 1)
            assert obs.shape == (259,) and action.shape == (2,)
            assert np.isfinite(obs).all() and np.isfinite(action).all()
            nxt, reward, terminated, truncated, info = env.step(action)
            nxt = np.asarray(nxt, dtype=np.float32)
            assert nxt.shape == obs.shape and np.isfinite(nxt).all()

            observations.append(obs)
            actions.append(action)
            next_observations.append(nxt)
            total_reward += float(reward)
            obs = nxt
            final_info = info
            ended = bool(terminated or truncated)
            if ended:
                break

        arr = np.stack(actions)
        success = bool(final_info.get("arrive_dest", False))
        collided = bool(final_info.get("crash_vehicle", False) or final_info.get("crash_object", False))
        episodes.append(
            {
                "seed": seed,
                "steps": int(len(arr)),
                "success": success,
                "collision": collided,
                "out_of_road": bool(final_info.get("out_of_road", False)),
                "truncated_without_success": bool(not success and not collided and not final_info.get("out_of_road", False)),
                "reward_sum": total_reward,
                "steering_active_fraction_abs_gt_0_10": float(np.mean(np.abs(arr[:, 0]) > 0.1)),
                "brake_fraction_pedal_lt_neg_0_05": float(np.mean(arr[:, 1] < -0.05)),
                "accel_fraction_pedal_gt_0_05": float(np.mean(arr[:, 1] > 0.05)),
                "mean_abs_steering": float(np.mean(np.abs(arr[:, 0]))),
                "pedal_min": float(np.min(arr[:, 1])),
                "pedal_max": float(np.max(arr[:, 1])),
            }
        )

        if success:
            success_count += 1
            success_records.append((seed, np.stack(observations), arr, np.stack(next_observations)))

    env.close()
    path = demo_dir / f"expert-success-density-{density:.2f}.npz"
    if success_records:
        samples = np.concatenate([x[1] for x in success_records], axis=0)
        commands = np.concatenate([x[2] for x in success_records], axis=0)
        successors = np.concatenate([x[3] for x in success_records], axis=0)
        ep_ids = np.concatenate([np.full(len(x[2]), x[0], dtype=np.int32) for x in success_records])
        ends = np.concatenate(
            [np.arange(len(x[2]), dtype=np.int32) == len(x[2]) - 1 for x in success_records]
        )
        np.savez_compressed(
            path,
            observations=samples,
            actions=commands,
            next_observations=successors,
            seed=ep_ids,
            done=ends,
            density=np.asarray(density, dtype=np.float32),
        )
        saved_count = int(len(samples))
    else:
        path = None
        saved_count = 0

    group = {
        "traffic_density": density,
        "num_scenarios": len(episodes),
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "collision_count": sum(x["collision"] for x in episodes),
        "collision_rate": sum(x["collision"] for x in episodes) / len(episodes),
        "truncated_count": sum(x["truncated_without_success"] for x in episodes),
        "steering_active_fraction": float(
            np.average([x["steering_active_fraction_abs_gt_0_10"] for x in episodes],
                       weights=[x["steps"] for x in episodes])
        ),
        "brake_fraction": float(
            np.average([x["brake_fraction_pedal_lt_neg_0_05"] for x in episodes],
                       weights=[x["steps"] for x in episodes])
        ),
        "saved_success_transitions": saved_count,
        "saved_success_path": str(path) if path else None,
        "episodes": episodes,
    }
    print(
        f"density={density:.2f}, success={success_count}/{len(episodes)}, "
        f"collision={group['collision_count']}, timeout={group['truncated_count']}, "
        f"active_steer={group['steering_active_fraction']:.3f}, "
        f"braking={group['brake_fraction']:.3f}, "
        f"successful_transitions={saved_count}",
        flush=True,
    )
    all_groups.append(group)

report = {
    "date_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "D0c data coverage gate (not model training)",
    "setup": {
        "simulator": "MetaDrive 0.4.3",
        "map": 3,
        "seeds": list(scenario_seeds),
        "max_steps": max_steps,
        "expert": "deterministic NumPy PPO expert",
        "input_dim": 259,
        "output_dim": 2,
        "only_manipulated_variable": "traffic_density",
    },
    "caveat": "Steering and braking action fractions are behavior proxies, NOT verified following, lead-vehicle braking, lane-change, or intersection coverage.",
    "results": all_groups,
}
out = report_dir / "expert_traffic_coverage.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Audit report saved to {out}")
