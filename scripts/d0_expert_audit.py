#!/usr/bin/env python3
"""D0：检查 MetaDrive PPO 专家的可用性与最小训练数据采集，不启动 D1。"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert

data_root = Path(os.environ["DLS_DATA_ROOT"]).resolve()
assert str(data_root).startswith("/mnt/datasets/lxb/dls")

report_dir = data_root / "reports" / "d0"
data_dir = data_root / "demonstrations"
report_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

env = MetaDriveEnv(
    {
        "use_render": False,
        "image_observation": False,
        "traffic_density": 0.1,
        "map": 3,
        "start_seed": 0,
        "num_scenarios": 32,
        "horizon": 400,
        "log_level": 50,
    }
)

states = []
actions = []
next_states = []
episode_ids = []
done_flags = []
episode_results = []

for seed in range(12):
    obs, info = env.reset(seed=seed)
    episode_reward = 0.0
    last_info = info
    final_done = False
    for step_idx in range(400):
        obs_array = np.asarray(obs, dtype=np.float32)
        action = np.clip(np.asarray(expert(env.agent, deterministic=True), dtype=np.float32), -1, 1)
        assert action.shape == (2,), action.shape
        assert np.isfinite(action).all()
        obs_next, reward, terminated, truncated, info = env.step(action)
        obs_next_array = np.asarray(obs_next, dtype=np.float32)
        assert obs_next_array.shape == obs_array.shape
        assert np.isfinite(obs_array).all() and np.isfinite(obs_next_array).all()

        states.append(obs_array)
        actions.append(action)
        next_states.append(obs_next_array)
        episode_ids.append(seed)
        done_flags.append(bool(terminated or truncated))
        episode_reward += float(reward)
        obs = obs_next
        last_info = info
        final_done = bool(terminated or truncated)
        if final_done:
            break

    episode_results.append(
        {
            "seed": seed,
            "steps": step_idx + 1,
            "done": final_done,
            "arrive_dest": bool(last_info.get("arrive_dest", False)),
            "out_of_road": bool(last_info.get("out_of_road", False)),
            "crash_vehicle": bool(last_info.get("crash_vehicle", False)),
            "crash_object": bool(last_info.get("crash_object", False)),
            "reward_sum": episode_reward,
        }
    )

env.close()

npz_path = data_dir / "d0-expert-probe.npz"
np.savez_compressed(
    npz_path,
    observations=np.stack(states),
    actions=np.stack(actions),
    next_observations=np.stack(next_states),
    episode_ids=np.asarray(episode_ids, dtype=np.int32),
    done=np.asarray(done_flags, dtype=np.bool_),
)
report = {
    "stage": "D0 expert data coverage probe, not a training result",
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "episodes": episode_results,
    "num_transitions": len(states),
    "observation_dim": len(states[0]),
    "success_rate": sum(x["arrive_dest"] for x in episode_results) / len(episode_results),
    "crash_rate": sum(x["crash_vehicle"] or x["crash_object"] for x in episode_results) / len(episode_results),
    "out_of_road_rate": sum(x["out_of_road"] for x in episode_results) / len(episode_results),
    "data_path": str(npz_path),
    "data_bytes": npz_path.stat().st_size,
}
report_path = report_dir / "expert_audit.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
