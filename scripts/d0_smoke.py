#!/usr/bin/env python3
"""P0/D0：验证无图形界面 MetaDrive 环境与可复现的结构化观测。"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

data_root = Path(os.environ["DLS_DATA_ROOT"]).resolve()
assert str(data_root).startswith("/mnt/datasets/lxb/dls")
report_dir = data_root / "reports" / "d0"
report_dir.mkdir(parents=True, exist_ok=True)

env = MetaDriveEnv(
    {
        "use_render": False,
        "image_observation": False,
        "traffic_density": 0.1,
        "num_scenarios": 5,
        "start_seed": 0,
        "log_level": 50,
    }
)

records = []
first_obs = None
for seed in (0, 1, 0):
    obs, info = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    assert obs.ndim == 1 and obs.size > 4
    assert np.isfinite(obs).all()
    if first_obs is None:
        first_obs = obs.copy()
    if len(records) == 2:
        assert np.allclose(first_obs, obs, atol=1e-5), "Seed-reset observation not reproducible"

    result = {
        "seed": seed,
        "observation_dim": int(obs.size),
        "observation_dtype": str(obs.dtype),
        "action_space": str(env.action_space),
        "steps": 0,
        "episode_reward": 0.0,
        "terminated": False,
        "truncated": False,
    }
    for _ in range(40):
        action = np.array([0.0, 0.25], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        assert obs.size == result["observation_dim"] and np.isfinite(obs).all()
        result["steps"] += 1
        result["episode_reward"] += float(reward)
        if terminated or truncated:
            result["terminated"] = bool(terminated)
            result["truncated"] = bool(truncated)
            break
    records.append(result)

env.close()

report = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "D0 initial structured-state smoke",
    "type": "infrastructure_check_not_experiment",
    "use_render": False,
    "image_observation": False,
    "runs": records,
}
report_path = report_dir / "smoke.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"Saved report to {report_path}")
