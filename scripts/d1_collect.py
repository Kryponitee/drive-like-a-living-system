#!/usr/bin/env python3
"""D1 成功专家示范采集：每进程一个 MetaDrive 环境，按 seed 切分，绝不混入 D0/测试种子。"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--episodes-per-worker", type=int, required=True)
    args = parser.parse_args()

    assert args.workers >= 3 and args.workers % 3 == 0
    assert 0 <= args.worker < args.workers
    assert args.episodes_per_worker >= 1
    data_root = Path(os.environ["DLS_DATA_ROOT"]).resolve()
    assert data_root == Path("/mnt/datasets/lxb/dls")
    assert os.path.ismount("/mnt/datasets")
    split_start = {"train": 1000, "val": 2000}[args.split]
    base_seed = split_start + args.worker * args.episodes_per_worker
    assert base_seed + args.episodes_per_worker <= (1600 if args.split == "train" else 2200)
    density = (0.0, 0.1, 0.25)[args.worker % 3]
    outdir = data_root / "demonstrations" / "d1-v1" / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"worker-{args.worker:02d}-density-{density:.2f}"
    archive_path = outdir / f"{stem}.npz"
    manifest_path = outdir / f"{stem}.json"
    assert not archive_path.exists() and not manifest_path.exists(), "Existing D1 shard must not be overwritten"
    started = time.monotonic()
    env = MetaDriveEnv(
        {
            "use_render": False,
            "image_observation": False,
            "traffic_density": density,
            "map": 3,
            "start_seed": base_seed,
            "num_scenarios": args.episodes_per_worker,
            "horizon": 800,
            "log_level": 50,
        }
    )
    arrays_obs = []
    arrays_act = []
    arrays_next = []
    episodes = []
    kept_seeds = []
    kept_lengths = []
    total_steps = 0
    for seed in range(base_seed, base_seed + args.episodes_per_worker):
        obs, _ = env.reset(seed=seed)
        o_buffer, a_buffer, n_buffer = [], [], []
        last_info = {}
        for t in range(800):
            state = np.asarray(obs, dtype=np.float32)
            action = np.clip(np.asarray(expert(env.agent, deterministic=True), dtype=np.float32), -1, 1)
            assert state.shape == (259,) and action.shape == (2,)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            next_obs, reward, terminated, truncated, last_info = env.step(action)
            next_state = np.asarray(next_obs, dtype=np.float32)
            assert next_state.shape == state.shape and np.isfinite(next_state).all()
            o_buffer.append(state)
            a_buffer.append(action)
            n_buffer.append(next_state)
            obs = next_obs
            if terminated or truncated:
                break
        success = bool(last_info.get("arrive_dest", False))
        collided = bool(last_info.get("crash_vehicle", False) or last_info.get("crash_object", False))
        episodes.append({"seed": seed, "steps": len(a_buffer), "success": success,
                         "collision": collided, "out_of_road": bool(last_info.get("out_of_road", False)),
                         "timeout": not success and not collided and not last_info.get("out_of_road", False)})
        total_steps += len(a_buffer)
        if success:
            kept_seeds.append(seed)
            kept_lengths.append(len(a_buffer))
            arrays_obs.append(np.stack(o_buffer))
            arrays_act.append(np.stack(a_buffer))
            arrays_next.append(np.stack(n_buffer))
    env.close()

    successful = len(kept_seeds)
    saved_steps = int(sum(kept_lengths))
    if successful:
        payload = {
            "observations": np.concatenate(arrays_obs).astype(np.float32),
            "actions": np.concatenate(arrays_act).astype(np.float32),
            "next_observations": np.concatenate(arrays_next).astype(np.float32),
            "episode_offsets": np.concatenate(([0], np.cumsum(kept_lengths))).astype(np.int64),
            "episode_seeds": np.asarray(kept_seeds, dtype=np.int32),
            "traffic_density": np.asarray([density] * successful, dtype=np.float32),
        }
    else:
        payload = {
            "observations": np.empty((0, 259), dtype=np.float32),
            "actions": np.empty((0, 2), dtype=np.float32),
            "next_observations": np.empty((0, 259), dtype=np.float32),
            "episode_offsets": np.zeros((1,), dtype=np.int64),
            "episode_seeds": np.empty((0,), dtype=np.int32),
            "traffic_density": np.empty((0,), dtype=np.float32),
        }
    assert payload["episode_offsets"][-1] == saved_steps
    np.savez_compressed(archive_path, **payload)
    sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    report = {
        "split": args.split, "worker": args.worker, "density": density,
        "seed_range": [base_seed, base_seed + args.episodes_per_worker - 1],
        "num_attempted": len(episodes), "num_successful": successful,
        "num_collision": sum(x["collision"] for x in episodes),
        "total_simulated_steps": total_steps, "saved_success_steps": saved_steps,
        "elapsed_seconds": time.monotonic() - started,
        "archive": str(archive_path), "archive_sha256": sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes,
        "configuration": {
            "metadrive": "0.4.3", "map": 3, "horizon": 800,
            "expert": "numpy PPO deterministic", "image_observation": False,
            "input_dim": 259, "action_dim": 2,
        },
    }
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
