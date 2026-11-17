#!/usr/bin/env python3
"""D2l: full RecurrentPPO MetaDrive timing, 32 vs 64 workers; no scientific policy comparison.

Both trials reload the same D2j 300k model and VecNormalize snapshots.
Each performs 32,768 actual environment transitions and PPO optimization.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

ROOT = Path("/mnt/datasets/lxb/dls")
SOURCE_MODEL = ROOT / "checkpoints/d2j/official-recurrentppo-300k.zip"
SOURCE_STATS = ROOT / "checkpoints/d2j/official-recurrentppo-vecnormalize-300k.pkl"
OUT = ROOT / "reports/d2l"
FRAMES = 32768
COUNTS = (32, 64)
STEPS_PER_ENV = 128


class SeedRangeMetaDriveEnv(MetaDriveEnv):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            start, n = int(self.start_index), int(self.num_scenarios)
            seed = int(seed)
            if not start <= seed < start + n:
                seed = start + seed % n
        assert options is None or options == {}
        return super().reset(seed=seed)


def make_env(rank):
    return SeedRangeMetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": 0.1,
        "map": 3,
        "start_seed": 16000 + rank * 80,
        "num_scenarios": 80,
        "horizon": 800,
        "log_level": 50,
    })


def measure(n):
    assert n in COUNTS and FRAMES % (n * STEPS_PER_ENV) == 0
    assert SOURCE_MODEL.is_file() and SOURCE_STATS.is_file()
    output = OUT / f"full-ppo-{n}-workers.json"
    assert not output.exists(), f"Output exists; no overwrite: {output}"
    print(f"Starting official 32,768-frame PPO optimizer-inclusive audit with {n} workers", flush=True)
    setup_start = time.perf_counter()
    base = SubprocVecEnv(
        [lambda idx=i: make_env(idx) for i in range(n)],
        start_method="spawn",
    )
    vec = VecNormalize.load(str(SOURCE_STATS), VecMonitor(base))
    assert vec.num_envs == n and vec.training and vec.norm_obs and vec.norm_reward
    assert vec.observation_space.shape == (259,) and vec.action_space.shape == (2,)
    model = RecurrentPPO.load(str(SOURCE_MODEL), env=vec, device="cpu")
    assert model.n_steps == STEPS_PER_ENV
    assert model.batch_size == 128 and model.n_epochs == 4
    assert model.n_envs == n and model.num_timesteps == 300032
    assert model.seed == 811
    setup_sec = time.perf_counter() - setup_start

    train_calls = 0
    train_wall = 0.0
    optimizer_steps = 0
    optimizer_wall = 0.0
    original_train = model.train
    original_step = model.policy.optimizer.step

    def counted_train(*args, **kwargs):
        nonlocal train_calls, train_wall
        start = time.perf_counter()
        result = original_train(*args, **kwargs)
        train_calls += 1
        train_wall += time.perf_counter() - start
        return result

    def counted_optimizer_step(*args, **kwargs):
        nonlocal optimizer_steps, optimizer_wall
        start = time.perf_counter()
        result = original_step(*args, **kwargs)
        optimizer_steps += 1
        optimizer_wall += time.perf_counter() - start
        return result

    model.train = counted_train
    model.policy.optimizer.step = counted_optimizer_step
    available_ram_before_gib = psutil.virtual_memory().available / (1024 ** 3)
    start = time.perf_counter()
    # reset_num_timesteps=True standardizes the same 32768-step local schedule
    # in both conditions (no continuation or overwrite of the D2j model).
    model.learn(total_timesteps=FRAMES, reset_num_timesteps=True, log_interval=2)
    learn_seconds = time.perf_counter() - start
    available_ram_after_gib = psutil.virtual_memory().available / (1024 ** 3)
    assert model.num_timesteps == FRAMES, (model.num_timesteps, FRAMES)
    assert train_calls == FRAMES // (n * STEPS_PER_ENV), (train_calls, n)
    assert optimizer_steps > 0 and 0 < optimizer_wall <= train_wall <= learn_seconds
    assert all(p.isfinite().all() for p in model.policy.parameters())
    per_worker_rss = [psutil.Process(p.pid).memory_info().rss / (1024 ** 2)
                      for p in base.processes]
    assert len(per_worker_rss) == n

    result = {
        "stage": "D2l official SB3-Contrib optimizer-INCLUSIVE full PPO CPU worker test",
        "workers": n,
        "total_training_frames": FRAMES,
        "n_steps_per_env": STEPS_PER_ENV,
        "rollout_size_per_update": n * STEPS_PER_ENV,
        "batch_size": model.batch_size,
        "ppo_epochs_per_update": model.n_epochs,
        "ppo_outer_updates": train_calls,
        "optimizer_steps": optimizer_steps,
        "training_setup_wall_seconds": setup_sec,
        "learn_wall_seconds": learn_seconds,
        "train_function_wall_seconds": train_wall,
        "optimizer_step_wall_seconds": optimizer_wall,
        "collection_and_other_wall_seconds": learn_seconds - train_wall,
        "frames_per_second_optimizer_inclusive": FRAMES / learn_seconds,
        "source_model": str(SOURCE_MODEL),
        "source_vecnormalization": str(SOURCE_STATS),
        "source_checkpoint_unchanged": True,
        "actor_architecture": "official MlpLstmPolicy, not custom A/B/C",
        "training_seed_ranges": [f"{16000+i*80}..{16000+i*80+79}" for i in range(n)],
        "RSS_worker_sum_MiB": float(sum(per_worker_rss)),
        "RSS_note": "Summed RSS double-counts shared pages; not unique memory footprint",
        "host_RAM_available_before_gib": available_ram_before_gib,
        "host_RAM_available_after_gib": available_ram_after_gib,
        "no_new_bc_data": True,
        "no_checkpoint_output": True,
        "interpretation": "Systems speed only: n_steps=128 fixed, so rollout batch and outer update frequency differ; total frames and counted PPO minibatch optimizer steps can be compared. No policy success comparison.",
    }
    vec.close()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"D2l finished {n} workers: {FRAMES / learn_seconds:.1f} full PPO frames/s, "
          f"learn={learn_seconds:.2f}s, sampling+overhead={learn_seconds-train_wall:.2f}s, "
          f"training={train_wall:.2f}s, optimizer steps={optimizer_steps}", flush=True)
    del model, vec, base
    gc.collect()
    return result


def main():
    assert os.path.ismount("/mnt/datasets")
    import stable_baselines3, sb3_contrib, gymnasium
    assert stable_baselines3.__version__ == sb3_contrib.__version__ == "2.6.0"
    assert gymnasium.__version__ == "1.1.1"
    assert torch.__version__.startswith("2.5.1")
    assert psutil.cpu_count(logical=True) >= 64
    torch.set_num_threads(2)
    torch.manual_seed(811)
    np.random.seed(811)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = OUT / "full-ppo-worker-comparison.json"
    assert not summary.exists()
    source = json.loads((ROOT/"reports/d2j/official-ppo-300k.json").read_text())
    assert source["target_frames"] == 300032
    rows = [measure(n) for n in COUNTS]
    assert [r["workers"] for r in rows] == [32, 64]
    assert rows[0]["optimizer_steps"] == rows[1]["optimizer_steps"], (
        "Mismatched PPO optimizer step counts; invalid matched-optimization interpretation"
    )
    speedup = rows[1]["frames_per_second_optimizer_inclusive"] / rows[0]["frames_per_second_optimizer_inclusive"]
    report = {
        "stage": "D2l full PPO optimizer-inclusive 32-vs-64 worker throughput decision",
        "results": rows,
        "equal_environment_frames": True,
        "equal_minibatch_optimizer_steps": True,
        "worker_64_speedup_vs_32": speedup,
        "decision_rule": "Prefer 64 only if its full-PPO throughput improves at least 10% with no OOM; otherwise use 32",
        "preferred_worker_count": 64 if speedup >= 1.10 else 32,
        "caveat": "Changing workers at fixed n_steps changes rollout batch/outer update cadence; for systems engineering only, not training outcome comparison.",
    }
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"D2l FINAL: 32={rows[0]['frames_per_second_optimizer_inclusive']:.1f}, "
          f"64={rows[1]['frames_per_second_optimizer_inclusive']:.1f} frames/s; "
          f"64/32={speedup:.3f}x; preferred={report['preferred_worker_count']} workers", flush=True)


if __name__ == "__main__":
    main()
