#!/usr/bin/env python3
"""D2k: sample-only MetaDrive CPU worker scaling on a frozen official RecurrentPPO.

Sequential 4, 8, 16, 32 workers; identical policy, obs normalizer and 8192
aggregate environment transitions per worker-count setting.
"""
import json
import os
from pathlib import Path
import time
import psutil

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

ROOT = Path("/mnt/datasets/lxb/dls")
CHECKPOINT = ROOT / "checkpoints/d2j/official-recurrentppo-300k.zip"
NORMALIZER = ROOT / "checkpoints/d2j/official-recurrentppo-vecnormalize-300k.pkl"
REPORT_DIR = ROOT / "reports/d2k"
COUNTS = (4, 8, 16, 32)
TOTAL_MEASURED_FRAMES = 8192
WARMUP_STEPS_PER_ENV = 2


class SeedRangeMetaDriveEnv(MetaDriveEnv):
    """Adapt SB3's global RNG reset seeds to each worker's own MetaDrive range."""
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            start, n = int(self.start_index), int(self.num_scenarios)
            original = int(seed)
            assert n >= 64
            if not start <= original < start + n:
                seed = start + original % n
        assert options is None or options == {}
        return super().reset(seed=seed)


def make_env(rank):
    return SeedRangeMetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": 0.1,
        "map": 3,
        "start_seed": 8100 + rank * 80,
        "num_scenarios": 80,
        "horizon": 800,
        "log_level": 50,
    })


def process_mem_mib(processes):
    rss = []
    for p in processes:
        info = psutil.Process(p.pid)
        assert info.is_running()
        rss.append(info.memory_info().rss / 1048576)
    return rss


def evaluate_workers(n):
    assert n in COUNTS and TOTAL_MEASURED_FRAMES % n == 0
    assert CHECKPOINT.exists() and NORMALIZER.exists()
    print(f"Starting {n} independent MetaDrive workers...", flush=True)
    started = time.perf_counter()
    base = SubprocVecEnv([lambda rank=i: make_env(rank) for i in range(n)],
                        start_method="spawn")
    wrapped = VecMonitor(base)
    vec = VecNormalize.load(str(NORMALIZER), wrapped)
    vec.training = False
    vec.norm_reward = False
    model = RecurrentPPO.load(str(CHECKPOINT), device="cpu")
    assert model.policy is not None
    assert vec.num_envs == n
    assert vec.observation_space.shape == (259,)
    assert vec.action_space.shape == (2,)
    reset_obs = vec.reset()
    assert reset_obs.shape == (n, 259)
    setup_seconds = time.perf_counter() - started

    hidden_state = None
    starts = np.ones((n,), dtype=bool)
    completed = 0
    frames_warmup = 0
    with torch.no_grad():
        for _ in range(WARMUP_STEPS_PER_ENV):
            actions, hidden_state = model.predict(
                reset_obs, state=hidden_state, episode_start=starts,
                deterministic=True
            )
            assert actions.shape == (n, 2)
            reset_obs, rewards, dones, infos = vec.step(actions)
            starts = np.asarray(dones, dtype=bool)
            frames_warmup += n
        # Measurement uses the same frozen policy but starts after warmup.
        host_memory_before = psutil.virtual_memory().available / 1048576
        rss_mib = process_mem_mib(base.processes)
        measured_started = time.perf_counter()
        for _ in range(TOTAL_MEASURED_FRAMES // n):
            actions, hidden_state = model.predict(
                reset_obs, state=hidden_state, episode_start=starts,
                deterministic=True
            )
            assert actions.shape == (n, 2) and np.isfinite(actions).all()
            reset_obs, rewards, dones, infos = vec.step(actions)
            starts = np.asarray(dones, dtype=bool)
            completed += int(np.asarray(dones).sum())
        elapsed = time.perf_counter() - measured_started
        host_memory_after = psutil.virtual_memory().available / 1048576
        final_rss_mib = process_mem_mib(base.processes)
        assert np.isfinite(reset_obs).all()
        assert elapsed > 0 and TOTAL_MEASURED_FRAMES / elapsed > 0
    vec.close()
    results = {
        "workers": n,
        "total_measured_env_steps": TOTAL_MEASURED_FRAMES,
        "per_worker_measured_steps": TOTAL_MEASURED_FRAMES // n,
        "warmup_steps_per_worker": WARMUP_STEPS_PER_ENV,
        "setup_and_first_reset_seconds": setup_seconds,
        "measurement_wall_seconds": elapsed,
        "aggregate_steps_per_second": TOTAL_MEASURED_FRAMES / elapsed,
        "completed_episodes_during_measurement": completed,
        "mean_worker_rss_mib_after_warmup": float(np.mean(rss_mib)),
        "mean_worker_rss_mib_after_measurement": float(np.mean(final_rss_mib)),
        "sum_worker_rss_mib_after_measurement": float(np.sum(final_rss_mib)),
        "worker_memory_rss_note": "Sum of per-process RSS double-counts shared mapped pages; not unique host RAM usage",
        "host_available_memory_mib_before_measurement": host_memory_before,
        "host_available_memory_mib_after_measurement": host_memory_after,
        "actor_model_unchanged": True,
        "normalization_frozen": True,
        "new_training_updates": 0,
    }
    print(f"RESULT workers={n}: {results['aggregate_steps_per_second']:.1f} "
          f"environment steps/sec, {elapsed:.1f}s measured, "
          f"{setup_seconds:.1f}s startup; worker_RSS_sum={np.sum(final_rss_mib):.0f} MiB",
          flush=True)
    return results


def main():
    assert os.path.ismount("/mnt/datasets")
    import gymnasium, stable_baselines3, sb3_contrib
    assert torch.__version__.startswith("2.5.1")
    assert gymnasium.__version__ == "1.1.1"
    assert stable_baselines3.__version__ == sb3_contrib.__version__ == "2.6.0"
    torch.set_num_threads(2)
    source = json.loads((ROOT/"reports/d2j/official-ppo-300k.json").read_text())
    assert source["target_frames"] == 300032
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = REPORT_DIR/"worker-scaling-summary.json"
    assert not result_path.exists()
    results = []
    for workers in COUNTS:
        point = REPORT_DIR / f"workers-{workers}.json"
        assert not point.exists()
        measurement = evaluate_workers(workers)
        point.write_text(json.dumps(measurement, indent=2), encoding="utf-8")
        results.append(measurement)

    raw_best = max(results, key=lambda r:r["aggregate_steps_per_second"])
    # Prefer a smaller worker count when the larger speed is within ten percent.
    eligible = [x for x in results
                if x["aggregate_steps_per_second"] >= 0.9 * raw_best["aggregate_steps_per_second"]]
    chosen = min(eligible, key=lambda x:x["workers"])
    report = {
        "stage": "D2k frozen official LSTM + VecNormalize CPU sampling-only worker scaling",
        "source_checkpoint": str(CHECKPOINT),
        "source_vecnormalize": str(NORMALIZER),
        "worker_counts_tested": list(COUNTS),
        "measured_env_steps_each": TOTAL_MEASURED_FRAMES,
        "scenario_seed_start_for_worker_rank_0": 8100,
        "only_worker_count_changed": True,
        "training_updates": 0,
        "results": results,
        "fastest_observed_workers": raw_best["workers"],
        "best_steps_per_second": raw_best["aggregate_steps_per_second"],
        "selected_workers_within_10pct_of_fastest": chosen["workers"],
        "selected_steps_per_second": chosen["aggregate_steps_per_second"],
        "speedup_vs_4": chosen["aggregate_steps_per_second"]/results[0]["aggregate_steps_per_second"],
        "caveat": "Only sampling with frozen model forward pass; benchmark PPO update overhead and time-limit handling separately before allocating long RL runs. Different worker count changes per-worker scene exposure.",
    }
    result_path.write_text(json.dumps(report,indent=2), encoding="utf-8")
    print(f"FINAL recommended workers={chosen['workers']}, measured speedup="
          f"{report['speedup_vs_4']:.2f}x versus 4 workers", flush=True)


if __name__ == "__main__":
    main()
