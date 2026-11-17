#!/usr/bin/env python3
"""D2l: optimizer-INCLUSIVE official RecurrentPPO throughput, 32 vs 64 workers.

Reload the exact 300k checkpoint and VecNormalize for each condition. No model
is saved/overwritten. 16,384 new environment frames and the official PPO update
loop run per condition, sequentially, with all other hyperparameters unchanged.
This is a SYSTEM performance test, not a matched learning-quality comparison.
"""

import json
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize


ROOT = Path("/mnt/datasets/lxb/dls")
MODEL = ROOT / "checkpoints/d2j/official-recurrentppo-300k.zip"
NORMALIZER = ROOT / "checkpoints/d2j/official-recurrentppo-vecnormalize-300k.pkl"
REPORT_DIR = ROOT / "reports/d2l"
N_ENV_CHOICES = (32, 64)
ADDITIONAL_FRAMES = 16_384
START_FRAMES = 300_032
N_STEPS = 128


class SeedRangeMetaDriveEnv(MetaDriveEnv):
    """SB3 VecEnv seed adapter for MetaDrive 0.4.3, no dynamics alteration."""
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            start, n = int(self.start_index), int(self.num_scenarios)
            assert n == 80
            original = int(seed)
            if not start <= original < start + n:
                seed = start + (original % n)
        assert options is None or options == {}
        return super().reset(seed=seed)


def make_env(rank):
    return SeedRangeMetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": 0.1,
        "map": 3,
        "start_seed": 14_000 + rank * 80,
        "num_scenarios": 80,
        "horizon": 800,
        "log_level": 50,
    })


class TimingCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.collection_started_at = None
        self.collection_durations = []
        self.rollout_end_times = []
        self.completed_episodes = 0

    def _on_rollout_start(self):
        self.collection_started_at = time.perf_counter()

    def _on_step(self):
        dones = self.locals.get("dones", None)
        if dones is not None:
            self.completed_episodes += int(np.asarray(dones, dtype=bool).sum())
        return True

    def _on_rollout_end(self):
        assert self.collection_started_at is not None
        now = time.perf_counter()
        self.collection_durations.append(now - self.collection_started_at)
        self.rollout_end_times.append(now)


def worker_rss(processes):
    rss = []
    for process in processes:
        p = psutil.Process(process.pid)
        assert p.is_running(), f"MetaDrive worker pid {process.pid} terminated"
        rss.append(p.memory_info().rss / (1024 ** 2))
    return {
        "sum_worker_RSS_MiB_not_unique_memory": float(sum(rss)),
        "mean_worker_RSS_MiB": float(np.mean(rss)),
        "max_worker_RSS_MiB": float(max(rss)),
    }


def one_condition(n_envs):
    assert n_envs in N_ENV_CHOICES
    assert ADDITIONAL_FRAMES % (n_envs * N_STEPS) == 0
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / f"official-ppo-train-{n_envs}workers.json"
    assert not output.exists(), f"Refusing overwrite: {output}"
    assert MODEL.is_file() and NORMALIZER.is_file()

    created = time.perf_counter()
    base = SubprocVecEnv(
        [lambda idx=i: make_env(idx) for i in range(n_envs)],
        start_method="spawn",
    )
    vec = VecNormalize.load(str(NORMALIZER), VecMonitor(base))
    assert vec.training and vec.norm_obs and vec.norm_reward
    model = RecurrentPPO.load(str(MODEL), env=vec, device="cpu")
    assert model.n_envs == n_envs and model.n_steps == N_STEPS
    assert model.batch_size == 128 and model.n_epochs == 4
    assert model.num_timesteps == START_FRAMES
    assert model.seed == 811
    assert model.policy is not None

    starting_memory = psutil.virtual_memory().available / (1024 ** 3)
    rss_start = worker_rss(base.processes)
    startup_seconds = time.perf_counter() - created
    callback = TimingCallback()

    print(f"TRAIN-START workers={n_envs} frames={ADDITIONAL_FRAMES}, "
          f"rollouts={ADDITIONAL_FRAMES // (n_envs * N_STEPS)}", flush=True)
    t0 = time.perf_counter()
    # SB3 convention: reset_num_timesteps=False means total_timesteps is
    # ADDITIONAL steps. All training observations/rewards are normalized by the
    # copy of the original saved VecNormalize statistics in this condition.
    model.learn(
        total_timesteps=ADDITIONAL_FRAMES,
        reset_num_timesteps=False,
        callback=callback,
        log_interval=99999,
    )
    total_train_seconds = time.perf_counter() - t0
    assert model.num_timesteps == START_FRAMES + ADDITIONAL_FRAMES
    n_rollouts = ADDITIONAL_FRAMES // (n_envs * N_STEPS)
    assert len(callback.collection_durations) == n_rollouts
    collection_seconds = sum(callback.collection_durations)
    non_collection_seconds = total_train_seconds - collection_seconds
    assert total_train_seconds > 0 and 0 <= non_collection_seconds < total_train_seconds

    rss_end = worker_rss(base.processes)
    end_memory = psutil.virtual_memory().available / (1024 ** 3)
    result = {
        "stage": "D2l optimizer-inclusive OFFICIAL SB3-Contrib RecurrentPPO worker scaling",
        "workers": n_envs,
        "total_additional_environment_frames": ADDITIONAL_FRAMES,
        "initial_model_frames": START_FRAMES,
        "final_model_frames_in_memory_only": model.num_timesteps,
        "n_steps_per_env_per_rollout": N_STEPS,
        "rollout_transitions": N_STEPS * n_envs,
        "number_of_rollouts": n_rollouts,
        "ppo_epochs_per_rollout": model.n_epochs,
        "ppo_minibatch_size": model.batch_size,
        "total_model_update_epochs_across_rollouts": n_rollouts * model.n_epochs,
        "approx_optimizer_minibatches": n_rollouts * model.n_epochs * (
            N_STEPS * n_envs // model.batch_size
        ),
        "wall_seconds_model_env_setup": startup_seconds,
        "wall_seconds_full_model_learn": total_train_seconds,
        "wall_seconds_rollout_collection": collection_seconds,
        "wall_seconds_non_collection_including_backward_optimizer": non_collection_seconds,
        "frames_per_second_end_to_end": ADDITIONAL_FRAMES / total_train_seconds,
        "frames_per_second_during_collection": ADDITIONAL_FRAMES / collection_seconds,
        "rollout_collection_seconds_each": callback.collection_durations,
        "completed_episodes_in_collection": callback.completed_episodes,
        "worker_memory_after_init": rss_start,
        "worker_memory_after_train": rss_end,
        "host_available_memory_GiB_before_train": starting_memory,
        "host_available_memory_GiB_after_train": end_memory,
        "official_checkpoint_unchanged": True,
        "no_checkpoint_or_vecnormalize_saved": True,
        "cpu_only_policy_and_optimizer": True,
        "optimization_frequency_differs_by_n_envs": True,
        "caveat": "Matched total frames, epochs, minibatch size and n_steps; 32 vs 64 environments have different PPO update frequencies and episode progress. Only end-to-end wall-clock speed is compared, NOT training effects or model quality.",
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"TRAIN-RESULT workers={n_envs} total_FPS={result['frames_per_second_end_to_end']:.1f} "
        f"collection_FPS={result['frames_per_second_during_collection']:.1f} "
        f"train={total_train_seconds:.1f}s "
        f"collection={collection_seconds:.1f}s "
        f"backward_and_overhead={non_collection_seconds:.1f}s",
        flush=True,
    )
    vec.close()
    return result


def main():
    assert os.path.ismount("/mnt/datasets")
    import gymnasium, stable_baselines3, sb3_contrib
    assert torch.__version__.startswith("2.5.1")
    assert gymnasium.__version__ == "1.1.1"
    assert stable_baselines3.__version__ == sb3_contrib.__version__ == "2.6.0"
    torch.set_num_threads(2)
    assert psutil.virtual_memory().available > 70 * 1024 ** 3, "Need >=70 GiB free system memory"
    previous = json.loads((ROOT / "reports/d2k-extended/worker-scaling-summary.json").read_text())
    assert previous["worker_counts_tested"] == [32, 48, 64]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "optimizer-inclusive-scaling-summary.json"
    assert not output.exists()
    runs = [one_condition(n) for n in N_ENV_CHOICES]
    for row in runs:
        assert row["total_additional_environment_frames"] == ADDITIONAL_FRAMES
        assert row["approx_optimizer_minibatches"] == 512, (
            "Expected identical total nominal minibatch count in both conditions"
        )
    reference, proposed = runs
    speedup = proposed["frames_per_second_end_to_end"] / reference["frames_per_second_end_to_end"]
    selected = 64 if speedup >= 0.9 else 32
    report = {
        "stage": "D2l final preparation systems gate: optimizer-inclusive 32 vs 64 MetaDrive workers",
        "source_D2j_model": str(MODEL),
        "source_normalizer": str(NORMALIZER),
        "source_D2k_extended": str(ROOT / "reports/d2k-extended/worker-scaling-summary.json"),
        "sequential_worker_settings": list(N_ENV_CHOICES),
        "matched_environment_steps_each": ADDITIONAL_FRAMES,
        "source_weights_unchanged": True,
        "training_updates_performed_but_discarded": True,
        "results": runs,
        "64_vs_32_full_training_throughput_ratio": speedup,
        "selected_workers_by_prespecified_10pct_gate": selected,
        "next_phase": "D2m custom A/B/C recurrent state and slow-clock adapter correctness; then D2n full training",
        "scientific_limits": "System performance only; not paired trajectory learner comparison. Workers change policy update frequency; cannot infer driving skill improvement.",
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"FINAL D2l: 32={reference['frames_per_second_end_to_end']:.1f}, "
          f"64={proposed['frames_per_second_end_to_end']:.1f} "
          f"frames/s; ratio={speedup:.2f}x; selected={selected} workers", flush=True)


if __name__ == "__main__":
    main()
