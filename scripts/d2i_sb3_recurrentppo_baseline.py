#!/usr/bin/env python3
"""D2i official SB3-Contrib RecurrentPPO baseline on MetaDrive (independent smoke).

This is NOT an A/B/C comparison: original A/B/C use customized recurrent states
and pretrained BC weights; SB3 has a newly trained built-in LSTM. The benchmark
tests mature PPO rollout, GAE, recurrent reset, value learning, and evaluation.
"""

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

ROOT = Path("/mnt/datasets/lxb/dls")
ENV_STEPS = 16_384
TRAIN_DENSITY = 0.1
TRAIN_SEED = 5000
EVAL_SEEDS = list(range(5400, 5412))
N_ENVS = 4


class SeedRangeMetaDriveEnv(MetaDriveEnv):
    """Translate SB3's global seed into this worker's MetaDrive seed range."""

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            start = int(self.start_index)
            n = int(self.num_scenarios)
            assert n > 0
            original = int(seed)
            if not (start <= original < start + n):
                seed = start + (original % n)
                print(f"SB3 seed mapping: requested={original} -> MetaDrive={seed}, "
                      f"allowed=[{start},{start + n})", flush=True)
        return super().reset(seed=seed, options=options)


def make_env(worker_index):
    # Called only inside a worker process after 'spawn'; no shared Panda3D engine.
    return SeedRangeMetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": TRAIN_DENSITY,
        "map": 3,
        "start_seed": TRAIN_SEED + worker_index * 80,
        "num_scenarios": 80,
        "horizon": 800,
        "log_level": 50,
    })


def run_evaluation(model, obs_normalizer):
    eval_env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": TRAIN_DENSITY,
        "map": 3,
        "start_seed": EVAL_SEEDS[0],
        "num_scenarios": len(EVAL_SEEDS),
        "horizon": 800,
        "log_level": 50,
    })
    summaries = []
    for scene_seed in EVAL_SEEDS:
        obs, _ = eval_env.reset(seed=scene_seed)
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        total_reward = 0.0
        last_info = {}
        for timestep in range(800):
            raw = np.asarray(obs, dtype=np.float32)
            assert raw.shape == (259,) and np.isfinite(raw).all()
            x = obs_normalizer.normalize_obs(raw.reshape(1, -1))
            action, lstm_states = model.predict(
                x, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            assert action.shape == (1, 2) and np.isfinite(action).all()
            next_obs, reward, term, trunc, last_info = eval_env.step(
                np.clip(action[0], -1., 1.).astype(np.float32)
            )
            total_reward += float(reward)
            episode_starts[:] = bool(term or trunc)
            obs = next_obs
            if term or trunc:
                break
        success = bool(last_info.get("arrive_dest", False))
        collision = bool(last_info.get("crash_vehicle", False) or last_info.get("crash_object", False))
        offroad = bool(last_info.get("out_of_road", False))
        summaries.append({
            "seed": scene_seed, "steps": timestep + 1,
            "success": success, "collision": collision,
            "out_of_road": offroad,
            "timeout": not (success or collision or offroad),
            "raw_episode_reward": total_reward,
            "route_completion": float(last_info.get("route_completion", 0.0)),
        })
    eval_env.close()
    return summaries


def main():
    assert os.path.ismount("/mnt/datasets")
    import stable_baselines3
    import sb3_contrib
    assert stable_baselines3.__version__ == "2.6.0"
    assert sb3_contrib.__version__ == "2.6.0"
    assert torch.__version__.startswith("2.5.1")
    torch.set_num_threads(2)
    random.seed(811)
    np.random.seed(811)
    torch.manual_seed(811)
    reports_dir = ROOT / "reports" / "d2i"
    checkpoints = ROOT / "checkpoints" / "d2i"
    logs = ROOT / "tensorboard" / "d2i"
    for path in (reports_dir, checkpoints, logs):
        path.mkdir(parents=True, exist_ok=True)
    model_path = checkpoints / "official-recurrentppo-16k.zip"
    vec_stats_path = checkpoints / "official-recurrentppo-vecnormalize-16k.pkl"
    output_path = reports_dir / "official-rl-smoke.json"
    assert not model_path.exists() and not vec_stats_path.exists() and not output_path.exists()

    # A unique MetaDrive engine process per worker, matching MetaDrive official examples.
    # Use 'spawn' explicitly: safer than forking an initialized PyTorch/Panda3D runtime.
    vec = SubprocVecEnv([lambda idx=i: make_env(idx) for i in range(N_ENVS)], start_method="spawn")
    vec = VecMonitor(vec)
    vec = VecNormalize(
        vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
        gamma=0.99, training=True
    )
    assert vec.observation_space.shape == (259,) and vec.action_space.shape == (2,)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec,
        n_steps=128,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=811,
        device="cpu",
        verbose=1,
        tensorboard_log=str(logs),
        policy_kwargs={
            "lstm_hidden_size": 64,
            "n_lstm_layers": 1,
            "net_arch": {"pi": [64], "vf": [64]},
            "log_std_init": -3.0,
        },
    )
    start = time.monotonic()
    model.learn(total_timesteps=ENV_STEPS, log_interval=4)
    training_elapsed = time.monotonic() - start
    assert model.num_timesteps == ENV_STEPS, (model.num_timesteps, ENV_STEPS)
    model.save(model_path)
    vec.save(vec_stats_path)
    vec.training = False
    vec.norm_reward = False
    print("Official PPO training completed, evaluating frozen deterministic LSTM...", flush=True)
    results = run_evaluation(model, vec)
    report = {
        "stage": "D2i OFFICIAL SB3-Contrib RecurrentPPO control pipeline smoke",
        "not_matching_ABC_architectures": True,
        "not_pretrained_by_BC": True,
        "original_ABC_results_untouched": True,
        "official_version_sb3": stable_baselines3.__version__,
        "official_version_sb3_contrib": sb3_contrib.__version__,
        "torch": torch.__version__,
        "n_train_environments": N_ENVS,
        "train_scenarios": "four independent seed ranges: [5000:5080,5080:5160,5160:5240,5240:5320)",
        "train_density": TRAIN_DENSITY,
        "train_steps_across_all_environments": model.num_timesteps,
        "training_seconds": training_elapsed,
        "policy_type": "MlpLstmPolicy",
        "normalization": "VecNormalize obs/reward during training; frozen obs stats/raw reward during eval",
        "eval_seeds": EVAL_SEEDS,
        "eval_success": sum(x["success"] for x in results),
        "eval_collision": sum(x["collision"] for x in results),
        "eval_out_of_road": sum(x["out_of_road"] for x in results),
        "eval_mean_reward": float(np.mean([x["raw_episode_reward"] for x in results])),
        "evaluation_episodes": results,
        "model_checkpoint": str(model_path),
        "vecnormalizer": str(vec_stats_path),
        "caveat": "16k frame smoke does not demonstrate learning convergence or any A/B/C architecture superiority. This baseline is independently randomly initialized.",
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RESULT: success={report['eval_success']}/{len(EVAL_SEEDS)}, "
          f"collisions={report['eval_collision']}, "
          f"offroad={report['eval_out_of_road']}, "
          f"mean_reward={report['eval_mean_reward']:.2f}, "
          f"training_seconds={training_elapsed:.1f}", flush=True)
    vec.close()


if __name__ == "__main__":
    main()
