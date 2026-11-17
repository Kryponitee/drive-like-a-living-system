#!/usr/bin/env python3
"""D2j: extend the proven official SB3-Contrib RecurrentPPO pipeline to 300k frames.

Pure continuation from the exact D2i 16,384-frame checkpoint. Keeps the same
official LSTM architecture, PPO hyperparameters and saved VecNormalize stats.
Moves training to disjoint MetaDrive scenario seed blocks and evaluates both
source and extended policies on identical NEW scenes.
"""
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from metadrive.envs.metadrive_env import MetaDriveEnv

ROOT = Path("/mnt/datasets/lxb/dls")
STARTED_FRAMES = 16_384
TARGET_FRAMES = 300_032  # precisely 16,384 + 554 * (4 * 128) = 300,032
N_ENVS = 4
EVAL_SEEDS = list(range(5500, 5524))
TRAIN_DENSITY = 0.1


class SeedRangeMetaDriveEnv(MetaDriveEnv):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            start = int(self.start_index)
            n = int(self.num_scenarios)
            assert n > 0
            incoming = int(seed)
            if not (start <= incoming < start + n):
                seed = start + (incoming % n)
        assert options is None or options == {}
        return super().reset(seed=seed)


def make_train_env(rank):
    return SeedRangeMetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": TRAIN_DENSITY,
        "map": 3,
        "start_seed": 6000 + rank * 80,
        "num_scenarios": 80,
        "horizon": 800,
        "log_level": 50,
    })


def snapshot_normalizer(vec):
    return {
        "mean": np.asarray(vec.obs_rms.mean, dtype=np.float64).copy(),
        "var": np.asarray(vec.obs_rms.var, dtype=np.float64).copy(),
        "epsilon": float(vec.epsilon),
        "clip_obs": float(vec.clip_obs),
    }


def normalize(raw_obs, snap):
    x = np.asarray(raw_obs, dtype=np.float32)
    assert x.shape == (259,) and np.isfinite(x).all()
    y = (x.astype(np.float64) - snap["mean"]) / np.sqrt(snap["var"] + snap["epsilon"])
    return np.clip(y, -snap["clip_obs"], snap["clip_obs"]).astype(np.float32).reshape(1, -1)


def eval_policy(model, snap):
    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": TRAIN_DENSITY,
        "map": 3,
        "start_seed": EVAL_SEEDS[0],
        "num_scenarios": len(EVAL_SEEDS),
        "horizon": 800,
        "log_level": 50,
    })
    results = []
    for scene_seed in EVAL_SEEDS:
        obs, _ = env.reset(seed=scene_seed)
        memory = None
        episode_start = np.array([True], dtype=bool)
        return_total = 0.0
        info = {}
        for t in range(800):
            action, memory = model.predict(
                normalize(obs, snap),
                state=memory,
                episode_start=episode_start,
                deterministic=True,
            )
            assert action.shape == (1,2) and np.isfinite(action).all()
            obs, reward, terminated, truncated, info = env.step(
                np.clip(action[0], -1, 1).astype(np.float32)
            )
            episode_start[:] = bool(terminated or truncated)
            return_total += float(reward)
            if terminated or truncated:
                break
        success = bool(info.get("arrive_dest", False))
        collision = bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
        offroad = bool(info.get("out_of_road", False))
        results.append({
            "seed": scene_seed, "steps": t+1,
            "success": success, "collision": collision,
            "out_of_road": offroad,
            "timeout": not (success or collision or offroad),
            "mean_route_completion": float(info.get("route_completion", 0.0)),
            "raw_episode_reward": return_total,
        })
    env.close()
    return results


def summary(episodes):
    return {
        "success": sum(e["success"] for e in episodes),
        "collision": sum(e["collision"] for e in episodes),
        "offroad": sum(e["out_of_road"] for e in episodes),
        "timeout": sum(e["timeout"] for e in episodes),
        "mean_reward": float(np.mean([e["raw_episode_reward"] for e in episodes])),
        "mean_completion": float(np.mean([e["mean_route_completion"] for e in episodes])),
    }


def main():
    assert os.path.ismount("/mnt/datasets")
    import stable_baselines3, sb3_contrib, gymnasium
    assert stable_baselines3.__version__ == sb3_contrib.__version__ == "2.6.0"
    assert gymnasium.__version__ == "1.1.1" and torch.__version__.startswith("2.5.1")
    torch.set_num_threads(2)
    random.seed(811)
    np.random.seed(811)
    torch.manual_seed(811)

    source_checkpoint = ROOT / "checkpoints/d2i/official-recurrentppo-16k.zip"
    source_stats = ROOT / "checkpoints/d2i/official-recurrentppo-vecnormalize-16k.pkl"
    dst_checkpoint = ROOT / "checkpoints/d2j/official-recurrentppo-300k.zip"
    dst_stats = ROOT / "checkpoints/d2j/official-recurrentppo-vecnormalize-300k.pkl"
    dst_report = ROOT / "reports/d2j/official-ppo-300k.json"
    for path in (source_checkpoint, source_stats):
        assert path.is_file(), path
    assert not any(p.exists() for p in (dst_checkpoint, dst_stats, dst_report))
    for p in (dst_checkpoint.parent, dst_report.parent, ROOT/"tensorboard/d2j"):
        p.mkdir(parents=True, exist_ok=True)

    from stable_baselines3.common.vec_env import VecNormalize
    vec = SubprocVecEnv(
        [lambda worker=i: make_train_env(worker) for i in range(N_ENVS)],
        start_method="spawn",
    )
    vec = VecMonitor(vec)
    vec = VecNormalize.load(str(source_stats), vec)
    assert vec.training and vec.norm_reward and vec.norm_obs
    before_normalization = snapshot_normalizer(vec)
    model = RecurrentPPO.load(str(source_checkpoint), env=vec, device="cpu")
    assert model.num_timesteps == STARTED_FRAMES
    assert model.n_steps == 128 and model.n_envs == N_ENVS
    assert model.batch_size == 128 and model.n_epochs == 4
    assert model.seed == 811
    source_for_eval = RecurrentPPO.load(str(source_checkpoint), device="cpu")

    remaining = TARGET_FRAMES - STARTED_FRAMES
    assert remaining > 0 and remaining % (N_ENVS * model.n_steps) == 0
    print(f"Continuing official RecurrentPPO from {STARTED_FRAMES} to {TARGET_FRAMES} total frames", flush=True)
    started = time.monotonic()
    model.tensorboard_log = str(ROOT / "tensorboard/d2j")
    model.learn(total_timesteps=remaining, reset_num_timesteps=False, log_interval=16)
    train_seconds = time.monotonic() - started
    assert model.num_timesteps == TARGET_FRAMES, model.num_timesteps
    model.save(str(dst_checkpoint))
    vec.save(str(dst_stats))
    after_normalization = snapshot_normalizer(vec)
    # Evaluation uses frozen observation statistics and UNNORMALIZED environment rewards.
    vec.training = False
    vec.norm_reward = False

    print("Evaluating original 16k model on fresh scenes...", flush=True)
    before_eval = eval_policy(source_for_eval, before_normalization)
    print("Evaluating continued 300k model on same fresh scenes...", flush=True)
    after_eval = eval_policy(model, after_normalization)
    assert [e["seed"] for e in before_eval] == [e["seed"] for e in after_eval] == EVAL_SEEDS
    before = summary(before_eval)
    after = summary(after_eval)
    gains = sum((not old["success"]) and new["success"] for old,new in zip(before_eval,after_eval))
    losses = sum(old["success"] and (not new["success"]) for old,new in zip(before_eval,after_eval))
    report = {
        "stage":"D2j official SB3-Contrib RecurrentPPO LONG baseline continuation, not A/B/C scientific result",
        "source_workflow":"https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44024379218",
        "source_frames":STARTED_FRAMES,
        "target_frames":TARGET_FRAMES,
        "additional_training_frames":remaining,
        "training_seconds":train_seconds,
        "train_envs":N_ENVS,
        "train_seed_ranges":["6000..6079","6080..6159","6160..6239","6240..6319"],
        "eval_scenes":EVAL_SEEDS,
        "official_policy":"MlpLstmPolicy; no BC initialization",
        "sb3_version":stable_baselines3.__version__,
        "sb3_contrib_version":sb3_contrib.__version__,
        "previous_ABC_checkpoints_untouched":True,
        "normalization":"Saved VecNormalize train obs/reward stats; each checkpoint evaluated using its own frozen observation stats, raw environment reward",
        "source_16k_on_fresh_scenes":before,
        "extended_300k_on_fresh_scenes":after,
        "paired_300k_only_success":gains,
        "paired_16k_only_success":losses,
        "all_source_episodes":before_eval,
        "all_extended_episodes":after_eval,
        "checkpoint":str(dst_checkpoint),
        "vecnormalize":str(dst_stats),
        "caveat":"This is a learned independent LSTM system reference, NOT a capacity- or BC-initialization-matched comparison with A/B/C. Repeated evaluation seeds are development scenes, not final confirmation.",
    }
    dst_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"OFFICIAL D2j: 16k success={before['success']}/24, "
          f"300k success={after['success']}/24, "
          f"paired improvement={gains-losses}, "
          f"300k mean reward={after['mean_reward']:.2f}", flush=True)
    vec.close()


if __name__ == "__main__":
    main()
