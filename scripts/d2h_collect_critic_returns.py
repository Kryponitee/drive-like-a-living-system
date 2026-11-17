#!/usr/bin/env python3
"""D2h: collect complete trajectories for value-only supervision with FROZEN BC actors.

Fresh, disjoint train/validation seeds. No expert queries; no actor training. All
arrays and results are written to the separately mounted DLS data volume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from dls_p0 import model_info
from dls_recurrent_ppo import IdenticalCritic

DATA = Path("/mnt/datasets/lxb/dls")
GAMMA = 0.99
SCENES = {"train": tuple(range(4600, 4624)), "validation": tuple(range(4700, 4716))}
SIGMA = 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    args = parser.parse_args()

    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    bc_path = DATA / "checkpoints" / "d1-v1" / f"{args.variant}-seed42.pt"
    baseline_path = DATA / "checkpoints" / "d2f-v1" / f"{args.variant}-seed42.pt"
    bc = torch.load(bc_path, map_location="cpu", weights_only=False)
    previous = torch.load(baseline_path, map_location="cpu", weights_only=False)
    assert bc["variant"] == previous["variant"] == args.variant
    actor, _ = model_info(args.variant)
    actor.load_state_dict(bc["state_dict"], strict=True)
    actor.eval().requires_grad_(False)
    old_critic = IdenticalCritic()
    old_critic.load_state_dict(previous["critic_state_dict"], strict=True)
    old_critic.eval().requires_grad_(False)
    mean = np.asarray(bc["train_mean"], dtype=np.float32)
    std = np.asarray(bc["train_std"], dtype=np.float32)
    assert mean.shape == std.shape == (259,) and float(std.min()) > 0
    assert np.array_equal(mean, np.asarray(previous["train_mean"], dtype=np.float32))
    seeds = SCENES[args.split]

    output = DATA / "rollouts" / "d2h-v1"
    output.mkdir(parents=True, exist_ok=True)
    target_npz = output / f"{args.split}-{args.variant}.npz"
    target_json = output / f"{args.split}-{args.variant}.json"
    assert not target_npz.exists() and not target_json.exists(), "Refuse accidental dataset overwrite"

    env = MetaDriveEnv({
        "use_render": False, "image_observation": False,
        "traffic_density": 0.1, "map": 3,
        "start_seed": seeds[0], "num_scenarios": len(seeds),
        "horizon": 800, "log_level": 50,
    })
    all_x, all_ret, all_old_v, all_ep_id, all_valid = [], [], [], [], []
    episodes = []
    offsets = [0]
    started = time.monotonic()

    with torch.no_grad():
        for scene_seed in seeds:
            obs, _ = env.reset(seed=scene_seed)
            state = actor.initial_state(1, torch.device("cpu"))
            # Same scene-indexed Gaussian samples for every A/B/C variant.
            rng = torch.Generator(device="cpu").manual_seed(600000 + scene_seed)
            ep_x, ep_rewards, ep_old_v = [], [], []
            last_info = {}
            episode_terminated = episode_truncated = False
            for t in range(800):
                raw = np.asarray(obs, dtype=np.float32)
                assert raw.shape == (259,) and np.isfinite(raw).all()
                x = np.clip((raw - mean) / std, -10.0, 10.0).copy()
                xt = torch.from_numpy(x).unsqueeze(0)
                mean_action, state = actor.step(xt, state, t)
                assert mean_action.shape == (1, 2) and torch.isfinite(mean_action).all()
                assert float(mean_action.abs().max()) < 0.999
                u = torch.atanh(mean_action) + SIGMA * torch.randn((1, 2), generator=rng)
                action = torch.tanh(u)[0].numpy().astype(np.float32)
                value = float(old_critic(xt).item())
                obs, reward, term, trunc, last_info = env.step(action)
                ep_x.append(x)
                ep_rewards.append(float(reward))
                ep_old_v.append(value)
                episode_terminated, episode_truncated = bool(term), bool(trunc)
                if term or trunc:
                    break

            time_limit = bool(not episode_terminated and (episode_truncated or t == 799))
            # Return-to-horizon labels are known but CENSORED when an episode
            # ends at the artificial horizon. Exclude censored labels from loss.
            ep_ret = np.empty(len(ep_rewards), dtype=np.float32)
            future = 0.0
            for idx in range(len(ep_rewards) - 1, -1, -1):
                future = ep_rewards[idx] + GAMMA * future
                ep_ret[idx] = future
            assert len(ep_x) == len(ep_ret) and np.isfinite(ep_ret).all()
            all_x.append(np.stack(ep_x))
            all_ret.append(ep_ret)
            all_old_v.append(np.asarray(ep_old_v, np.float32))
            all_ep_id.append(np.full(len(ep_x), scene_seed, np.int32))
            all_valid.append(np.full(len(ep_x), not time_limit, np.bool_))
            offsets.append(offsets[-1] + len(ep_x))
            episodes.append({
                "seed": scene_seed, "steps": len(ep_x),
                "arrive_dest": bool(last_info.get("arrive_dest", False)),
                "collision": bool(last_info.get("crash_vehicle", False) or last_info.get("crash_object", False)),
                "out_of_road": bool(last_info.get("out_of_road", False)),
                "censored_time_limit": time_limit,
                "undiscounted_episode_reward": float(sum(ep_rewards)),
                "first_state_discounted_return_to_horizon": float(ep_ret[0]),
            })
            print(f"{args.variant}/{args.split}: seed={scene_seed} steps={len(ep_x)} "
                  f"success={episodes[-1]['arrive_dest']} censored={time_limit}", flush=True)
    env.close()

    xs = np.concatenate(all_x).astype(np.float32)
    targets = np.concatenate(all_ret).astype(np.float32)
    originals = np.concatenate(all_old_v).astype(np.float32)
    ep_ids = np.concatenate(all_ep_id)
    valid = np.concatenate(all_valid)
    assert xs.shape == (len(targets), 259)
    assert len(originals) == len(ep_ids) == len(targets) == len(valid)
    assert len(offsets) == len(seeds) + 1 and offsets[-1] == len(targets)
    assert len(set(ep_ids.tolist())) == len(seeds)
    assert np.isfinite(xs).all() and np.isfinite(originals).all()
    assert valid.sum() >= 1000 and sum(not e["censored_time_limit"] for e in episodes) >= 8, (
        "Too few fully terminated episodes for supervised value learning"
    )

    np.savez_compressed(
        target_npz,
        normalized_observations=xs,
        realized_discounted_return_to_horizon=targets,
        original_d2f_value=originals,
        scene_seed=ep_ids,
        episode_offsets=np.asarray(offsets, dtype=np.int64),
        uncensored_label_mask=valid,
    )
    digest = hashlib.sha256(target_npz.read_bytes()).hexdigest()
    record = {
        "stage": "D2h frozen original BC actor low-sigma full-episode value dataset",
        "split": args.split, "variant": args.variant,
        "scene_seeds": list(seeds), "actor_BC_seed": 42,
        "sigma_pre_tanh": SIGMA, "gamma": GAMMA,
        "density": 0.1, "map": 3, "horizon": 800,
        "total_steps": int(len(targets)),
        "uncensored_supervised_steps": int(valid.sum()),
        "censored_steps": int((~valid).sum()),
        "arrivals": sum(e["arrive_dest"] for e in episodes),
        "timeouts": sum(e["censored_time_limit"] for e in episodes),
        "original_critic": str(baseline_path),
        "output": str(target_npz), "sha256": digest,
        "wall_seconds": time.monotonic() - started,
        "episodes": episodes,
        "note": "Complete trajectories used. For horizon-censored episodes the finite-horizon returns are saved but EXCLUDED from primary supervised training and validation.",
    }
    target_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"FINISHED {args.variant} {args.split}: {len(targets)} steps; "
          f"{int(valid.sum())} fully observed target steps; {record['arrivals']} successes; "
          f"{record['timeouts']} horizon-censored", flush=True)


if __name__ == "__main__":
    main()
