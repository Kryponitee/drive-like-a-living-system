#!/usr/bin/env python3
"""D2b: paired deterministic BC-before vs PPO-after closed-loop evaluation, new scenes."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert
from dls_p0 import model_info, nparams


ROOT = Path("/mnt/datasets/lxb/dls")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("A", "B", "C", "expert"), required=True)
    p.add_argument("--phase", choices=("bc", "ppo", "expert"), required=True)
    p.add_argument("--density", type=float, choices=(0.0, 0.1, 0.25), required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    assert (args.phase == "expert") == (args.variant == "expert")
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    actor = None
    mean = std = None
    if args.phase != "expert":
        if args.phase == "bc":
            path = ROOT / "checkpoints" / "d1-v1" / f"{args.variant}-seed{args.seed}.pt"
            state_key = "state_dict"
        else:
            path = ROOT / "checkpoints" / "d2b-v1" / f"{args.variant}-seed{args.seed}.pt"
            state_key = "actor_state_dict"
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        assert ckpt["variant"] == args.variant
        actor, _ = model_info(args.variant)
        actor.load_state_dict(ckpt[state_key], strict=True)
        expected_params = (ckpt["num_params"] if args.phase == "bc"
                           else ckpt["num_actor_params"])
        assert nparams(actor) == expected_params
        actor.eval()
        mean = np.asarray(ckpt["train_mean"], dtype=np.float32)
        std = np.asarray(ckpt["train_std"], dtype=np.float32)
        assert mean.shape == std.shape == (259,) and std.min() > 0

    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": args.density,
        "map": 3,
        "start_seed": 3300,
        "num_scenarios": 12,
        "horizon": 800,
        "log_level": 50,
    })
    episodes = []
    started = time.monotonic()
    with torch.no_grad():
        for scene_seed in range(3300, 3312):
            obs, _ = env.reset(seed=scene_seed)
            memory = actor.initial_state(1, torch.device("cpu")) if actor else None
            episode_reward = 0.0
            for t in range(800):
                if actor is None:
                    action = np.asarray(expert(env.agent, deterministic=True), dtype=np.float32)
                else:
                    x = np.clip((np.asarray(obs, dtype=np.float32) - mean) / std, -10.0, 10.0)
                    assert x.shape == (259,) and np.isfinite(x).all()
                    action_t, memory = actor.step(torch.from_numpy(x).unsqueeze(0), memory, t)
                    action = action_t[0].numpy().astype(np.float32)
                assert action.shape == (2,) and np.isfinite(action).all()
                obs, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            success = bool(info.get("arrive_dest", False))
            collision = bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
            offroad = bool(info.get("out_of_road", False))
            episodes.append({
                "seed": scene_seed,
                "steps": t + 1,
                "success": success,
                "collision": collision,
                "out_of_road": offroad,
                "timeout": not (success or collision or offroad),
                "route_completion": float(info.get("route_completion", 0.0)),
                "reward": episode_reward,
            })
    env.close()

    result = {
        "stage": "D2b training sanity closed-loop evaluation, not causal coordination",
        "variant": args.variant,
        "phase": args.phase,
        "train_seed": args.seed if actor is not None else None,
        "traffic_density": args.density,
        "seeds": list(range(3300, 3312)),
        "horizon": 800,
        "success": sum(x["success"] for x in episodes),
        "collision": sum(x["collision"] for x in episodes),
        "offroad": sum(x["out_of_road"] for x in episodes),
        "timeout": sum(x["timeout"] for x in episodes),
        "mean_completion": float(np.mean([x["route_completion"] for x in episodes])),
        "mean_reward": float(np.mean([x["reward"] for x in episodes])),
        "duration_seconds": time.monotonic() - started,
        "episodes": episodes,
    }
    out = ROOT / "reports" / "d2b" / f"eval-{args.phase}-{args.variant}-density{args.density:.2f}.json"
    assert not out.exists(), "Output exists; refuse to overwrite independent evaluation"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{args.phase}/{args.variant} density={args.density}: "
          f"success={result['success']}/12 collisions={result['collision']} "
          f"offroad={result['offroad']} timeout={result['timeout']} "
          f"completion={result['mean_completion']:.3f}", flush=True)


if __name__ == "__main__":
    main()
