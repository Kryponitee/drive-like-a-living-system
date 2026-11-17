#!/usr/bin/env python3
"""D1 A/B/C 的相同 seed 闭环预检，附同条件教师参考，绝不使用测试专家轨迹做训练。"""

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


DATA_ROOT = Path("/mnt/datasets/lxb/dls")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A", "B", "C", "expert"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-seed", type=int, default=3000)
    parser.add_argument("--scene-count", type=int, default=8)
    parser.add_argument("--density", type=float, choices=(0.0, 0.1, 0.25), default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    assert 3000 <= args.start_seed < 4000
    assert 1 <= args.scene_count <= 100
    assert args.start_seed + args.scene_count <= 4000
    assert all(ch.isalnum() or ch == "-" for ch in args.tag)
    if not args.tag:
        assert args.start_seed == 3000 and args.scene_count == 8 and args.density is None
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    model = None
    stats = None
    if args.variant != "expert":
        checkpoint = DATA_ROOT / "checkpoints" / "d1-v1" / f"{args.variant}-seed{args.seed}.pt"
        assert checkpoint.is_file()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert ckpt["variant"] == args.variant and ckpt["seed"] == args.seed
        assert ckpt["data_id"] == "d1-v1"
        model, arch = model_info(args.variant)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        assert ckpt["num_params"] == nparams(model)
        model.eval()
        stats = (np.asarray(ckpt["train_mean"], np.float32),
                 np.asarray(ckpt["train_std"], np.float32))
        assert stats[0].shape == (259,) and stats[1].shape == (259,)

    results = []
    started = time.monotonic()
    with torch.no_grad():
        for density in ((args.density,) if args.density is not None else (0.0, 0.1, 0.25)):
            env = MetaDriveEnv(
                {
                    "use_render": False,
                    "image_observation": False,
                    "traffic_density": density,
                    "map": 3,
                    "start_seed": args.start_seed,
                    "num_scenarios": args.scene_count,
                    "horizon": 800,
                    "log_level": 50,
                }
            )
            episodes = []
            for seed in range(args.start_seed, args.start_seed + args.scene_count):
                obs, _ = env.reset(seed=seed)
                state = model.initial_state(1, torch.device("cpu")) if model else None
                last_info = {}
                rewards = 0.0
                for t in range(800):
                    if model:
                        x = np.asarray(obs, np.float32)
                        assert x.shape == (259,)
                        x = np.clip((x - stats[0]) / stats[1], -10.0, 10.0)
                        action_t, state = model.step(torch.from_numpy(x).unsqueeze(0), state, t)
                        action = np.clip(action_t.squeeze(0).numpy(), -1.0, 1.0)
                    else:
                        action = np.clip(
                            np.asarray(expert(env.agent, deterministic=True), dtype=np.float32), -1, 1
                        )
                    obs, reward, terminated, truncated, last_info = env.step(action)
                    rewards += float(reward)
                    if terminated or truncated:
                        break
                arrived = bool(last_info.get("arrive_dest", False))
                collision = bool(last_info.get("crash_vehicle", False) or last_info.get("crash_object", False))
                out_of_road = bool(last_info.get("out_of_road", False))
                episodes.append({
                    "seed": seed, "steps": t+1, "arrive_dest": arrived,
                    "collision": collision, "out_of_road": out_of_road,
                    "timeout": not (arrived or collision or out_of_road),
                    "route_completion": float(last_info.get("route_completion", 0.0)),
                    "episode_reward": rewards,
                })
            env.close()
            group = {
                "density": density,
                "success": sum(x["arrive_dest"] for x in episodes),
                "collisions": sum(x["collision"] for x in episodes),
                "out_of_road": sum(x["out_of_road"] for x in episodes),
                "timeout": sum(x["timeout"] for x in episodes),
                "mean_route_completion": float(np.mean([x["route_completion"] for x in episodes])),
                "episodes": episodes,
            }
            results.append(group)
            print(f"{args.variant}: traffic={density}: "
                  f"success={group['success']}/{args.scene_count} collisions={group['collisions']}/{args.scene_count} "
                  f"offroad={group['out_of_road']}/{args.scene_count} timeout={group['timeout']}/{args.scene_count} "
                  f"completion={group['mean_route_completion']:.3f}", flush=True)

    report = {
        "stage": "D1 first CLOSED-LOOP pilot; do not infer motor synergy",
        "variant": args.variant,
        "seed": args.seed if model else None,
        "evaluated_seeds": list(range(args.start_seed, args.start_seed + args.scene_count)),
        "evaluation_tag": args.tag or "pilot",
        "densities": [args.density] if args.density is not None else [0.0, 0.1, 0.25],
        "horizon": 800,
        "map": 3,
        "teacher_action_sampling": "deterministic" if not model else None,
        "wall_seconds": time.monotonic() - started,
        "results": results,
    }
    filename = (
        f"closed-loop-{args.tag}-{args.variant}-seed{args.seed}-density{args.density:.2f}.json"
        if args.tag else f"closed-loop-{args.variant}-seed{args.seed}.json"
    )
    output = DATA_ROOT / "reports" / "d1" / filename
    assert not output.exists(), "Do not overwrite previous closed-loop pilot"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
