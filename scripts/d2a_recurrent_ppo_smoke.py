#!/usr/bin/env python3
"""D2a: real MetaDrive rollout and one recurrent PPO update. This is only a correctness audit."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from metadrive.envs.metadrive_env import MetaDriveEnv
from dls_p0 import model_info, nparams
from dls_recurrent_ppo import (
    RecurrentSquashedPolicy, IdenticalCritic,
    squashed_gaussian_log_probability, gae_with_correct_terminal_handling,
)


ROOT = Path("/mnt/datasets/lxb/dls")


def normalized_obs(observation, mean, std):
    obs = np.asarray(observation, dtype=np.float32)
    assert obs.shape == (259,) and np.isfinite(obs).all()
    obs = np.clip((obs - mean) / std, -10.0, 10.0)
    return torch.from_numpy(obs.copy()).unsqueeze(0)


def detached_state(state):
    return tuple(item.detach().clone() for item in state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    np.random.seed(1234)

    ckpt = torch.load(
        ROOT / "checkpoints" / "d1-v1" / f"{args.variant}-seed42.pt",
        map_location="cpu", weights_only=False
    )
    actor, _ = model_info(args.variant)
    actor.load_state_dict(ckpt["state_dict"], strict=True)
    actor.eval()
    mean = np.asarray(ckpt["train_mean"], dtype=np.float32)
    std = np.asarray(ckpt["train_std"], dtype=np.float32)
    assert mean.shape == std.shape == (259,)
    assert np.min(std) > 0
    policy = RecurrentSquashedPolicy(actor)
    critic = IdenticalCritic()

    # Numerical edge-case test, including a squashed action close to 1.
    edge_u = torch.tensor([[0.0, 2.0], [9.0, -9.0]], dtype=torch.float32)
    edge_lp = squashed_gaussian_log_probability(
        torch.zeros_like(edge_u), torch.full((2,), -1.3), edge_u
    )
    assert edge_lp.shape == (2,) and torch.isfinite(edge_lp).all()

    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": 0.1,
        "map": 3,
        "start_seed": 3200,
        "num_scenarios": 4,
        "horizon": 800,
        "log_level": 50,
    })
    state = policy.initial_state(1, torch.device("cpu"))
    obs, _ = env.reset(seed=3200)
    global_t = 0
    restart = 0
    previous_done = False
    segments = []

    with torch.no_grad():
        for segment_idx in range(3):
            if previous_done:
                restart += 1
                obs, _ = env.reset(seed=3200 + restart)
                state = policy.initial_state(1, torch.device("cpu"))
                global_t = 0

            initial_state = detached_state(state)
            initial_time_index = global_t
            items = {"x": [], "latent": [], "logp_old": [], "value_old": [],
                     "reward": [], "done": []}

            for _ in range(16):
                x = normalized_obs(obs, mean, std)
                a, latent, old_logp, state = policy.sample(x, state, global_t)
                assert a.shape == (1, 2) and torch.isfinite(a).all()
                assert (a.abs() < 1).all()
                next_obs, reward, terminated, truncated, info = env.step(
                    a.squeeze(0).cpu().numpy().astype(np.float32)
                )
                done = bool(terminated or truncated)
                items["x"].append(x.squeeze(0).detach())
                items["latent"].append(latent.squeeze(0).detach())
                items["logp_old"].append(old_logp.squeeze(0).detach())
                items["value_old"].append(critic(x).squeeze(0).detach())
                items["reward"].append(float(reward))
                items["done"].append(done)
                global_t += 1
                obs = next_obs
                if done:
                    break
            previous_done = items["done"][-1]

            # Bootstrap only at time-limit of the mini-rollout, never after an environment terminal.
            bootstrap = (torch.tensor(0.0) if previous_done else
                         critic(normalized_obs(obs, mean, std)).squeeze(0).detach())
            segments.append({
                "state": initial_state, "initial_t": initial_time_index,
                "x": torch.stack(items["x"]),
                "latent": torch.stack(items["latent"]),
                "logp_old": torch.stack(items["logp_old"]),
                "values_old": torch.stack(items["value_old"]),
                "rewards": torch.tensor(items["reward"], dtype=torch.float32),
                "dones": torch.tensor(items["done"], dtype=torch.bool),
                "bootstrap": bootstrap,
            })
    env.close()

    # Exact same model, latent controls, initial recurrent state and time offset
    # must reproduce the behavior-policy density prior to optimizer.step().
    old_parts, new_parts, advantages, returns, critic_estimates = [], [], [], [], []
    for s in segments:
        h = detached_state(s["state"])
        logps = []
        for i in range(len(s["x"])):
            mu, h = policy.step_parameters(
                s["x"][i].unsqueeze(0), h, s["initial_t"] + i
            )
            new_logp = squashed_gaussian_log_probability(
                mu, policy.log_std, s["latent"][i].unsqueeze(0)
            )
            logps.append(new_logp.squeeze(0))
        old_parts.append(s["logp_old"])
        new_parts.append(torch.stack(logps))
        with torch.no_grad():
            adv, target = gae_with_correct_terminal_handling(
                s["rewards"], s["values_old"], s["dones"], s["bootstrap"]
            )
        advantages.append(adv)
        returns.append(target)
        critic_estimates.append(critic(s["x"]))

    old_lp = torch.cat(old_parts)
    new_lp = torch.cat(new_parts)
    ratio = torch.exp(new_lp - old_lp)
    max_ratio_error = float((ratio - 1.0).abs().max().detach().item())
    assert max_ratio_error < 2e-5, f"Old-policy ratio cannot be reproduced: {max_ratio_error}"

    advantage = torch.cat(advantages)
    targets = torch.cat(returns)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
    critic_pred = torch.cat(critic_estimates)
    clipped_ratio = ratio.clamp(0.8, 1.2)
    policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
    value_loss = F.mse_loss(critic_pred, targets)
    objective = policy_loss + 0.5 * value_loss
    assert torch.isfinite(objective)

    before = [p.detach().clone() for p in policy.parameters()]
    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(critic.parameters()),
        lr=1e-4
    )
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
    critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0))
    assert np.isfinite(actor_grad_norm) and np.isfinite(critic_grad_norm)
    assert actor_grad_norm > 0 and critic_grad_norm > 0
    optimizer.step()
    update_magnitude = sum(
        float((old - p).abs().sum().item())
        for old, p in zip(before, policy.parameters())
    )
    assert update_magnitude > 0, "PPO update did not change the policy"

    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "D2a PPO mathematical and recurrent correctness smoke, NOT full RL training",
        "variant": args.variant,
        "bc_training_seed": 42,
        "data_disk": str(ROOT),
        "metadrive": "0.4.3",
        "traffic_density": 0.1,
        "scene_seeds": [3200, 3201, 3202, 3203],
        "segments": [
            {"length": int(len(s["x"])),
             "time_offset": int(s["initial_t"]),
             "environment_terminal": bool(s["dones"][-1].item()),
             "initial_state_norm": float(sum(v.norm().item() for v in s["state"]))}
            for s in segments
        ],
        "total_rollout_steps": sum(len(s["x"]) for s in segments),
        "max_replayed_old_policy_ratio_error": max_ratio_error,
        "ppo_policy_loss_before_update": float(policy_loss.detach()),
        "ppo_value_loss_before_update": float(value_loss.detach()),
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
        "policy_parameter_abs_change_after_one_optimizer_step": update_magnitude,
        "policy_trainable_params_including_logstd": nparams(policy),
        "value_trainable_params": nparams(critic),
        "tanh_log_prob_stress_finite": True,
        "read_only_bc_checkpoint": True,
        "actual_learning_performance": "not tested",
    }
    path = ROOT / "reports" / "d2a" / f"recurrent-ppo-smoke-{args.variant}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), "Audit report already exists; refusing silent overwrite"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
