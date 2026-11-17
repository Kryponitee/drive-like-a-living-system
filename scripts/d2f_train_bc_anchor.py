#!/usr/bin/env python3
"""D2d: fixed-budget recurrent PPO pilot with exact on-policy sequence replay.

One MetaDrive simulator per process. Recompute recurrent hidden state from observed
episode history after policy updates; no reuse of a stale pre-update state.
D2d is a small training sanity test, not a full scientific architecture comparison.
"""

import argparse
import copy
import math
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from metadrive.envs.metadrive_env import MetaDriveEnv

from dls_p0 import model_info, nparams, run_sequence
from dls_bc_anchor import FrozenBCAnchorBatches
from dls_recurrent_ppo import (
    RecurrentSquashedPolicy,
    IdenticalCritic,
    squashed_gaussian_log_probability,
    gae_with_correct_terminal_handling,
)


ROOT = Path("/mnt/datasets/lxb/dls")
ANCHOR_LAMBDA = 10.0
ANCHOR_BATCH_SIZE = 16


def normalize(obs, mean, std, device):
    x = np.asarray(obs, dtype=np.float32)
    assert x.shape == (259,) and np.isfinite(x).all()
    normed = np.clip((x - mean) / std, -10.0, 10.0)
    return torch.from_numpy(normed.copy()).unsqueeze(0).to(device)


def detach_copy(state):
    return tuple(x.detach().clone() for x in state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--env-steps", type=int, default=4096)
    parser.add_argument("--rollout", type=int, default=128)
    args = parser.parse_args()
    assert args.env_steps % args.rollout == 0
    assert args.rollout % 32 == 0 and args.env_steps >= args.rollout
    assert os.environ["DLS_D2_TRAINING"] == "1"
    assert os.path.ismount("/mnt/datasets")
    assert torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(args.model_seed + 10000)
    np.random.seed(args.model_seed + 10000)
    random.seed(args.model_seed + 10000)

    source = ROOT / "checkpoints" / "d1-v1" / f"{args.variant}-seed{args.model_seed}.pt"
    ckpt = torch.load(source, weights_only=False, map_location="cpu")
    actor, _ = model_info(args.variant)
    assert ckpt["variant"] == args.variant and ckpt["num_params"] == nparams(actor)
    actor.load_state_dict(ckpt["state_dict"], strict=True)
    mean = np.asarray(ckpt["train_mean"], dtype=np.float32)
    std = np.asarray(ckpt["train_std"], dtype=np.float32)
    assert mean.shape == std.shape == (259,) and float(std.min()) > 0
    policy = RecurrentSquashedPolicy(actor, initial_log_std=math.log(0.05)).to(device)
    assert torch.allclose(policy.log_std.exp().detach(), torch.full((2,), 0.05, device=device))
    frozen_bc = copy.deepcopy(policy.actor).eval()
    frozen_bc.requires_grad_(False)
    anchor_batches = FrozenBCAnchorBatches(ROOT, mean, std, seed=1957, length=40)
    assert anchor_batches.steps == 91717
    value_net = IdenticalCritic().to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(policy.parameters()), "lr": 5e-5},
            {"params": list(value_net.parameters()), "lr": 2e-4},
        ],
        weight_decay=0.0,
    )

    out_root = ROOT / "reports" / "d2f"
    ckpt_root = ROOT / "checkpoints" / "d2f-v1"
    tb = ROOT / "tensorboard" / "d2f-v1" / f"{args.variant}-seed{args.model_seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_root / f"{args.variant}-seed{args.model_seed}.pt"
    report_path = out_root / f"train-{args.variant}-seed{args.model_seed}.json"
    assert not checkpoint_path.exists() and not report_path.exists()
    writer = SummaryWriter(log_dir=str(tb))

    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": 0.1,
        "map": 3,
        "start_seed": 4000,
        "num_scenarios": 500,
        "horizon": 800,
        "log_level": 50,
    })
    episode_seed = 4000
    obs, _ = env.reset(seed=episode_seed)
    state = policy.initial_state(1, device)
    timestep = 0
    episode_history = []  # Seen observations in this episode; used only to recompute current h.
    cumulative_episode_reward = 0.0
    completed_episodes = []
    updates = []
    started = time.monotonic()

    for batch_idx in range(args.env_steps // args.rollout):
        # Sample exactly args.rollout steps with the same behavior-policy weights.
        # Segments are at most 32 steps; they never cross a real episode terminal.
        x_all, latent_all, old_logps, values, rewards, dones = [], [], [], [], [], []
        segments = []
        segment = None
        with torch.no_grad():
            for j in range(args.rollout):
                if segment is None:
                    segment = {
                        "start": j, "end": None,
                        "start_state": detach_copy(state),
                        "start_t": timestep,
                    }
                x = normalize(obs, mean, std, device)
                action, latent, logp, state = policy.sample(x, state, timestep)
                assert action.shape == (1, 2) and torch.isfinite(action).all()
                nxt, reward, terminated, truncated, last_info = env.step(
                    action[0].detach().cpu().numpy().astype(np.float32)
                )
                done = bool(terminated or truncated)
                x_all.append(x[0].detach())
                latent_all.append(latent[0].detach())
                old_logps.append(logp[0].detach())
                values.append(value_net(x)[0].detach())
                rewards.append(float(reward))
                dones.append(done)
                episode_history.append(x.detach())
                timestep += 1
                cumulative_episode_reward += float(reward)
                obs = nxt

                if done:
                    completed_episodes.append({
                        "scene_seed": episode_seed,
                        "steps": timestep,
                        "success": bool(last_info.get("arrive_dest", False)),
                        "collision": bool(last_info.get("crash_vehicle", False) or
                                          last_info.get("crash_object", False)),
                        "out_of_road": bool(last_info.get("out_of_road", False)),
                        "reward": cumulative_episode_reward,
                    })
                    episode_seed += 1
                    obs, _ = env.reset(seed=episode_seed)
                    state = policy.initial_state(1, device)
                    timestep = 0
                    episode_history = []
                    cumulative_episode_reward = 0.0

                if done or (j - segment["start"] + 1 >= 32):
                    segment["end"] = j + 1
                    segments.append(segment)
                    segment = None
            if segment is not None:
                segment["end"] = args.rollout
                segments.append(segment)

            bootstrap = value_net(normalize(obs, mean, std, device))[0].detach()
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            values_tensor = torch.stack(values)
            done_tensor = torch.tensor(dones, dtype=torch.bool, device=device)
            advantage, returns = gae_with_correct_terminal_handling(
                rewards_tensor, values_tensor, done_tensor, bootstrap
            )

        xs = torch.stack(x_all)
        latents = torch.stack(latent_all)
        lp_old = torch.stack(old_logps)
        assert len(xs) == args.rollout
        assert sum(s["end"] - s["start"] for s in segments) == args.rollout
        assert all(1 <= s["end"] - s["start"] <= 32 for s in segments)

        # Full rollout probability must be reproducible at ratio=1 before update.
        def replay(segment):
            state_for_grad = detach_copy(segment["start_state"])
            lp_list = []
            for i in range(segment["start"], segment["end"]):
                loc, state_for_grad = policy.step_parameters(
                    xs[i].unsqueeze(0),
                    state_for_grad,
                    segment["start_t"] + (i - segment["start"]),
                )
                lp_list.append(
                    squashed_gaussian_log_probability(
                        loc, policy.log_std, latents[i].unsqueeze(0)
                    )[0]
                )
            return torch.stack(lp_list)

        with torch.no_grad():
            deviations = []
            for s in segments:
                new_lp = replay(s)
                deviations.append((torch.exp(new_lp - lp_old[s["start"]:s["end"]]) - 1).abs().max())
            max_replay_deviation = float(torch.stack(deviations).max().item())
        assert max_replay_deviation < 3e-5, max_replay_deviation

        # D2f 唯一机制变化：从原有 D1 成功示范的状态分布抽样，约束策略相对冻结 BC 的动作漂移。
        anchor_x, anchor_mask = anchor_batches.sample(ANCHOR_BATCH_SIZE, device)
        with torch.no_grad():
            anchor_target = run_sequence(frozen_bc, anchor_x).detach()
        assert anchor_target.shape == (ANCHOR_BATCH_SIZE, 40, 2)
        adv_norm = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
        final_loss = 0.0
        final_kl = 0.0
        final_anchor_mse = 0.0
        for epoch in range(2):
            recomputed_lp = torch.cat([replay(s) for s in segments])
            ratio = (recomputed_lp - lp_old).exp()
            surrogate1 = ratio * adv_norm
            surrogate2 = ratio.clamp(0.8, 1.2) * adv_norm
            loss_actor = -torch.minimum(surrogate1, surrogate2).mean()
            loss_value = F.mse_loss(value_net(xs), returns)
            anchor_action = run_sequence(policy.actor, anchor_x)
            anchor_error = (anchor_action - anchor_target).square()
            anchor_mse = (anchor_error * anchor_mask.unsqueeze(-1)).sum() / (2 * anchor_mask.sum())
            loss = loss_actor + 0.5 * loss_value + ANCHOR_LAMBDA * anchor_mse
            assert torch.isfinite(loss) and torch.isfinite(anchor_mse)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_gnorm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
            critic_gnorm = float(torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0))
            assert np.isfinite(actor_gnorm) and np.isfinite(critic_gnorm)
            optimizer.step()
            with torch.no_grad():
                policy.log_std.clamp_(-4.0, -0.5)
            final_loss = float(loss.detach().item())
            final_anchor_mse = float(anchor_mse.detach().item())
            final_kl = float((lp_old - recomputed_lp.detach()).mean().item())

        # Policy has changed. Recompute h from all actually observed states
        # in the CURRENT partial episode. No old-policy hidden state is retained.
        with torch.no_grad():
            state = policy.initial_state(1, device)
            for t_history, x_history in enumerate(episode_history):
                _, state = policy.step_parameters(x_history, state, t_history)
            assert len(episode_history) == timestep

        step_count = (batch_idx + 1) * args.rollout
        record = {
            "steps": step_count,
            "segments": len(segments),
            "max_preupdate_ratio_error": max_replay_deviation,
            "actor_grad_norm": actor_gnorm,
            "critic_grad_norm": critic_gnorm,
            "loss": final_loss,
            "anchor_mse_last_epoch": final_anchor_mse,
            "approx_kl_last_epoch": final_kl,
            "terminal_episodes_seen": len(completed_episodes),
            "logstd": policy.log_std.detach().cpu().tolist(),
        }
        updates.append(record)
        writer.add_scalar("ppo/loss", record["loss"], step_count)
        writer.add_scalar("ppo/kl", record["approx_kl_last_epoch"], step_count)
        writer.add_scalar("anchor/bc_action_mse", record["anchor_mse_last_epoch"], step_count)
        writer.add_scalar("ppo/replay_ratio_error", record["max_preupdate_ratio_error"], step_count)
        writer.add_scalar("episodes/completed", len(completed_episodes), step_count)
        if batch_idx % 4 == 0 or batch_idx == args.env_steps // args.rollout - 1:
            print(f"{args.variant}: steps={step_count}/{args.env_steps}, "
                  f"episodes={len(completed_episodes)}, "
                  f"ratio_err={max_replay_deviation:.2e}, kl={final_kl:.4g}, "
                  f"anchor_mse={final_anchor_mse:.6f}, logstd={record['logstd']}", flush=True)
    env.close()
    writer.close()

    torch.save({
        "stage": "D2f BC-function-anchored PPO pilot",
        "variant": args.variant,
        "model_seed": args.model_seed,
        "actor_state_dict": policy.actor.state_dict(),
        "log_std": policy.log_std.detach().cpu(),
        "critic_state_dict": value_net.state_dict(),
        "train_mean": mean,
        "train_std": std,
        "training_environment_steps": args.env_steps,
        "train_density": 0.1,
        "initial_bc_checkpoint": str(source),
        "initial_data_id": "d1-v1",
        "num_actor_params": nparams(policy.actor),
        "num_policy_params": nparams(policy),
        "recurrent_segment_length": 32,
        "rollout_length": args.rollout,
        "ppo_epochs_per_rollout": 2,
        "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.2,
        "same_critic_for_A_B_C": True,
        "initial_exploration_sigma": 0.05,
        "initial_exploration_log_std": math.log(0.05),
        "log_std_clamp": [-4.0, -0.5],
        "anchor_coefficient": ANCHOR_LAMBDA,
        "anchor_batch_sequences": ANCHOR_BATCH_SIZE,
        "anchor_sequence_length": 40,
        "anchor_source": "D1 train success-only 91717 steps (unmodified)",
        "anchor_reference": "frozen D1 BC model mean action (zero-initialized RNN 40-step chunks)",
    }, checkpoint_path)

    report = {
        "stage": "D2f BC functional anchor (one added factor), not evidence for communication",
        "only_changed_hyperparameter": "initial_log_std exp(-1.3) to log(0.05); clamp extended solely to permit sigma=0.05",
        "initial_exploration_sigma": 0.05,
        "only_added_mechanism": "BC frozen reference action-MSE anchor, lambda 10, 16x40 D1 training sequences per PPO update",
        "anchor_lambda": ANCHOR_LAMBDA,
        "anchor_sequences_per_update": ANCHOR_BATCH_SIZE,
        "anchor_steps_available": anchor_batches.steps,
        "mean_anchor_mse_last_epoch": float(np.mean([u["anchor_mse_last_epoch"] for u in updates])),
        "variant": args.variant,
        "seed": args.model_seed,
        "gpu": args.gpu,
        "environment_steps": args.env_steps,
        "rollout_length": args.rollout,
        "updates": len(updates),
        "training_seed_first": 4000,
        "training_seed_last": episode_seed,
        "training_traffic_density": 0.1,
        "wall_seconds": time.monotonic() - started,
        "actor_param_count": nparams(policy.actor),
        "policy_param_count": nparams(policy),
        "critic_param_count": nparams(value_net),
        "max_ppo_behavior_replay_error": max(u["max_preupdate_ratio_error"] for u in updates),
        "mean_approx_kl_last_epoch": float(np.mean([u["approx_kl_last_epoch"] for u in updates])),
        "completed_episodes": completed_episodes,
        "num_successful_completed_training_episodes": sum(e["success"] for e in completed_episodes),
        "checkpoint": str(checkpoint_path),
        "update_history": updates,
        "note": "All recurrent hidden states are rebuilt from observed episode history after every update; exact environment transition counts matched. Same model, BC initialization and rollout budget; no claims of cross-seed effect.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("update_history", "completed_episodes")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
