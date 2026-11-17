"""D2a: recurrent PPO policy distribution, per-segment state replay and advantage algebra.

Only policy x_t is used as input. Physical future states and teacher internals are not used.
This module is intentionally small; D2a validates mechanics, not driving performance.
"""

import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from dls_p0 import OBS_DIM, ACTION_DIM


class RecurrentSquashedPolicy(nn.Module):
    def __init__(self, bc_actor, initial_log_std=-1.3):
        super().__init__()
        self.actor = bc_actor
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), initial_log_std))

    def initial_state(self, batch, device):
        return self.actor.initial_state(batch, device)

    def step_parameters(self, x, state, global_t):
        # BC actor originally returned tanh(preactivation). Inverse is mathematically
        # identical except at extreme saturation, where we fail fast on hard clipping.
        mean_action, next_state = self.actor.step(x, state, global_t)
        assert torch.isfinite(mean_action).all()
        assert torch.max(torch.abs(mean_action)).item() < 0.999, "BC tanh saturated; expose logits explicitly"
        location = torch.atanh(mean_action)
        return location, next_state

    def sample(self, x, state, global_t):
        location, state_next = self.step_parameters(x, state, global_t)
        latent = Normal(location, self.log_std.exp()).sample()
        action = torch.tanh(latent)
        log_probability = squashed_gaussian_log_probability(
            location, self.log_std, latent
        )
        return action, latent, log_probability, state_next


def squashed_gaussian_log_probability(location, log_std, latent):
    """Log density of a=tanh(u); u is retained to avoid unstable atanh(action)."""
    assert location.shape == latent.shape
    assert location.shape[-1] == ACTION_DIM
    # log(1-tanh(u)^2) = 2(log(2) - u - softplus(-2u)), stable for large |u|.
    log_jacobian = 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
    normal_lp = Normal(location, log_std.exp()).log_prob(latent)
    answer = (normal_lp - log_jacobian).sum(dim=-1)
    assert torch.isfinite(answer).all(), "Invalid squashed Gaussian log probability"
    return answer


class IdenticalCritic(nn.Module):
    """Same independent value-network capacity for all A/B/C variants."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def gae_with_correct_terminal_handling(rewards, values, dones, bootstrap_value,
                                       gamma=0.99, gae_lambda=0.95):
    """For a rollout segment: bootstrap at artificial BPTT boundary, not true done."""
    assert rewards.ndim == values.ndim == dones.ndim == 1
    assert rewards.shape == values.shape == dones.shape
    assert len(rewards) > 0
    advantages = torch.zeros_like(values)
    next_advantage = values.new_zeros(())
    next_value = bootstrap_value.detach()
    for i in reversed(range(len(rewards))):
        alive = 1.0 - dones[i].float()
        delta = rewards[i] + gamma * next_value * alive - values[i]
        next_advantage = delta + gamma * gae_lambda * alive * next_advantage
        advantages[i] = next_advantage
        next_value = values[i]
    return advantages, advantages + values
