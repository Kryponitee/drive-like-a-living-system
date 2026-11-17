"""SB3-Contrib 2.6.0 native RecurrentPPO bridge for original DLS A/B/C actors.

This is a custom policy class using the library's OFFICIAL RecurrentPPO
collector, recurrent rollout buffer, GAE, clipped PPO objective and optimizer.
A zero-parameter state-shape carrier adapts the library's LSTM buffer format to
our GRU/module, slow state and per-environment five-step clock.

The A/B/C actor itself is not replaced and loads its original frozen D1 BC
initialization. This is NOT an altered SB3 PPO implementation.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from dls_p0 import ACTION_DIM, OBS_DIM, model_info, nparams
from dls_recurrent_ppo import IdenticalCritic
from dls_vector_state_adapter import DLSVectorAdapter


class ShapeOnlyLSTMCarrier(nn.Module):
    """LSTM-like interface required by SB3 recurrent buffer; NEVER executed."""
    def __init__(self, hidden_size):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = int(hidden_size)

    def forward(self, *args, **kwargs):
        raise AssertionError("State carrier is not a real LSTM; DLS policy methods must be used")


class DLSBCInitializedRecurrentPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 variant, bc_checkpoint_path, **kwargs):
        assert variant in ("A", "B", "C")
        assert isinstance(observation_space, spaces.Box) and observation_space.shape == (OBS_DIM,)
        assert isinstance(action_space, spaces.Box) and action_space.shape == (ACTION_DIM,)
        assert np.allclose(action_space.low, -1.0) and np.allclose(action_space.high, 1.0)
        assert kwargs.get("use_sde", False) is False, "State-dependent exploration is not yet validated"

        actor, _ = model_info(variant)
        dimension = DLSVectorAdapter(actor).flat_state_size()
        assert dimension > 16
        kwargs["lstm_hidden_size"] = dimension
        kwargs["n_lstm_layers"] = 1
        kwargs["shared_lstm"] = False
        kwargs["enable_critic_lstm"] = False
        kwargs["net_arch"] = dict(pi=[], vf=[])
        kwargs["ortho_init"] = False
        kwargs["log_std_init"] = math.log(0.05)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Freeze all legacy SB3 policy modules, which are not used. The official
        # algorithm/buffer still receives the same API and recurrence shape.
        for param in self.parameters():
            param.requires_grad_(False)
        self.lstm_actor = ShapeOnlyLSTMCarrier(dimension)
        self.lstm_critic = None
        self.lstm_hidden_state_shape = (1, 1, dimension)

        checkpoint = torch.load(
            Path(bc_checkpoint_path), map_location="cpu", weights_only=False
        )
        assert checkpoint["variant"] == variant
        actor.load_state_dict(checkpoint["state_dict"], strict=True)
        assert nparams(actor) == checkpoint["num_params"]
        self.actor = actor
        self.vector_adapter = DLSVectorAdapter(self.actor)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2048)
            self.critic = IdenticalCritic()
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,),math.log(0.05)))
        self.register_buffer("dls_mean",torch.as_tensor(
            np.asarray(checkpoint["train_mean"],dtype=np.float32)))
        self.register_buffer("dls_std",torch.as_tensor(
            np.asarray(checkpoint["train_std"],dtype=np.float32)))
        assert self.dls_mean.shape==self.dls_std.shape==(OBS_DIM,)
        assert torch.all(self.dls_std>0)

        self.variant = variant
        self.bc_checkpoint_path = str(bc_checkpoint_path)
        self.state_carrier_size = dimension
        self.optimizer = self.optimizer_class(
            [*self.actor.parameters(), *self.critic.parameters(), self.log_std],
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        assert nparams(self.actor) == checkpoint["num_params"]
        assert not any(p.requires_grad for p in self.lstm_actor.parameters())

    def _normalized(self, obs):
        assert obs.ndim == 2 and obs.shape[-1] == OBS_DIM
        return ((obs - self.dls_mean) / self.dls_std).clamp(-10.0,10.0)

    def _batched_actor(self, obs, pi_states, episode_starts):
        """Follow SB3-Contrib's recurrent buffer layout (n_seq, max_seq_len)."""
        assert len(pi_states) == 2
        packed, unused_cell = pi_states
        assert packed.ndim == 3 and packed.shape[0] == 1
        assert packed.shape[2] == self.state_carrier_size
        n_seq = packed.shape[1]
        assert n_seq >= 1 and obs.shape[0] % n_seq == 0
        assert episode_starts.shape == (obs.shape[0],)
        length = obs.shape[0] // n_seq
        raw = self._normalized(obs).reshape(n_seq,length,OBS_DIM)
        starts = episode_starts.reshape(n_seq,length)
        state = self.vector_adapter.unflatten(packed.squeeze(0))
        means = []
        for j in range(length):
            mean_t, state = self.vector_adapter.step(raw[:,j,:], state, starts[:,j].bool())
            means.append(mean_t)
        actions_mean = torch.stack(means,dim=1).reshape(-1,ACTION_DIM)
        assert torch.isfinite(actions_mean).all()
        assert bool(torch.all(actions_mean.abs()<0.999)), "BC actor saturating tanh"
        next_packed = self.vector_adapter.flatten(state).unsqueeze(0)
        next_cell = torch.zeros_like(next_packed)
        return torch.atanh(actions_mean), (next_packed,next_cell)

    def _distribution(self, locations):
        return SquashedDiagGaussianDistribution(ACTION_DIM).proba_distribution(
            locations,self.log_std
        )

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        assert isinstance(lstm_states,RNNStates)
        locations, next_pi = self._batched_actor(obs,lstm_states.pi,episode_starts)
        dist = self._distribution(locations)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions,dist.gaussian_actions)
        values = self.critic(self._normalized(obs)).unsqueeze(-1)
        # The critic is feed-forward: its LSTM-shaped state is a fixed dummy.
        return actions, values, log_prob, RNNStates(next_pi, lstm_states.vf)

    def get_distribution(self, obs, lstm_states, episode_starts):
        locations, next_pi = self._batched_actor(obs,lstm_states,episode_starts)
        return self._distribution(locations), next_pi

    def predict_values(self, obs, lstm_states, episode_starts):
        return self.critic(self._normalized(obs)).unsqueeze(-1)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        locations,_ = self._batched_actor(obs,lstm_states.pi,episode_starts)
        dist = self._distribution(locations)
        values = self.critic(self._normalized(obs)).unsqueeze(-1)
        log_prob = dist.log_prob(actions)
        return values,log_prob,dist.entropy()
