"""Vectorized state/clock adapter for existing DLS A/B/C recurrent controllers.

P0 actors expose step(obs, (hidden, slow), scalar_t), where only scalar_t % 5
matters. A vectorized RL collector requires separate clocks per environment,
because episodes terminate and reset asynchronously.

This adapter groups environments by their independent slow-clock phase and
calls the ORIGINAL actor.step, WITHOUT modifying the model or public API.

It provides differentiable state and action tensor updates for batched TBPTT,
and a flattened state representation for future recurrent-buffer integration.
It is not by itself a complete SB3-Contrib policy or PPO trainer.
"""

from typing import NamedTuple

import torch

from dls_p0 import ACTION_DIM, OBS_DIM, SLOW_DIM, SLOW_PERIOD


class DLSVectorState(NamedTuple):
    hidden: torch.Tensor
    slow: torch.Tensor
    phase: torch.Tensor  # int64 [N], phase of NEXT action (mod SLOW_PERIOD)


class DLSVectorAdapter:
    def __init__(self, actor):
        self.actor = actor

    def initial_state(self, n_envs, device):
        hidden, slow = self.actor.initial_state(n_envs, device)
        assert slow.shape == (n_envs, SLOW_DIM)
        return DLSVectorState(
            hidden=hidden, slow=slow,
            phase=torch.zeros(n_envs, dtype=torch.int64, device=device),
        )

    def step(self, obs, state, episode_starts):
        """One vectorized control tick. episode_starts applies BEFORE this tick."""
        assert obs.ndim == 2 and obs.shape[-1] == OBS_DIM
        n_envs = obs.shape[0]
        assert state.hidden.shape[0] == n_envs
        assert state.slow.shape == (n_envs, SLOW_DIM)
        assert state.phase.shape == (n_envs,) and state.phase.dtype == torch.int64
        assert obs.device == state.hidden.device == state.slow.device == state.phase.device
        assert episode_starts.shape == (n_envs,) and episode_starts.dtype == torch.bool
        assert torch.all((state.phase >= 0) & (state.phase < SLOW_PERIOD))

        keep = ~episode_starts
        keep_h = keep.reshape(n_envs, *([1] * (state.hidden.ndim - 1)))
        initial_hidden = torch.where(keep_h, state.hidden, torch.zeros_like(state.hidden))
        initial_slow = torch.where(keep.unsqueeze(-1), state.slow, torch.zeros_like(state.slow))
        phase = torch.where(keep, state.phase, torch.zeros_like(state.phase))

        actions = obs.new_zeros((n_envs, ACTION_DIM))
        next_hidden = torch.zeros_like(initial_hidden)
        next_slow = torch.zeros_like(initial_slow)

        for t_mod in range(SLOW_PERIOD):
            selected = torch.nonzero(phase == t_mod, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            a, (h, z) = self.actor.step(
                obs.index_select(0, selected),
                (initial_hidden.index_select(0, selected),
                 initial_slow.index_select(0, selected)),
                int(t_mod),
            )
            assert a.shape == (len(selected), ACTION_DIM)
            assert h.shape == initial_hidden.index_select(0, selected).shape
            assert z.shape == (len(selected), SLOW_DIM)
            actions = actions.index_copy(0, selected, a)
            next_hidden = next_hidden.index_copy(0, selected, h)
            next_slow = next_slow.index_copy(0, selected, z)

        return actions, DLSVectorState(next_hidden, next_slow, (phase + 1) % SLOW_PERIOD)

    def flat_state_size(self):
        h, z = self.actor.initial_state(1, torch.device("cpu"))
        assert z.shape == (1, SLOW_DIM)
        return h[0].numel() + SLOW_DIM + 1

    def flatten(self, state):
        n_envs = state.hidden.shape[0]
        return torch.cat(
            [
                state.hidden.reshape(n_envs, -1),
                state.slow,
                state.phase.to(state.slow.dtype).unsqueeze(-1),
            ],
            dim=1,
        )

    def unflatten(self, flat):
        assert flat.ndim == 2 and flat.shape[1] == self.flat_state_size()
        batch = len(flat)
        init_h, init_z = self.actor.initial_state(batch, flat.device)
        h_elements = init_h[0].numel()
        hidden = flat[:, :h_elements].reshape_as(init_h)
        slow = flat[:, h_elements:h_elements + SLOW_DIM]
        assert slow.shape == init_z.shape
        clock = flat[:, -1]
        assert torch.all(clock == clock.round()), "Clock phase must be exactly integral"
        phase = clock.to(torch.int64)
        assert torch.all((phase >= 0) & (phase < SLOW_PERIOD))
        return DLSVectorState(hidden, slow, phase)
