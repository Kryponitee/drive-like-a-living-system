"""DLS纵向双踏板控制器：独立Sigmoid油门/制动输出。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


OBS_DIM = 259
ACTION_DIM = 2
MODULES = 8
LOCAL_DIM = 33
HIDDEN = 49
MESSAGE_DIM = 8
PROJECTION_SEED = 20261117


def make_fixed_projection():
    """固定正交投影：259维观测扩展到8份33维局部输入。"""
    generator = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
    matrix = torch.randn(MODULES * LOCAL_DIM, OBS_DIM, generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    projection = q.T.contiguous()
    assert projection.shape == (OBS_DIM, MODULES * LOCAL_DIM)
    torch.testing.assert_close(
        projection @ projection.T,
        torch.eye(OBS_DIM),
        rtol=1e-4,
        atol=1e-4,
    )
    return projection


class MonolithicController(nn.Module):
    """参数量匹配的单体循环基线。"""

    def __init__(self, hidden=102):
        super().__init__()
        self.hidden = hidden
        self.rnn = nn.GRUCell(OBS_DIM, hidden)
        self.action_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.Tanh(),
            nn.Linear(32, ACTION_DIM),
        )

    def initial_state(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden, device=device)

    def step(self, observation, state, t):
        state = self.rnn(observation, state)
        action = torch.sigmoid(self.action_head(state))
        return action, state


class LocalMotorController(nn.Module):
    """8个局部循环模块；local只用自身消息，cross接收其他模块的消息。"""

    def __init__(self, arm):
        super().__init__()
        assert arm in ("local", "cross")
        self.arm = arm
        self.register_buffer("projection", make_fixed_projection(), persistent=True)
        self.cells = nn.ModuleList([
            nn.GRUCell(LOCAL_DIM + MESSAGE_DIM, HIDDEN)
            for _ in range(MODULES)
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, 1), nn.Tanh())
            for _ in range(MODULES)
        ])
        self.query_weights = nn.Parameter(torch.empty(MODULES, HIDDEN, MESSAGE_DIM))
        self.value_weights = nn.Parameter(torch.empty(MODULES, HIDDEN, MESSAGE_DIM))
        for i in range(MODULES):
            nn.init.xavier_uniform_(self.query_weights[i])
            nn.init.xavier_uniform_(self.value_weights[i])
        self.decoder = nn.Parameter(torch.empty(ACTION_DIM, MODULES))
        nn.init.xavier_uniform_(self.decoder)

    def initial_state(self, batch_size, device):
        return torch.zeros(batch_size, MODULES, HIDDEN, device=device)

    def step(
        self,
        observation,
        state,
        t,
        block_messages=False,
        return_diagnostics=False,
        unit_mask=None,
        blocked_sender=None,
    ):
        assert observation.ndim == 2 and observation.shape[-1] == OBS_DIM
        assert state.shape == (observation.shape[0], MODULES, HIDDEN)
        local = (observation @ self.projection).reshape(-1, MODULES, LOCAL_DIM)
        queries = torch.einsum("bkh,khm->bkm", state, self.query_weights)
        values = torch.einsum("bkh,khm->bkm", state, self.value_weights)
        gate = torch.sigmoid(
            (queries * values).sum(-1, keepdim=True) / math.sqrt(MESSAGE_DIM)
        )

        if self.arm == "local":
            assert blocked_sender is None
            source = values
        elif blocked_sender is not None:
            assert isinstance(blocked_sender, int) and 0 <= blocked_sender < MODULES
            sender_mask = torch.ones(MODULES, dtype=values.dtype, device=values.device)
            sender_mask[blocked_sender] = 0.0
            retained = values * sender_mask[None, :, None]
            source = (retained.sum(dim=1, keepdim=True) - retained) / (MODULES - 1)
        else:
            source = (values.sum(dim=1, keepdim=True) - values) / (MODULES - 1)

        message = gate * source
        if block_messages:
            assert self.arm == "cross"
            message = torch.zeros_like(message)

        updated = torch.stack(
            [
                self.cells[i](
                    torch.cat((local[:, i], message[:, i]), dim=-1),
                    state[:, i],
                )
                for i in range(MODULES)
            ],
            dim=1,
        )
        activities = torch.cat(
            [self.heads[i](updated[:, i]) for i in range(MODULES)],
            dim=-1,
        )
        if unit_mask is not None:
            mask = torch.as_tensor(
                unit_mask, device=activities.device, dtype=activities.dtype
            )
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            assert mask.ndim == 2 and mask.shape[1] == MODULES
            assert mask.shape[0] in (1, observation.shape[0])
            assert bool(((mask == 0) | (mask == 1)).all())
            activities_to_output = activities * mask
        else:
            activities_to_output = activities

        # 两个执行头独立归一化，不施加T+B=1或互斥约束。
        action = torch.sigmoid(F.linear(activities_to_output, self.decoder))
        if return_diagnostics:
            return action, updated, {
                "activities": activities,
                "mean_abs_messages": message.abs().mean(dim=(0, 2)),
                "mean_abs_local_inputs": local.abs().mean(dim=(0, 2)),
            }
        return action, updated


def model_info(variant):
    assert variant in ("A", "B", "C")
    if variant == "A":
        model = MonolithicController(hidden=102)
    elif variant == "B":
        model = LocalMotorController("local")
    else:
        model = LocalMotorController("cross")
    count = num_params(model)
    reference = 114880
    assert abs(count / reference - 1.0) < 0.01, (variant, count, reference)
    return model, {
        "variant": variant,
        "architecture": (
            {"type": "monolithic_gru", "hidden": 102}
            if variant == "A"
            else {"type": "8_local_gru_modules", "arm": "local" if variant == "B" else "cross",
                  "local_dim": LOCAL_DIM, "hidden": HIDDEN, "message_dim": MESSAGE_DIM}
        ),
        "num_params": count,
        "reference_params": reference,
        "action_interface": "[absolute_throttle, absolute_brake] in [0,1]^2",
    }


def num_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_sequence(model, observations, block_messages=False, unit_mask=None):
    assert observations.ndim == 3 and observations.shape[-1] == OBS_DIM
    batch_size, n_steps, _ = observations.shape
    state = model.initial_state(batch_size, observations.device)
    actions = []
    for t in range(n_steps):
        if isinstance(model, LocalMotorController):
            action, state = model.step(
                observations[:, t],
                state,
                t,
                block_messages=block_messages,
                unit_mask=unit_mask,
            )
        else:
            action, state = model.step(observations[:, t], state, t)
        actions.append(action)
    return torch.stack(actions, dim=1)
