"""D6：八个互补局部观测模块，等参数比较自身信息反馈与跨模块消息。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


OBS_DIM = 259
MOTORS = 8
LOCAL_DIM = 33
HIDDEN = 49
MESSAGE_DIM = 8
ACTION_DIM = 2
PROJECTION_SEED = 20261117


def make_fixed_projection():
    """P: 259×264，行正交；八份局部输入的联合线性信息无损。"""
    gen = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
    matrix = torch.randn(MOTORS * LOCAL_DIM, OBS_DIM, generator=gen)
    thin_q, _ = torch.linalg.qr(matrix, mode="reduced")
    projection = thin_q.T.contiguous()
    assert projection.shape == (OBS_DIM, MOTORS * LOCAL_DIM)
    torch.testing.assert_close(
        projection @ projection.T,
        torch.eye(OBS_DIM),
        rtol=1e-4, atol=1e-4,
    )
    return projection


class LocalMotorController(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ("local", "cross")
        self.arm = arm
        self.register_buffer("projection", make_fixed_projection(), persistent=True)
        self.cells = nn.ModuleList([
            nn.GRUCell(LOCAL_DIM + MESSAGE_DIM, HIDDEN) for _ in range(MOTORS)
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, 1), nn.Tanh())
            for _ in range(MOTORS)
        ])
        self.query_weights = nn.Parameter(torch.empty(MOTORS, HIDDEN, MESSAGE_DIM))
        self.value_weights = nn.Parameter(torch.empty(MOTORS, HIDDEN, MESSAGE_DIM))
        for i in range(MOTORS):
            nn.init.xavier_uniform_(self.query_weights[i])
            nn.init.xavier_uniform_(self.value_weights[i])
        self.decoder = nn.Parameter(torch.empty(ACTION_DIM, MOTORS))
        nn.init.xavier_uniform_(self.decoder)

    def initial_state(self, batch_size, device):
        return torch.zeros(batch_size, MOTORS, HIDDEN, device=device)

    def step(self, observation, state, t, block_messages=False, return_diagnostics=False, unit_mask=None):
        assert observation.ndim == 2 and observation.shape[-1] == OBS_DIM
        assert state.shape == (observation.shape[0], MOTORS, HIDDEN)
        local = (observation @ self.projection).reshape(-1, MOTORS, LOCAL_DIM)
        queries = torch.einsum("bkh,khm->bkm", state, self.query_weights)
        values = torch.einsum("bkh,khm->bkm", state, self.value_weights)
        gate = torch.sigmoid((queries * values).sum(-1, keepdim=True) / math.sqrt(MESSAGE_DIM))
        if self.arm == "local":
            source = values
        else:
            source = (values.sum(dim=1, keepdim=True) - values) / (MOTORS - 1)
        message = gate * source
        if block_messages:
            assert self.arm == "cross", "block_messages is a causal intervention on trained cross only"
            message = torch.zeros_like(message)
        updated = torch.stack([
            cell(torch.cat((local[:, i], message[:, i]), dim=-1), state[:, i])
            for i, cell in enumerate(self.cells)
        ], dim=1)
        activities = torch.cat([self.heads[i](updated[:, i]) for i in range(MOTORS)], dim=-1)
        if unit_mask is not None:
            mask = torch.as_tensor(unit_mask, device=activities.device, dtype=activities.dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            assert mask.ndim == 2 and mask.shape[1] == MOTORS
            assert mask.shape[0] in (1, observation.shape[0])
            assert bool(((mask == 0) | (mask == 1)).all())
            activities_to_output = activities * mask
        else:
            activities_to_output = activities
        action = torch.tanh(F.linear(activities_to_output, self.decoder))
        if return_diagnostics:
            return action, updated, {
                "activities": activities,
                "mean_abs_messages": message.abs().mean(dim=(0, 2)),
                "mean_abs_local_inputs": local.abs().mean(dim=(0, 2)),
            }
        return action, updated


def run_sequence(model, observations, block_messages=False, unit_mask=None):
    assert observations.ndim == 3 and observations.shape[-1] == OBS_DIM
    batch_size, n_steps, _ = observations.shape
    state = model.initial_state(batch_size, observations.device)
    actions = []
    for t in range(n_steps):
        action, state = model.step(observations[:, t], state, t, block_messages=block_messages, unit_mask=unit_mask)
        actions.append(action)
    return torch.stack(actions, dim=1)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
