"""D5a：八个无预设语义的可学习运动单元，共同产生两个MetaDrive绝对控制量。"""

import torch
from torch import nn
from torch.nn import functional as F

from dls_p0 import OBS_DIM, ACTION_DIM, FEATURE_DIM, SLOW_DIM, SharedEncoder


MOTOR_UNITS = 8
MOTOR_HIDDEN = 29


class RedundantMotorController(SharedEncoder):
    """单元之间只有共同车辆观测/慢环；不额外引入跨单元消息或预设功能标签。"""

    def __init__(self):
        super().__init__()
        self.cells = nn.ModuleList([
            nn.GRUCell(FEATURE_DIM + SLOW_DIM, MOTOR_HIDDEN)
            for _ in range(MOTOR_UNITS)
        ])
        self.motor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(MOTOR_HIDDEN, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
                nn.Tanh(),
            )
            for _ in range(MOTOR_UNITS)
        ])
        # 每列对应一个单元的二维控制贡献；没有人为的转向/制动划分。
        self.decoder = nn.Parameter(torch.empty(ACTION_DIM, MOTOR_UNITS))
        nn.init.xavier_uniform_(self.decoder)

    def initial_state(self, batch_size, device):
        return (
            torch.zeros(batch_size, MOTOR_UNITS, MOTOR_HIDDEN, device=device),
            torch.zeros(batch_size, SLOW_DIM, device=device),
        )

    def step(self, obs, state, t, unit_mask=None, return_activities=False):
        assert obs.ndim == 2 and obs.shape[-1] == OBS_DIM
        h, z = state
        assert h.shape == (obs.shape[0], MOTOR_UNITS, MOTOR_HIDDEN)
        x, z = self.encode(obs, z, t)
        incoming = torch.cat((x, z), dim=-1)
        h_new = torch.stack([
            cell(incoming, h[:, i])
            for i, cell in enumerate(self.cells)
        ], dim=1)
        activity = torch.cat([
            head(h_new[:, i])
            for i, head in enumerate(self.motor_heads)
        ], dim=-1)
        if unit_mask is not None:
            mask = torch.as_tensor(unit_mask, dtype=activity.dtype, device=activity.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            assert mask.ndim == 2 and mask.shape[1] == MOTOR_UNITS
            assert mask.shape[0] in (1, obs.shape[0])
            assert bool(((mask == 0) | (mask == 1)).all())
            used = activity * mask
        else:
            used = activity
        action = torch.tanh(F.linear(used, self.decoder))
        if return_activities:
            return action, (h_new, z), activity
        return action, (h_new, z)


def run_motor_sequence(model, observations, unit_mask=None):
    """unit_mask固定在同一40步序列期间；[batch,8]，健康训练传None。"""
    assert observations.ndim == 3 and observations.shape[-1] == OBS_DIM
    batch, steps, _ = observations.shape
    state = model.initial_state(batch, observations.device)
    outputs = []
    for t in range(steps):
        action, state = model.step(observations[:, t], state, t, unit_mask=unit_mask)
        outputs.append(action)
    return torch.stack(outputs, dim=1)


def motor_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
