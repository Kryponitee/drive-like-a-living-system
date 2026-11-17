"""P0 的 A/B/C 循环控制器：B/C 在可训练参数上严格匹配，不人工指定模块功能。"""

import math

import torch
from torch import nn


OBS_DIM = 259
ACTION_DIM = 2
FEATURE_DIM = 64
SLOW_DIM = 16
MODULES = 4
MODULE_HIDDEN = 32
SLOW_PERIOD = 5


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(OBS_DIM, FEATURE_DIM), nn.Tanh())
        self.slow = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, SLOW_DIM), nn.Tanh()
        )

    def encode(self, obs, z, step_index):
        if step_index % SLOW_PERIOD == 0:
            z = self.slow(obs)
        x = self.features(obs)
        return x, z


class MonolithicController(SharedEncoder):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.rnn = nn.GRUCell(FEATURE_DIM + SLOW_DIM, hidden)
        self.action_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, ACTION_DIM)
        )

    def initial_state(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.hidden, device=device),
            torch.zeros(batch_size, SLOW_DIM, device=device),
        )

    def step(self, obs, state, t):
        h, z = state
        x, z = self.encode(obs, z, t)
        h = self.rnn(torch.cat([x, z], -1), h)
        action = self.action_head(h).tanh()
        return action, (h, z)


class ModularController(SharedEncoder):
    def __init__(self, mode):
        super().__init__()
        assert mode in ("B", "C")
        self.mode = mode
        self.cells = nn.ModuleList(
            [nn.GRUCell(FEATURE_DIM + SLOW_DIM + MODULE_HIDDEN, MODULE_HIDDEN)
             for _ in range(MODULES)]
        )
        # 相同的 q/k/value 参数：B 只处理自身 h_i，C 才读取其他 h_j。
        self.query = nn.Linear(MODULE_HIDDEN, 16)
        self.key = nn.Linear(MODULE_HIDDEN, 16)
        self.value_weights = nn.Parameter(torch.empty(MODULES, MODULE_HIDDEN, MODULE_HIDDEN))
        nn.init.xavier_uniform_(self.value_weights)
        self.output_weights = nn.Parameter(torch.empty(MODULES, MODULE_HIDDEN, MODULE_HIDDEN))
        nn.init.xavier_uniform_(self.output_weights)
        self.router = nn.Sequential(
            nn.Linear(FEATURE_DIM + SLOW_DIM + MODULES * MODULE_HIDDEN, 64),
            nn.Tanh(),
            nn.Linear(64, MODULES),
        )
        self.action_head = nn.Sequential(
            nn.Linear(MODULE_HIDDEN, 32), nn.Tanh(), nn.Linear(32, ACTION_DIM)
        )

    def initial_state(self, batch_size, device):
        return (
            torch.zeros(batch_size, MODULES, MODULE_HIDDEN, device=device),
            torch.zeros(batch_size, SLOW_DIM, device=device),
        )

    def step(self, obs, state, t):
        h, z = state
        x, z = self.encode(obs, z, t)
        q = self.query(h)
        k = self.key(h)
        v = torch.einsum("bkd,kdf->bkf", h, self.value_weights)
        if self.mode == "C":
            # 自身链接被严格屏蔽，所有消息只来自 j != i 的旧状态。
            scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
            diag = torch.eye(MODULES, device=scores.device, dtype=torch.bool).unsqueeze(0)
            weights = scores.masked_fill(diag, float("-inf")).softmax(dim=-1)
            aux = torch.bmm(weights, v)
        else:
            # 局部-only 对照：同数目参数、同维消息分支，却不接收其他模块的状态。
            gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / math.sqrt(q.shape[-1]))
            aux = v * gate
        h_next = torch.stack([
            self.cells[i](torch.cat([x, z, aux[:, i]], dim=-1), h[:, i])
            for i in range(MODULES)
        ], dim=1)
        routing_logits = self.router(torch.cat([x, z, h_next.flatten(1)], dim=-1))
        alpha = routing_logits.softmax(dim=-1)
        projected = torch.einsum("bkd,kdf->bkf", h_next, self.output_weights)
        pooled = (projected * alpha.unsqueeze(-1)).sum(dim=1)
        action = self.action_head(pooled).tanh()
        return action, (h_next, z)


def nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_info(variant):
    assert variant in ("A", "B", "C")
    if variant == "A":
        target = nparams(ModularController("C"))
        hidden = min(range(32, 256), key=lambda h: abs(nparams(MonolithicController(h)) - target))
        return MonolithicController(hidden), {"variant": "A", "hidden": hidden, "reference_params": target}
    model = ModularController(variant)
    return model, {"variant": variant, "K": MODULES, "d": MODULE_HIDDEN, "q": SLOW_PERIOD}


def run_sequence(model, observations):
    assert observations.ndim == 3 and observations.shape[-1] == OBS_DIM
    batch, steps, _ = observations.shape
    state = model.initial_state(batch, observations.device)
    predictions = []
    for t in range(steps):
        action, state = model.step(observations[:, t], state, t)
        predictions.append(action)
    return torch.stack(predictions, dim=1)
