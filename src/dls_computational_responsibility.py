"""实验性计算责任路由器。输入归属不等于功能责任，后者必须用反事实检验。"""
import math
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


class ComputationalResponsibilityRouter(nn.Module):
    """学习输入变量归属、模块间定向通信、故障下重分配；不预设变量语义。"""

    def __init__(self, num_features, num_modules=4, hidden_size=24,
                 message_size=4, num_outputs=2, balanced_access=True):
        super().__init__()
        assert num_features > 0 and num_modules >= 2
        assert hidden_size > 0 and message_size > 0 and num_outputs > 0
        self.num_features = num_features
        self.num_modules = num_modules
        self.num_outputs = num_outputs
        self.balanced_access = balanced_access
        self.ownership_logits = nn.Parameter(torch.zeros(num_modules, num_features))
        nn.init.normal_(self.ownership_logits, std=0.02)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(num_features, hidden_size), nn.Tanh())
            for _ in range(num_modules)])
        self.queries = nn.ModuleList([nn.Linear(hidden_size, message_size)
                                      for _ in range(num_modules)])
        self.keys = nn.ModuleList([nn.Linear(hidden_size, message_size)
                                   for _ in range(num_modules)])
        self.values = nn.ModuleList([nn.Linear(hidden_size, message_size)
                                     for _ in range(num_modules)])
        self.receivers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size + message_size, hidden_size),
                          nn.Tanh(), nn.Linear(hidden_size, 1), nn.Tanh())
            for _ in range(num_modules)])
        # 避免一个强大非线性输出头绕过模块的联合计算。
        self.readout = nn.Linear(num_modules, num_outputs)

    def ownership(self, available=None, temperature=1.0, *, hard=True):
        """[K,F]列和=1；可选均衡容量约束；ST前向为严格离散归属。"""
        assert temperature > 0
        if available is None:
            available = torch.ones(self.num_modules, dtype=torch.bool,
                                   device=self.ownership_logits.device)
        assert available.shape == (self.num_modules,)
        assert available.dtype == torch.bool and bool(available.any())
        logits = self.ownership_logits.masked_fill(~available[:, None], -1e9)
        if self.balanced_access:
            soft = torch.exp((logits - logits.max(dim=0, keepdim=True).values)
                             / temperature) * available[:, None].to(logits.dtype)
            target = self.num_features / int(available.sum().item())
            for _ in range(8):
                soft = soft / soft.sum(dim=0, keepdim=True).clamp_min(1e-12)
                soft = (soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-12)
                        * target * available[:, None].to(logits.dtype))
            soft = soft / soft.sum(dim=0, keepdim=True).clamp_min(1e-12)
        else:
            soft = torch.softmax(logits / temperature, dim=0)
        if not hard:
            return soft
        if self.balanced_access:
            # 将每个存活模块复制为 ceil(F/K_alive) 个输入槽，求最优匹配。
            live = torch.nonzero(available, as_tuple=False).flatten().tolist()
            slots = [i for i in live
                     for _ in range(math.ceil(self.num_features / len(live)))]
            costs = -logits.detach().cpu().numpy()[slots]
            slot_idx, feature_idx = linear_sum_assignment(costs)
            chosen = torch.zeros_like(soft)
            for slot, feature in zip(slot_idx, feature_idx):
                chosen[slots[int(slot)], int(feature)] = 1.0
        else:
            chosen = F.one_hot(soft.argmax(dim=0), self.num_modules).T.to(soft.dtype)
        return chosen - soft.detach() + soft

    def forward(self, x, *, available=None, temperature=1.0,
                communication=True, hard_ownership=True, diagnostics=False):
        assert x.ndim == 2 and x.shape[1] == self.num_features
        if available is None:
            available = torch.ones(self.num_modules, dtype=torch.bool, device=x.device)
        assert available.shape == (self.num_modules,)
        assert available.dtype == torch.bool and available.device == x.device
        assert bool(available.any())
        owners = self.ownership(available, temperature, hard=hard_ownership)
        states = torch.stack([
            module(x * owners[i]) for i, module in enumerate(self.encoders)
        ], dim=1)
        queries = torch.stack([m(states[:, i]) for i, m in enumerate(self.queries)], dim=1)
        keys = torch.stack([m(states[:, i]) for i, m in enumerate(self.keys)], dim=1)
        values = torch.stack([m(states[:, i]) for i, m in enumerate(self.values)], dim=1)
        affinity = queries @ keys.transpose(1, 2) / math.sqrt(keys.shape[-1])
        mask = ((available[None, :] &
                 ~torch.eye(self.num_modules, dtype=torch.bool, device=x.device))
                & available[:, None])
        attention = torch.softmax(affinity.masked_fill(~mask[None], -1e9), dim=-1)
        attention = attention * mask[None].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if not communication:
            attention = torch.zeros_like(attention)
        messages = attention @ values
        activity = torch.stack([
            module(torch.cat((states[:, i], messages[:, i]), dim=-1)).squeeze(-1)
            for i, module in enumerate(self.receivers)
        ], dim=1)
        activity = activity * available[None].to(activity.dtype)
        logits = self.readout(activity)
        output = torch.sigmoid(logits)
        if not diagnostics:
            return output
        return output, {
            "feature_ownership": owners.detach(),
            "directed_attention": attention.detach(),
            "module_activity": activity,
            "module_logits": activity[:, :, None] * self.readout.weight.T[None],
            "output_logits": logits,
        }


def lesion_effect(model, x, target, *, communication=True):
    """模块失活造成的MSE变化，允许负值，不能单独作为协同证明。"""
    assert x.shape[0] == target.shape[0]
    model.eval()
    with torch.no_grad():
        base = F.mse_loss(model(x, communication=communication), target)
        effects = []
        for i in range(model.num_modules):
            available = torch.ones(model.num_modules, dtype=torch.bool,
                                   device=x.device)
            available[i] = False
            pred = model(x, available=available, communication=communication)
            effects.append(F.mse_loss(pred, target) - base)
    return torch.stack(effects)
