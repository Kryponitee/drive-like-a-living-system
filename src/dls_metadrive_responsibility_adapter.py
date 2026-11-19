"""对已有 D1' LocalMotorController 做冻结权重的时间序列功能责任审计。

此适配器计算实际 8 个模块对两个双踏板 logit 的精确输出贡献。
不假设 259 维观测特征相互独立，也不任意打乱物理状态。
"""
import torch
from torch.nn import functional as F
from dls_longitudinal_motor import LocalMotorController, MODULES, OBS_DIM


class LongitudinalTemporalResponsibilityAdapter:
    def __init__(self, model):
        assert isinstance(model, LocalMotorController)
        self.model = model.eval()
        self.num_modules = MODULES
        self.num_features = OBS_DIM

    @torch.no_grad()
    def step(self, observation, state, t, *, block_messages=False):
        out, state_next, diagnostics = self.model.step(
            observation, state, t,
            block_messages=block_messages,
            return_diagnostics=True)
        activities = diagnostics["activities"]
        logits_k = activities.unsqueeze(-1) * self.model.decoder.T.unsqueeze(0)
        logits = logits_k.sum(1)
        assert torch.allclose(out, torch.sigmoid(logits), atol=2e-6, rtol=1e-5)
        return out, state_next, logits_k

    @torch.no_grad()
    def audit_sequence(self, observations, teacher, *, max_steps=None):
        assert observations.ndim == 2 and observations.shape[1] == OBS_DIM
        assert teacher.shape == (observations.shape[0], 2)
        assert torch.isfinite(observations).all() and torch.isfinite(teacher).all()
        if max_steps is not None:
            assert 0 < max_steps <= observations.shape[0]
            observations = observations[:max_steps]
            teacher = teacher[:max_steps]
        state = self.model.initial_state(1, observations.device)
        results = []
        for t, (obs, goal) in enumerate(zip(observations, teacher)):
            x = obs.unsqueeze(0)
            actual, new_state, components = self.step(x, state, t)
            # 单步反事实：阻断本步直接跨模块消息，保留同一历史状态。
            if self.model.arm == "cross":
                no_message, _, _ = self.step(x, state, t, block_messages=True)
            else:
                no_message = actual
            logits_k = components[0]
            # 单个模块只屏蔽当步输出到 decoder 的标量；
            # 不会切断其循环记忆，不作为长期损伤或责任接管证据。
            all_logits = logits_k.sum(0)
            no_unit_logits = all_logits.unsqueeze(0) - logits_k
            no_unit_predictions = torch.sigmoid(no_unit_logits)
            error_base = (actual[0] - goal).square().mean()
            error_blocked = (no_message[0] - goal).square().mean()
            error_no_units = (no_unit_predictions - goal).square().mean(dim=1)
            results.append({
                "actions": actual[0].cpu(),
                "unit_logit_contributions": logits_k.cpu(),
                "unit_lesion_increase_mse": (error_no_units - error_base).cpu(),
                "message_block_increase_mse": (error_blocked - error_base).cpu(),
                "teacher": goal.cpu(),
                "baseline_mse": error_base.cpu(),
            })
            state = new_state
        packed = {key: torch.stack([item[key] for item in results])
                  for key in results[0]}
        assert packed["unit_logit_contributions"].shape == (len(results), MODULES, 2)
        return packed


def summarise_responsibility(packed):
    """只报告可重复、可审计的模块贡献和瞬时失活效应。"""
    y = packed["teacher"]
    pred = packed["actions"]
    contribution = packed["unit_logit_contributions"]
    base = packed["baseline_mse"]
    labels = {
        "teacher_throttle_dominant": (y[:, 0] > y[:, 1] + 0.05),
        "teacher_brake_dominant": (y[:, 1] > y[:, 0] + 0.05),
        "teacher_near_zero": ((y[:, 0] <= 0.05) & (y[:, 1] <= 0.05)),
    }
    def compact(mask):
        n = int(mask.sum().item())
        if n == 0:
            return {"steps": 0, "mean_mse": None, "mean_abs_unit_logits": None,
                    "mean_unit_removal_effect": None, "mean_message_block_effect": None}
        return {
            "steps": n,
            "mean_mse": float(base[mask].mean().item()),
            "mean_abs_unit_logits": contribution[mask].abs().mean(0).tolist(),
            "mean_unit_removal_effect": packed["unit_lesion_increase_mse"][mask].mean(0).tolist(),
            "mean_message_block_effect": float(packed["message_block_increase_mse"][mask].mean().item()),
        }
    return {
        "num_steps": len(y),
        "overall": compact(torch.ones(len(y), dtype=torch.bool)),
        "teacher_conditioned": {key: compact(mask) for key, mask in labels.items()},
        "mse": float(F.mse_loss(pred, y).item()),
        "mean_pedal_coactivation": float(((pred[:, 0] > 0.1) & (pred[:, 1] > 0.1)).float().mean().item()),
        "interpretation_scope": "frozen-history, immediate module output & one-step message intervention; not long-horizon reorganization",
    }
