"""DLS 功能责任审计：反事实混合差分 + 模块 logit 分解 + 失效重归属。

用于人工独立可控输入。对真实相关状态必须由用户提供有效的条件反事实生成器，
否则打乱变量会落在数据流形之外，不能作物理因果解释。
"""
from itertools import combinations
import torch
from torch.nn import functional as F


class CRRAdapter:
    """在不修改 CRR 公共接口的条件下，读取其精确的模块输出贡献。"""

    def __init__(self, model):
        self.model = model
        self.num_features = model.num_features
        self.num_modules = model.num_modules

    def evaluate(self, x, *, available=None, communication=True):
        y, d = self.model(x, available=available, communication=communication,
                          diagnostics=True)
        module_logits = d["module_logits"]
        logits = d["output_logits"]
        bias = self.model.readout.bias
        assert module_logits.ndim == 3
        assert module_logits.shape[:2] == (x.shape[0], self.num_modules)
        assert torch.allclose(module_logits.sum(1) + bias, logits, atol=3e-6, rtol=1e-5)
        assert torch.allclose(torch.sigmoid(logits), y, atol=3e-6, rtol=1e-5)
        return {
            "pred": y,
            "logits": logits,
            "contributions": module_logits,
            "ownership": d["feature_ownership"],
        }


def _four_worlds(x, donor, i, j):
    one = x.clone()
    two = x.clone()
    both = x.clone()
    one[:, i] = donor[:, i]
    two[:, j] = donor[:, j]
    both[:, i] = donor[:, i]
    both[:, j] = donor[:, j]
    return x, one, two, both


def audit_pairwise_responsibility(adapter, x, donor, target_logit,
                                  *, available=None, communication=True,
                                  valid_interventions=False, epsilon=1e-9):
    """度量每个变量对在整体及各模块中实现的二阶计算责任。

    target_logit: 接受 [B,F] 输入并返回 [B,O] 的无状态真值函数；
    教学实验中只用于评分，绝不输入学习器。
    valid_interventions: 必须由调用者确认独立变量生成机制或有效条件反事实。
    """
    assert valid_interventions, "先证明反事实输入有效：相关真实特征不可直接交换"
    assert x.ndim == 2 and x.shape == donor.shape
    assert x.shape[1] == adapter.num_features
    assert not torch.equal(x, donor)
    assert target_logit is not None
    assert x.device == donor.device
    with torch.no_grad():
        original = adapter.evaluate(x, available=available,
                                    communication=communication)
        target = target_logit(x)
        assert target.shape == original["logits"].shape
        assert torch.isfinite(target).all()
        pred_loss = F.mse_loss(original["pred"], torch.sigmoid(target)).item()
        results = []
        for i, j in combinations(range(x.shape[1]), 2):
            worlds = _four_worlds(x, donor, i, j)
            scores = [
                adapter.evaluate(z, available=available,
                                 communication=communication)
                for z in worlds
            ]
            truth = [target_logit(z) for z in worlds]
            delta = scores[0]["logits"] - scores[1]["logits"] - scores[2]["logits"] + scores[3]["logits"]
            delta_module = (scores[0]["contributions"] - scores[1]["contributions"]
                            - scores[2]["contributions"] + scores[3]["contributions"])
            delta_target = truth[0] - truth[1] - truth[2] + truth[3]
            assert torch.allclose(delta_module.sum(1), delta, atol=6e-6, rtol=2e-5)
            true_energy = delta_target.square().mean().item()
            model_energy = delta.square().mean().item()
            error_energy = (delta - delta_target).square().mean().item()
            if true_energy > epsilon:
                projection = (delta_module * delta_target[:, None]).mean(dim=(0, 2)) / true_energy
                signed_share = projection.cpu().tolist()
                target_reconstruction = float((delta * delta_target).mean().item() / true_energy)
            else:
                signed_share = None
                target_reconstruction = None
            results.append({
                "pair": [i, j],
                "target_interaction_rms": true_energy ** 0.5,
                "learned_interaction_rms": model_energy ** 0.5,
                "interaction_error_rms": error_energy ** 0.5,
                "module_interaction_rms": delta_module.square().mean(dim=(0, 2)).sqrt().cpu().tolist(),
                "target_aligned_signed_module_projection": signed_share,
                "target_aligned_total_projection": target_reconstruction,
            })
        output = {
            "prediction_mse": float(pred_loss),
            "communication_enabled": communication,
            "available_module_indices": ([i for i in range(adapter.num_modules)]
                                         if available is None else available.nonzero().flatten().tolist()),
            "learned_feature_owners": original["ownership"].argmax(dim=0).cpu().tolist(),
            "input_pairs": results,
            "scope": "interventional functional decomposition in output logit space; not coordinate-invariant",
        }
        return output


def compare_failures(adapter, x, donor, target_logit, *,
                     valid_interventions=False):
    """对比健康、只断消息和各单模块失效后的功能责任。

    此处失效按现有 CRR 的逻辑重新分配输入，不能直接证明责任接管。
    """
    base = audit_pairwise_responsibility(
        adapter, x, donor, target_logit,
        valid_interventions=valid_interventions)
    no_comm = audit_pairwise_responsibility(
        adapter, x, donor, target_logit, communication=False,
        valid_interventions=valid_interventions)
    faults = []
    for k in range(adapter.num_modules):
        alive = torch.ones(adapter.num_modules, dtype=torch.bool, device=x.device)
        alive[k] = False
        result = audit_pairwise_responsibility(
            adapter, x, donor, target_logit, available=alive,
            valid_interventions=valid_interventions)
        assert k not in result["available_module_indices"]
        faults.append({"disabled_module": k, **result})
    return {"healthy": base, "communication_blocked": no_comm,
            "fault_with_input_reassignment": faults}
