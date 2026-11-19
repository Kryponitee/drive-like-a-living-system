#!/usr/bin/env python3
"""计算责任原型的最小完整性审计与教学型双变量交互试验。

数据是人工构造的纯计算任务；结果不是驾驶闭环，也不是运动协同证明。
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dls_computational_responsibility import (
    ComputationalResponsibilityRouter, lesion_effect,
)


def teacher(x):
    # 两类交互，均需跨变量非线性计算。
    a = 1.6 * (x[:, 0] * x[:, 1]) - 0.9 * (x[:, 2] * x[:, 3])
    b = -1.3 * (x[:, 0] * x[:, 1]) + 1.4 * (x[:, 2] * x[:, 3])
    return torch.sigmoid(torch.stack((a, b), dim=-1))


def validate_interfaces():
    torch.manual_seed(7)
    model = ComputationalResponsibilityRouter(5, num_modules=4)
    x = torch.randn(13, 5)
    target = torch.rand(13, 2)
    pred, audit = model(x, diagnostics=True)
    ownership = audit["feature_ownership"]
    assert pred.shape == (13, 2) and torch.isfinite(pred).all()
    assert torch.equal(ownership.sum(dim=0), torch.ones(5))
    assert ownership.sum(dim=1).max() <= 2
    assert torch.allclose(
        audit["module_logits"].sum(dim=1) + model.readout.bias,
        audit["output_logits"], atol=1e-6)
    assert torch.equal(
        audit["directed_attention"].diagonal(dim1=1, dim2=2),
        torch.zeros(13, 4))
    loss = F.mse_loss(pred, target)
    loss.backward()
    assert model.ownership_logits.grad is not None
    assert torch.isfinite(model.ownership_logits.grad).all()
    assert model.ownership_logits.grad.abs().sum() > 0
    failures = lesion_effect(model, x, target)
    assert failures.shape == (4,) and torch.isfinite(failures).all()
    for k in range(4):
        alive = torch.ones(4, dtype=torch.bool)
        alive[k] = False
        p, d = model(x, available=alive, diagnostics=True)
        assert torch.isfinite(p).all()
        assert torch.all(d["feature_ownership"][k] == 0)
        assert torch.all(d["directed_attention"][:, :, k] == 0)
        assert torch.all(d["directed_attention"][:, k, :] == 0)
        assert torch.all(d["module_activity"][:, k] == 0)
        assert d["feature_ownership"].sum(dim=1).max() <= 2
    only_one = torch.tensor([True, False, False, False])
    p, d = model(x, available=only_one, diagnostics=True)
    assert torch.isfinite(p).all()
    assert torch.all(d["directed_attention"] == 0)


def train(seed, communication, fault_training=False):
    torch.manual_seed(seed)
    train_x = 2 * torch.rand(4096, 4) - 1
    train_y = teacher(train_x)
    test_x = 2 * torch.rand(1024, 4) - 1
    test_y = teacher(test_x)
    model = ComputationalResponsibilityRouter(
        num_features=4, num_modules=4, hidden_size=32,
        message_size=6, num_outputs=2, balanced_access=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    for epoch in range(25):
        for ids in torch.randperm(len(train_x)).split(256):
            available = torch.ones(4, dtype=torch.bool)
            if fault_training and torch.rand(()) < 0.4:
                available[int(torch.randint(4, (1,)))] = False
            pred = model(train_x[ids], available=available,
                         communication=communication)
            loss = F.mse_loss(pred, train_y[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        output, diag = model(test_x, communication=communication, diagnostics=True)
        healthy_mse = float(F.mse_loss(output, test_y).item())
        owners = diag["feature_ownership"].argmax(dim=0).tolist()
        failure_mses = []
        for k in range(4):
            available = torch.ones(4, dtype=torch.bool)
            available[k] = False
            output = model(test_x, available=available,
                           communication=communication)
            failure_mses.append(float(F.mse_loss(output, test_y).item()))
    return {
        "seed": seed,
        "communication": communication,
        "fault_training": fault_training,
        "test_healthy_mse": healthy_mse,
        "test_one_module_failure_mse_mean": sum(failure_mses) / len(failure_mses),
        "test_one_module_failure_mse": failure_mses,
        "feature_owner_module_indices": owners,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    validate_interfaces()
    rows = []
    for seed in (7, 8):
        for communication in (False, True):
            result = train(seed, communication)
            rows.append(result)
            print(json.dumps(result), flush=True)
    # 只改变是否在训练过程中随机暂时失活单个模块。
    result = train(7, True, fault_training=True)
    rows.append(result)
    print(json.dumps(result), flush=True)
    report = {
        "stage": "DLS computational responsibility primitive v0: smoke and controlled toy",
        "scope": "synthetic algebraic task, NOT MetaDrive or motor-synergy evidence",
        "conditions": "4 inputs, 4 modules, 1 feature/module while healthy, input assignment learned, directed soft communication",
        "training": "25 epochs / Adam 0.007 / batch256 / 4096 train / 1024 held-out",
        "results": rows,
    }
    if args.output is not None:
        assert not args.output.exists()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("PASS: exact access budget / routing gradient / directed-message mask / fault reassignment / logit decomposition", flush=True)


if __name__ == "__main__":
    main()
