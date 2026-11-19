#!/usr/bin/env python3
"""DLS 计算责任测量工具：正负函数真值校准、学习网络归因与故障责任漂移。

人工独立输入；实验只验证工具与受约束计算，不宣称驾驶协调组已经形成。
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dls_computational_responsibility import ComputationalResponsibilityRouter
from dls_functional_responsibility import (
    CRRAdapter, audit_pairwise_responsibility, compare_failures,
)


def target_logits(x, task):
    assert x.shape[-1] == 4
    if task == "interaction":
        z01 = x[:, 0] * x[:, 1]
        z23 = x[:, 2] * x[:, 3]
        return torch.stack((1.6 * z01 - 0.9 * z23,
                            -1.3 * z01 + 1.4 * z23), dim=1)
    assert task == "additive"
    weights = x.new_tensor([[0.4, -0.5],
                            [0.8, 0.3],
                            [-0.6, 0.7],
                            [0.3, -0.2]])
    return x @ weights


class ExactOracle:
    """显式按两组交互实现的函数，只作为测量仪器的数学真值。"""

    num_features = 4
    num_modules = 4

    def __init__(self, task):
        self.task = task

    def evaluate(self, x, *, available=None, communication=True):
        assert available is None and communication
        component = torch.zeros(x.shape[0], 4, 2, device=x.device, dtype=x.dtype)
        if self.task == "interaction":
            a = x[:, 0] * x[:, 1]
            b = x[:, 2] * x[:, 3]
            component[:, 0, 0] = 1.6 * a
            component[:, 0, 1] = -1.3 * a
            component[:, 1, 0] = -0.9 * b
            component[:, 1, 1] = 1.4 * b
        else:
            weights = x.new_tensor([[0.4, -0.5],
                                    [0.8, 0.3],
                                    [-0.6, 0.7],
                                    [0.3, -0.2]])
            component = x[:, :, None] * weights[None]
        logits = component.sum(dim=1)
        assert torch.allclose(logits, target_logits(x, self.task), atol=1e-6)
        return {"pred": torch.sigmoid(logits), "logits": logits,
                "contributions": component,
                "ownership": torch.eye(4, dtype=x.dtype, device=x.device)}


def check_oracle(seed, x, donor):
    out = {}
    for task in ("interaction", "additive"):
        oracle = ExactOracle(task)
        result = audit_pairwise_responsibility(
            oracle, x, donor, lambda z: target_logits(z, task),
            valid_interventions=True)
        assert result["prediction_mse"] < 1e-12
        pair = {tuple(r["pair"]):r for r in result["input_pairs"]}
        if task == "interaction":
            assert pair[(0, 1)]["target_interaction_rms"] > 0.1
            assert pair[(2, 3)]["target_interaction_rms"] > 0.1
            assert all(pair[p]["target_interaction_rms"] < 1e-6 for p in pair if p not in ((0,1), (2,3)))
            assert pair[(0, 1)]["target_aligned_signed_module_projection"][0] > 0.999
            assert pair[(2, 3)]["target_aligned_signed_module_projection"][1] > 0.999
            assert all(abs(pair[p]["target_aligned_total_projection"] - 1) < 1e-5
                       for p in ((0,1), (2,3)))
        else:
            assert all(p["target_interaction_rms"] < 1e-6 for p in pair.values())
            assert all(p["target_aligned_signed_module_projection"] is None for p in pair.values())
        out[task] = {"max_interaction_error_rms":max(r["interaction_error_rms"] for r in result["input_pairs"]),
                     "pair_scores": {str(k):v["target_interaction_rms"] for k,v in pair.items()}}
    print(f"PASS exact-oracle and additive-negative calibration: seed={seed}", flush=True)
    return out


def train(seed, task, communication, fault_training, train_x, val_x):
    torch.manual_seed(130000 + seed)
    model = ComputationalResponsibilityRouter(4, 4, 32, 6, 2, balanced_access=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    train_y = torch.sigmoid(target_logits(train_x, task))
    val_y = torch.sigmoid(target_logits(val_x, task))
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    for epoch in range(1, 26):
        model.train()
        order = torch.randperm(len(train_x))
        for ids in order.split(256):
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
            val = F.mse_loss(model(val_x, communication=communication), val_y).item()
        if val < best_val:
            best_val = val
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    assert best_epoch > 0
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_val


def summarize_run(seed, task, communication, fault_training, model, x, donor,
                  best_epoch, validation_mse):
    adapter = CRRAdapter(model)
    func = lambda z: target_logits(z, task)
    audit = compare_failures(adapter, x, donor, func, valid_interventions=True)
    if not communication:
        audit = {"healthy": audit["communication_blocked"],
                 "communication_blocked": audit["communication_blocked"],
                 "fault_with_input_reassignment": [
                     audit_pairwise_responsibility(adapter, x, donor, func,
                                                  available=torch.arange(4)!=i,
                                                  communication=False,
                                                  valid_interventions=True)
                     for i in range(4)
                 ]}
    pos_pairs = {(0,1),(2,3)}
    def compact(report):
        pairs = {tuple(a["pair"]):a for a in report["input_pairs"]}
        return {
            "mse":report["prediction_mse"],
            "feature_owners":report["learned_feature_owners"],
            "target_pair_interaction_rms":{str(pair):pairs[pair]["target_interaction_rms"] for pair in pos_pairs},
            "model_pair_interaction_rms":{str(pair):pairs[pair]["learned_interaction_rms"] for pair in pos_pairs},
            "wrong_pair_max_interaction_rms":max(v["learned_interaction_rms"] for p,v in pairs.items() if p not in pos_pairs),
            "pair_module_alignment":{
                str(pair):pairs[pair]["target_aligned_signed_module_projection"]
                for pair in pos_pairs
            },
            "pair_total_alignment":{
                str(pair):pairs[pair]["target_aligned_total_projection"]
                for pair in pos_pairs
            },
        }
    main = compact(audit["healthy"])
    no_message = compact(audit["communication_blocked"])
    failures = [compact(d) for d in audit["fault_with_input_reassignment"]]
    result = {
        "seed":seed,"task":task,"train_with_communication":communication,
        "fault_training":fault_training,
        "best_epoch":best_epoch,"validation_mse":validation_mse,
        "healthy":main, "communication_blocked":no_message,
        "four_single_module_faults_reassigned":failures,
        "mean_fault_mse":sum(r["mse"] for r in failures)/4,
    }
    print(f"assay seed={seed} task={task} comm={communication} faulttrain={fault_training} "
          f"MSE={main['mse']:.7f} no_msg={no_message['mse']:.7f} "
          f"fault_mean={result['mean_fault_mse']:.7f} owners={main['feature_owners']}", flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=(7,8,9),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint-dir",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    assert not args.output.exists()

    torch.manual_seed(args.seed + 700000)
    train_x = torch.rand(4096, 4)*2 - 1
    val_x = torch.rand(1024, 4)*2 - 1
    test_x = torch.rand(1024, 4)*2 - 1
    donor = torch.rand_like(test_x)*2 - 1
    oracle = check_oracle(args.seed, test_x, donor)

    rows = []
    for task, communication, fault_training in (
        ("interaction",False,False),
        ("interaction",True,False),
        ("interaction",True,True),
        ("additive",True,False),
    ):
        model,epoch,val = train(args.seed,task,communication,fault_training,
                                train_x,val_x)
        ckpt = args.checkpoint_dir / f"{task}-comm{int(communication)}-fault{int(fault_training)}-seed{args.seed}.pt"
        assert not ckpt.exists()
        args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":model.state_dict(),
                    "architecture":"ComputationalResponsibilityRouter(4,4,32,6,2,balanced_access=True)",
                    "training_seed":args.seed, "task":task, "communication":communication,
                    "fault_training":fault_training, "epochs":25},ckpt)
        row=summarize_run(args.seed,task,communication,fault_training,
                          model,test_x,donor,epoch,val)
        row["checkpoint"]=str(ckpt)
        rows.append(row)
    output={
        "stage":"DLS functional responsibility audit v1",
        "seed":args.seed,
        "scope":"independent synthetic inputs / sigmoid two-channel readout / logit attribution",
        "oracle_calibration":oracle,
        "method":"paired factorial finite differences in output logits, decomposed exactly into module-level logit contributions",
        "limitations":"attribution depends on module basis; faults reroute input by existing CRR rule; no closed loop",
        "rows":rows,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print("PASS: positive/negative oracle, model logit decomposition, counterfactual pairwise responsibility, 4 failure routes", flush=True)


if __name__=="__main__":
    main()
