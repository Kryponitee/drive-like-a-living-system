#!/usr/bin/env python3
"""读取 CRR checkpoint，独立于训练程序重做功能责任审计。

--target none 时只观察模型自身的非加性计算和模块责任，不访问真值内部公式。
此命令使用本实验的独立均匀输入，不可直接替代真实驾驶条件反事实。
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_computational_responsibility import ComputationalResponsibilityRouter
from dls_functional_responsibility import CRRAdapter,audit_pairwise_responsibility
from dls_functional_responsibility_experiment import target_logits


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--target",choices=("oracle","none"),default="none")
    args=p.parse_args()
    torch.set_num_threads(1)
    assert args.checkpoint.is_file() and not args.output.exists()
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    assert ckpt["architecture"]=="ComputationalResponsibilityRouter(4,4,32,6,2,balanced_access=True)"
    seed=ckpt["training_seed"]
    assert seed in (7,8,9)
    torch.manual_seed(seed+700000)
    _train=torch.rand(4096,4)*2-1
    _val=torch.rand(1024,4)*2-1
    x=torch.rand(1024,4)*2-1
    donor=torch.rand_like(x)*2-1
    del _train,_val
    model=ComputationalResponsibilityRouter(4,4,32,6,2,balanced_access=True)
    model.load_state_dict(ckpt["state_dict"],strict=True)
    model.eval()
    adapter=CRRAdapter(model)
    func=(lambda z:target_logits(z,ckpt["task"])) if args.target=="oracle" else None
    labels=torch.sigmoid(target_logits(x,ckpt["task"]))
    def audit(*,available=None,communication=True):
        return audit_pairwise_responsibility(
            adapter,x,donor,func,available=available,
            communication=communication,target_output=labels,
            valid_interventions=True)
    healthy=audit(communication=ckpt["communication"])
    blocked=audit(communication=False)
    faults=[]
    for i in range(4):
        alive=torch.ones(4,dtype=torch.bool)
        alive[i]=False
        faults.append({"disabled_module":i,**audit(available=alive,
                                                   communication=ckpt["communication"])})
    # 均可独立确认模块贡献严格加总为模型自身交互效应。
    for state in [healthy,blocked]+faults:
        for pair in state["input_pairs"]:
            coeff=pair["model_aligned_signed_module_projection"]
            if coeff is not None:
                assert abs(sum(coeff)-1)<5e-4
            if args.target=="none":
                assert pair["target_interaction_rms"] is None
                assert pair["target_aligned_signed_module_projection"] is None
    result={
        "stage":"Read-only CRR computational responsibility probe",
        "checkpoint":str(args.checkpoint),
        "seed":seed, "task":ckpt["task"],
        "target_mode":args.target,
        "scope":"independent toy counterfactuals, NOT physical driving interventions",
        "healthy":healthy,
        "communication_blocked":blocked,
        "single_fault_with_input_reallocation":faults,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({
        "checkpoint":str(args.checkpoint),"seed":seed,
        "target_mode":args.target,
        "healthy_test_mse":healthy["prediction_mse"],
        "blocked_mse":blocked["prediction_mse"],
        "mean_fault_mse":sum(t["prediction_mse"] for t in faults)/4,
        "learned_pairwise_interaction_rms":{
            str(pair["pair"]):pair["learned_interaction_rms"] for pair in healthy["input_pairs"]
        }
    },indent=2,ensure_ascii=False),flush=True)
    print("PASS target-free self-consistent computational responsibility attribution",flush=True)


if __name__=="__main__":
    main()
