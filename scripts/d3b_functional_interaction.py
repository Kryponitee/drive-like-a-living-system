#!/usr/bin/env python3
"""D3b补充：盲功能交互识别，而非仅识别消息连通性。

对两路踏板的pre-sigmoid logit做双模块输入的有限差分混合导数。
采用独立验证集上无必要协作的"有通信"阴性模型校准阈值，
再在独立测试集判断真值组是否能被恢复。
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path("/mnt/datasets/lxb/dls")
DATA_ID="d1-longitudinal-v1-44117653285"
REPORT_ROOT=ROOT/"reports"/DATA_ID/"d3-identifiability-v1"
CKPT_ROOT=ROOT/"checkpoints"/DATA_ID/"d3-identifiability-v1"
VARIANTS=("paired","no_comm","wrong_pair","null_unwired","null_wired")
EPS=0.06
N=1024


def get_module():
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from d3_artificial_group_calibration import ModularPedalController
    return ModularPedalController


def input_pair_hessian(model, x, device):
    """任务函数的两路logit混合导数：只访问模型输入/输出，不读内部拓扑。"""
    x=torch.as_tensor(x[:N],dtype=torch.float32,device=device)
    assert x.shape==(N,4)
    def read_logits(inp):
        out=model(inp)
        return torch.logit(out.clamp(1e-6,1.0-1e-6))
    scores=np.zeros((4,4),dtype=np.float64)
    with torch.no_grad():
        for i,j in itertools.combinations(range(4),2):
            a=x.clone();b=x.clone();c=x.clone();d=x.clone()
            a[:,i]+=EPS;a[:,j]+=EPS
            b[:,i]+=EPS;b[:,j]-=EPS
            c[:,i]-=EPS;c[:,j]+=EPS
            d[:,i]-=EPS;d[:,j]-=EPS
            derivative=(read_logits(a)-read_logits(b)-read_logits(c)+read_logits(d))/(4*EPS*EPS)
            value=float(derivative.abs().mean().item())
            scores[i,j]=scores[j,i]=value
    return scores


def recover(score,threshold):
    possible=[[(0,1),(2,3)],[(0,2),(1,3)],[(0,3),(1,2)]]
    ranked=sorted(
        ((sum(score[i,j] for i,j in pairing),pairing) for pairing in possible),
        key=lambda item:item[0],reverse=True
    )
    winner=ranked[0][1]
    detected=all(score[i,j]>threshold for i,j in winner)
    return {
        "pairs":sorted([sorted(pair) for pair in winner]) if detected else [],
        "detected":bool(detected),
        "top_matching_score":float(ranked[0][0]),
        "second_matching_score":float(ranked[1][0]),
        "confidence_gap":float(ranked[0][0]-ranked[1][0]),
        "threshold":float(threshold),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",choices=(42,43,44),type=int,required=True)
    ap.add_argument("--gpu",type=int,default=-1)
    args=ap.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    device=torch.device(f"cuda:{args.gpu}") if args.gpu>=0 else torch.device("cpu")
    if args.gpu>=0:
        assert torch.cuda.is_available() and args.gpu<torch.cuda.device_count()
    model_class=get_module()
    old_report=json.loads((REPORT_ROOT/f"d3a-seed{args.seed}.json").read_text())
    data_path=REPORT_ROOT/f"calibration-data-seed{args.seed}.npz"
    assert hashlib.sha256(data_path.read_bytes()).hexdigest()==old_report["dataset_sha256"]
    report_path=REPORT_ROOT/f"d3b-functional-seed{args.seed}.json"
    assert not report_path.exists()
    with np.load(data_path,allow_pickle=False) as z:
        val_x=np.array(z["x_val"])
        test_x=np.array(z["x_test"])
    hessians={}
    for variant in VARIANTS:
        ckpt=torch.load(CKPT_ROOT/f"{variant}-seed{args.seed}.pt",weights_only=False,map_location="cpu")
        assert ckpt["variant"]==variant and ckpt["seed"]==args.seed
        model=model_class(ckpt["partners"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        hessians[variant]={
            "validation":input_pair_hessian(model,val_x,device),
            "test":input_pair_hessian(model,test_x,device),
        }
    null_val=hessians["null_wired"]["validation"]
    max_null_val=float(max(null_val[i,j] for i,j in itertools.combinations(range(4),2)))
    # 阈值只使用阴性条件的validation split，不读取positive真值和test标签。
    threshold=max(0.015, 2.0*max_null_val)
    results={}
    for variant in VARIANTS:
        test=hessians[variant]["test"]
        results[variant]={
            "test_mixed_partial_score":test.tolist(),
            "validation_mixed_partial_score":hessians[variant]["validation"].tolist(),
            "blind_group_inference":recover(test,threshold),
            "largest_pairwise_score":float(np.max(test)),
        }
        print(f"D3b-functional seed={args.seed} variant={variant} max={results[variant]['largest_pairwise_score']:.6f} pairs={results[variant]['blind_group_inference']['pairs']}",flush=True)
    truth=sorted([sorted(p) for p in old_report["ground_truth_pairs"]])
    recovered=results["paired"]["blind_group_inference"]["pairs"]==truth
    null_wired_fp=results["null_wired"]["blind_group_inference"]["detected"]
    null_unwired_fp=results["null_unwired"]["blind_group_inference"]["detected"]
    output={
        "stage":"D3b pre-sigmoid functional mixed-partial test",
        "seed":args.seed,
        "epsilon":EPS,
        "test_samples":N,
        "threshold_source":"max(0.015, 2 * maximum_pairwise_mixed_partial_of_null_wired_on_validation)",
        "calibrated_threshold":threshold,
        "null_wired_validation_max":max_null_val,
        "ground_truth_revealed_only_after_prediction":truth,
        "positive_exact_recovery":recovered,
        "null_wired_false_positive":null_wired_fp,
        "null_unwired_false_positive":null_unwired_fp,
        "scientific_pass":bool(recovered and not null_wired_fp and not null_unwired_fp),
        "all_variants":results,
        "interpretation":"This detects mixed functional dependence of the motor mapping; not proof that inter-module communication itself is naturally emergent.",
    }
    report_path.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps({
        "seed":args.seed,"threshold":threshold,"true":truth,
        "positive":results["paired"]["blind_group_inference"],
        "null_wired":results["null_wired"]["blind_group_inference"],
        "scientific_pass":output["scientific_pass"],
    },ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
