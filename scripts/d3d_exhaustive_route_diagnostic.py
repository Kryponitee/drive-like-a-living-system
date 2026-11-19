#!/usr/bin/env python3
"""D3d诊断：在冻结模块权重下穷举 4^4=256 种路由。

区分「学习到的模块可支持正确合作，但路由优化选错」和
「冻结模块本身没形成互补专门化」。不重新训练，也不使用真值选模型。
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
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent))
from d3d_free_routing import FreeRoutingPedalController, ROOT, DATA_ID, SOURCE_DIR, REPORT_DIR, CHECKPOINT_DIR, INIT_SEEDS, TASKS

assert os.path.ismount("/mnt/datasets")
ALL_CHOICES = tuple(itertools.product(range(4),repeat=4))
assert len(ALL_CHOICES)==256


def evaluate_all_routes(model, x, y):
    model.eval()
    with torch.no_grad():
        messages=[module(x[:,i:i+1]) for i,module in enumerate(model.encoders)]
        activities=[]
        for receiver in range(4):
            candidate_activities=[]
            for sender in model.options[receiver]:
                m=torch.zeros_like(messages[receiver]) if sender == -1 else messages[sender]
                a=model.receivers[receiver](torch.cat((x[:,receiver:receiver+1],m),dim=1))
                candidate_activities.append(a[:,0])
            activities.append(torch.stack(candidate_activities,dim=0))
        routing_indices=torch.tensor(ALL_CHOICES,dtype=torch.long,device=x.device)
        matrix=torch.stack([
            activities[receiver].index_select(0,routing_indices[:,receiver])
            for receiver in range(4)
        ],dim=-1)
        logits=F.linear(matrix,model.decoder.weight,model.decoder.bias)
        y_pred=torch.sigmoid(logits)
        mse=((y_pred-y.unsqueeze(0))**2).mean(dim=(1,2))
        return mse.cpu().numpy()


def route_to_index(model,senders):
    selected=tuple(model.options[i].index(senders[i]) for i in range(4))
    return ALL_CHOICES.index(selected)


def partner_index_for_pairs(model,pairs):
    senders=[-1]*4
    for a,b in pairs:
        senders[a]=b
        senders[b]=a
    return route_to_index(model,senders)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-seed",type=int,choices=(42,43,44),required=True)
    p.add_argument("--gpu",type=int,default=-1)
    args=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(f"cuda:{args.gpu}") if args.gpu>=0 else torch.device("cpu")
    if args.gpu>=0:
        assert torch.cuda.is_available() and args.gpu<torch.cuda.device_count()

    source_manifest=json.loads((SOURCE_DIR/f"d3a-seed{args.data_seed}.json").read_text())
    data_path=SOURCE_DIR/f"calibration-data-seed{args.data_seed}.npz"
    assert hashlib.sha256(data_path.read_bytes()).hexdigest()==source_manifest["dataset_sha256"]
    results_path=REPORT_DIR/f"routing-exhaustive-seed{args.data_seed}.json"
    assert not results_path.exists()

    with np.load(data_path,allow_pickle=False) as data:
        x={
            split:torch.tensor(np.array(data[f"x_{split}"]),device=device)
            for split in ("val","test")
        }
        y={
            task:{
                split:torch.tensor(np.array(data[f"{'y' if task == 'interaction' else 'additive'}_{split}"]),device=device)
                for split in ("val","test")
            }
            for task in TASKS
        }

    truth=sorted([sorted(pair) for pair in source_manifest["ground_truth_pairs"]])
    complete_pairings = [
        ((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))
    ]
    rows=[]
    for task in TASKS:
        for init_seed in INIT_SEEDS:
            ckpt_path=CHECKPOINT_DIR/f"{task}-d{args.data_seed}-i{init_seed}.pt"
            ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=False)
            assert ckpt["task"]==task and ckpt["init_seed"]==init_seed and ckpt["data_seed"]==args.data_seed
            model=FreeRoutingPedalController().to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            learned_index=route_to_index(model,model.chosen_senders())

            val_errors=evaluate_all_routes(model,*(
                x["val"],y[task]["val"]))
            test_errors=evaluate_all_routes(model,*(
                x["test"],y[task]["test"]))
            best_index=int(np.argmin(val_errors))
            zero_index=route_to_index(model,[-1]*4)
            true_index=partner_index_for_pairs(model,truth)
            pair_ids=[partner_index_for_pairs(model,pairs) for pairs in complete_pairings]
            best_pair_index=min(pair_ids,key=lambda idx:float(val_errors[idx]))
            with torch.no_grad():
                actual_pred=model(x["val"],mode="hard")
                direct_mse=float(F.mse_loss(actual_pred,y[task]["val"]).item())
            assert np.isclose(direct_mse,val_errors[learned_index],rtol=2e-4,atol=1e-7)

            def decode(index):
                choices=ALL_CHOICES[index]
                return [model.options[i][choices[i]] for i in range(4)]

            row={
                "task":task,
                "data_seed":args.data_seed,
                "init_seed":init_seed,
                "learned_senders":decode(learned_index),
                "learned_val_mse":float(val_errors[learned_index]),
                "learned_test_mse":float(test_errors[learned_index]),
                "val_optimal_senders":decode(best_index),
                "val_optimal_val_mse":float(val_errors[best_index]),
                "val_optimal_test_mse":float(test_errors[best_index]),
                "learned_val_regret":float(val_errors[learned_index]-val_errors[best_index]),
                "learned_test_minus_val_selected":float(test_errors[learned_index]-test_errors[best_index]),
                "best_pair_val_selected_senders":decode(best_pair_index),
                "best_pair_val_selected_test_mse":float(test_errors[best_pair_index]),
                "best_pair_is_true":best_pair_index==true_index,
                "true_pair_senders":decode(true_index),
                "true_pair_val_mse":float(val_errors[true_index]),
                "true_pair_test_mse":float(test_errors[true_index]),
                "true_pair_val_rank_among_256":int(np.count_nonzero(val_errors<val_errors[true_index])+1),
                "true_pair_val_rank_among_3_complete_matchings":int(sum(val_errors[index]<val_errors[true_index] for index in pair_ids)+1),
                "no_comm_val_mse":float(val_errors[zero_index]),
                "no_comm_test_mse":float(test_errors[zero_index]),
                "unique_best_val_topology_index":best_index,
                "unique_best_test_topology_index_diagnostic_only":int(np.argmin(test_errors)),
            }
            rows.append(row)
            print(f"D3d exhaustive d={args.data_seed} i={init_seed} task={task} "
                  f"learned={row['learned_val_mse']:.6g} oracle-val={row['val_optimal_val_mse']:.6g} "
                  f"true-rank={row['true_pair_val_rank_among_256']} "
                  f"best-pair-true={row['best_pair_is_true']}",flush=True)

    result={
        "stage":"D3d frozen-weights exhaustive routing diagnostic",
        "scope":"4-module longitudinal-state surrogate, NOT MetaDrive closed-loop",
        "data_seed":args.data_seed,
        "input_sha256":source_manifest["dataset_sha256"],
        "candidate_topologies":256,
        "routing_selection":"argmin of validation MSE over 256 hard directed graphs; heldout test only afterward",
        "no_retraining":True,
        "truth_used_for_scoring_only":truth,
        "rows":rows,
    }
    results_path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({
        "stage":result["stage"],
        "data_seed":args.data_seed,
        "interaction_best_pair_true_count":sum(r["best_pair_is_true"] for r in rows if r["task"]=="interaction"),
        "interaction_mean_val_routing_regret":float(np.mean([r["learned_val_regret"] for r in rows if r["task"]=="interaction"])),
    },indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
