#!/usr/bin/env python3
"""D6 A-only strictly paired heldout MetaDrive following evaluation.

Reuse D4's exact sham-vs-pulse physical seeds and protocol. This makes no
assumptions about a motor-coordination-group effect.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from dls_d4_eval_fullseq80 import rollout,verify_matching,TEST_SEEDS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_longitudinal_motor import model_info

ROOT=Path("/mnt/datasets/lxb/dls")
DATA_ID="d4-dynamic-follow-v1-44481405397"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=("replay","onpolicy"),required=True)
    p.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    a=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    ckpt=ROOT/"checkpoints"/DATA_ID/"d6-a-pilot"/f"{a.arm}-seed{a.seed}.pt"
    path=ROOT/"reports"/DATA_ID/"d6-a-pilot"/"evaluation"/f"{a.arm}-seed{a.seed}.json"
    assert ckpt.is_file() and not path.exists()
    weights=torch.load(ckpt,map_location="cpu",weights_only=False)
    assert weights["variant"]=="A" and weights["seed"]==a.seed and weights["arm"]==a.arm
    mean=np.asarray(weights["train_mean"],dtype=np.float32)
    std=np.asarray(weights["train_std"],dtype=np.float32)
    model,_=model_info("A")
    model.load_state_dict(weights["state_dict"],strict=True)
    model.eval()
    rows=[];matched=0
    for scene in TEST_SEEDS:
        sham=rollout(model,"A",mean,std,scene,"sham")
        pulse=rollout(model,"A",mean,std,scene,"pulse")
        matched+=int(verify_matching(sham,pulse))
        rows.extend([sham,pulse])
        print("D6_EVAL",a.arm,a.seed,scene,
              "sham safe",sham["success"] and not sham["collision"],
              "pulse safe",pulse["success"] and not pulse["collision"],
              "pulse collision",pulse["collision"],flush=True)
    assert len(rows)==24 and matched==12
    report={
        "stage":"D6 causal data-coverage experiment: A onpolicy vs A replay matched-budget",
        "arm":a.arm,"seed":a.seed,"data_id":DATA_ID,
        "checkpoint":str(ckpt),
        "checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "matched_pulse_sham_prefixes":matched,
        "test_scenes":list(TEST_SEEDS),
        "episodes":rows,
        "student_input":"unmodified MetaDrive 259D",
        "scope":"one dataset-aggregation iteration, fixed A GRU model, new teacher labels only at visited states",
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("D6_EVAL_COMPLETE",a.arm,a.seed,"matched",matched,flush=True)

if __name__=="__main__":
    main()
