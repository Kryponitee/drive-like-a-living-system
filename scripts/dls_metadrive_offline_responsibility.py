#!/usr/bin/env python3
"""冻结 D1' B/C + 原始 MetaDrive 留出专家轨迹的真实观测责任审计。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_longitudinal_motor import model_info
from dls_metadrive_responsibility_adapter import (
    LongitudinalTemporalResponsibilityAdapter,summarise_responsibility)

DATA_ID="d1-longitudinal-v1-44117653285"
ROOT=Path("/mnt/datasets/lxb/dls")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("B","C"),required=True)
    p.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    p.add_argument("--episodes-per-density",type=int,default=2)
    p.add_argument("--max-steps",type=int,default=160)
    args=p.parse_args()
    assert 1<=args.episodes_per_density<=10
    assert 10<=args.max_steps<=800
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    output=ROOT/"reports"/DATA_ID/"responsibility-offline-v1"/f"{args.variant}-seed{args.seed}.json"
    assert not output.exists()
    ckpt=ROOT/"checkpoints"/DATA_ID/f"{args.variant}-seed{args.seed}.pt"
    checkpoint=torch.load(ckpt,map_location="cpu",weights_only=False)
    assert checkpoint["variant"]==args.variant
    assert checkpoint["data_id"]==DATA_ID
    model,_=model_info(args.variant)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    model.eval()
    adapter=LongitudinalTemporalResponsibilityAdapter(model)
    mean=np.asarray(checkpoint["train_mean"],dtype=np.float32)
    std=np.asarray(checkpoint["train_std"],dtype=np.float32)
    assert mean.shape==std.shape==(259,) and np.all(std>0)
    all_results=[]
    started=time.monotonic()
    for density in (0.0,0.1,0.25):
        stem=f"density-{density:.2f}"
        manifest_path=ROOT/"demonstrations"/DATA_ID/"test"/f"{stem}.json"
        archive_path=ROOT/"demonstrations"/DATA_ID/"test"/f"{stem}.npz"
        meta=json.loads(manifest_path.read_text())
        assert meta["split"]=="test" and float(meta["density"])==density
        assert hashlib.sha256(archive_path.read_bytes()).hexdigest()==meta["archive_sha256"]
        with np.load(archive_path,allow_pickle=False) as data:
            obs=np.asarray(data["observations"],dtype=np.float32)
            teacher=np.asarray(data["actions"],dtype=np.float32)
            offsets=np.asarray(data["episode_offsets"],dtype=np.int64)
            seeds=np.asarray(data["episode_seeds"],dtype=np.int64)
            success=np.asarray(data["episode_success"],dtype=np.bool_)
        assert obs.shape[1]==259 and teacher.shape==(len(obs),2)
        assert offsets[-1]==len(obs) and len(seeds)+1==len(offsets)
        for ep in range(args.episodes_per_density):
            start,end=map(int,offsets[ep:ep+2])
            assert 30000<=seeds[ep]<30300 and start<end
            if not bool(success[ep]):
                # 需要报告失败案例；不静默筛掉。
                pass
            limit=min(end-start,args.max_steps)
            x=np.clip((obs[start:start+limit]-mean)/std,-10,10).astype(np.float32)
            y=teacher[start:start+limit]
            packed=adapter.audit_sequence(torch.from_numpy(x),torch.from_numpy(y))
            summary=summarise_responsibility(packed)
            summary.update({
                "episode_seed":int(seeds[ep]), "density":density,
                "original_episode_success":bool(success[ep]),
                "original_episode_length":end-start,
                "prefix_steps":limit,
            })
            all_results.append(summary)
            print(f"{args.variant} seed={args.seed} density={density} scene={seeds[ep]} "
                  f"prefix={limit} mse={summary['mse']:.6f} "
                  f"message_delta={summary['overall']['mean_message_block_effect']:.6f}",
                  flush=True)
    assert len(all_results)==3*args.episodes_per_density
    def weighted(key):
        n=sum(r["num_steps"] for r in all_results)
        return sum(r["num_steps"]*r["overall"][key] for r in all_results)/n
    report={
        "stage":"DLS D1-prime frozen temporal responsibility adapter validation",
        "scope":"MetaDrive recorded observations / expert targets / teacher-forced one-step and recurrent hidden-state replay",
        "data_id":DATA_ID,"variant":args.variant,"training_seed":args.seed,
        "episodes_per_density":args.episodes_per_density,
        "max_steps_per_episode":args.max_steps,
        "num_observed_episodes":len(all_results),
        "total_observed_steps":sum(r["num_steps"] for r in all_results),
        "weighted_mse":weighted("mean_mse"),
        "weighted_one_step_message_block_loss_increase":weighted("mean_message_block_effect"),
        "checkpoint_path":str(ckpt),"checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "experiments":all_results,
        "limits":"No feature-swapped physical counterfactuals; no preassigned feature-to-module semantics; output ablation does not erase recurrence; not a new closed-loop test",
        "wall_seconds":time.monotonic()-started,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("PASS: MetaDrive D1-prime saved model, held-out real observations, exact module logit reconstruction",flush=True)


if __name__=="__main__":
    main()
