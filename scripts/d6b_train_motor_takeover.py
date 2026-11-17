#!/usr/bin/env python3
"""D6b：复用D5a的输出单元随机失活采样，在D6同容量局部观测L/X网络训练。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from d1_bc import ExpertSequences, SEQ
from d5a_train_redundant_motor import sample_unit_masks
from dls_local_motor import LocalMotorController, num_params, MOTORS, run_sequence


ROOT = Path("/mnt/datasets/lxb/dls")


def validate(model, loader, device, lesion_unit=None, blocked=False):
    model.eval()
    total = 0.0
    seen = 0.0
    with torch.no_grad():
        for x, y, time_mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            time_mask = time_mask.to(device, non_blocking=True)
            mask = None
            if lesion_unit is not None:
                mask = torch.ones(x.shape[0], MOTORS, device=device)
                mask[:, lesion_unit] = 0.0
            p = run_sequence(model, x, block_messages=blocked, unit_mask=mask)
            e = F.smooth_l1_loss(p, y, beta=0.1, reduction="none")
            total += float((e * time_mask.unsqueeze(-1)).sum())
            seen += float(time_mask.sum())
    assert int(seen) == 9822
    return total / (2 * seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("local","cross"), required=True)
    parser.add_argument("--seed", type=int, choices=(42,43,44), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert os.environ.get("DLS_D6B_TRAINING") == "1"
    assert torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
    assert SEQ == 40 and MOTORS == 8
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    train = ExpertSequences("train")
    val = ExpertSequences("val", train.mean, train.std)
    assert train.step_count == 91717 and val.step_count == 9822
    assert len(train.episodes) == 219 and len(val.episodes) == 23
    assert train.seed_set.isdisjoint(val.seed_set)
    model = LocalMotorController(args.arm)
    assert num_params(model) == 114880
    model.to(device)
    train_loader = DataLoader(train,batch_size=48,shuffle=True,num_workers=0,
                              pin_memory=True,drop_last=False)
    val_loader = DataLoader(val,batch_size=48,shuffle=False,num_workers=0,
                            pin_memory=True,drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-5)
    checkpointdir = ROOT/"checkpoints/d6b"
    reportdir = ROOT/"reports/d6b"
    tbdir = ROOT/"tensorboard/d6b"/f"{args.arm}-seed{args.seed}"
    for p in (checkpointdir,reportdir,tbdir):
        p.mkdir(parents=True,exist_ok=True)
    path = checkpointdir/f"{args.arm}-seed{args.seed}.pt"
    reportpath = reportdir/f"train-{args.arm}-seed{args.seed}.json"
    assert not path.exists() and not reportpath.exists()
    oldC=ROOT/"checkpoints/d1-v1"/f"C-seed{args.seed}.pt"
    oldsha=hashlib.sha256(oldC.read_bytes()).hexdigest()
    originalD6=ROOT/"checkpoints/d6"/f"{args.arm}-seed{args.seed}.pt"
    originalD6sha=hashlib.sha256(originalD6.read_bytes()).hexdigest()
    writer=SummaryWriter(log_dir=str(tbdir))
    history=[]
    nupdates=0
    start=time.monotonic()
    for epoch in range(8):
        model.train()
        trainloss=0.0
        seen=0.0
        for x,y,time_mask in train_loader:
            x=x.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)
            time_mask=time_mask.to(device,non_blocking=True)
            masks=sample_unit_masks(x.shape[0],device)
            predicted=run_sequence(model,x,unit_mask=masks)
            e=F.smooth_l1_loss(predicted,y,beta=0.1,reduction="none")
            loss=(e*time_mask.unsqueeze(-1)).sum()/(2*time_mask.sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            nupdates+=1
            n=float(time_mask.sum())
            seen+=n
            trainloss+=float(loss.detach())*n
        assert int(seen)==91717
        val_loss=validate(model,val_loader,device)
        history.append({"epoch":epoch+1,"train_masked_huber":trainloss/seen,"healthy_val_huber":val_loss})
        writer.add_scalar("bc/train_masked_huber",trainloss/seen,epoch+1)
        writer.add_scalar("bc/healthy_val_huber",val_loss,epoch+1)
        print(f"D6b {args.arm} seed{args.seed} epoch{epoch+1}/8 "
              f"train={trainloss/seen:.6f} val={val_loss:.6f}",flush=True)
    assert nupdates==408
    lesion_vals=[validate(model,val_loader,device,lesion_unit=i) for i in range(MOTORS)]
    blocked_lesion_vals=([validate(model,val_loader,device,lesion_unit=i,blocked=True)
                          for i in range(MOTORS)] if args.arm=="cross" else None)
    blocked_healthy=validate(model,val_loader,device,blocked=True) if args.arm=="cross" else None
    writer.add_scalar("bc/mean_single_lesion_val_huber",float(np.mean(lesion_vals)),8)
    writer.close()
    assert hashlib.sha256(oldC.read_bytes()).hexdigest()==oldsha
    assert hashlib.sha256(originalD6.read_bytes()).hexdigest()==originalD6sha
    torch.save({
        "state_dict":model.state_dict(),"variant":"D6b","arm":args.arm,"seed":args.seed,
        "num_params":114880,"num_motors":MOTORS,
        "data_id":"d1-v1","train_mean":train.mean,"train_std":train.std,
        "epochs":8,"batch_size":48,"optimizer_updates":nupdates,
        "healthy_val_huber":history[-1]["healthy_val_huber"],
        "single_lesion_val_huber":lesion_vals,
        "blocked_healthy_val_huber":blocked_healthy,
        "blocked_lesion_val_huber":blocked_lesion_vals,
        "old_C_sha256":oldsha,"original_D6_sha256":originalD6sha,
    },path)
    report={
        "stage":"D6b isolated output failure pressure with localized input",
        "arm":args.arm,"seed":args.seed,"gpu":args.gpu,
        "params":114880,"train_steps":train.step_count,"val_steps":val.step_count,
        "train_episodes":len(train.episodes),"val_episodes":len(val.episodes),
        "epochs":8,"batch_size":48,"optimizer_updates":nupdates,
        "lesion_distribution":"p(0)=0.25,p(1)=0.45,p(2)=0.30 (output-only)",
        "healthy_val_huber":history[-1]["healthy_val_huber"],
        "single_lesion_val_huber":lesion_vals,
        "blocked_healthy_val_huber":blocked_healthy,
        "blocked_lesion_val_huber":blocked_lesion_vals,
        "original_D6_sha256":originalD6sha,
        "old_C_sha256":oldsha,
        "checkpoint":str(path),"wall_seconds":time.monotonic()-start,
        "history":history,
    }
    reportpath.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"D6b {args.arm} seed{args.seed} complete "
          f"healthy={report['healthy_val_huber']:.6f} "
          f"mean_lesion={float(np.mean(lesion_vals)):.6f} "
          f"blocked_mean={float(np.mean(blocked_lesion_vals)) if blocked_lesion_vals is not None else None} "
          f"updates={nupdates}",flush=True)


if __name__=="__main__":
    main()
