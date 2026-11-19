#!/usr/bin/env python3
"""DLS D5: quantify recoverable leader state from ordinary MetaDrive 259D sensors.

Positive sensor-decoding results imply information is present in principle,
not that A/B/C have learned or can exploit it. Teacher trajectories and
fullseq80 on-policy trajectories are reported separately.
"""
import copy,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path("/mnt/datasets/lxb/dls")
DATA_ID="d4-dynamic-follow-v1-44481405397"
BASE=ROOT/"reports"/DATA_ID/"d5-observability-v1"
OUT=BASE/"sensor-decodability-report.json"


def load(split):
    path=BASE/f"{split}.npz"
    expected=json.loads((BASE/f"{split}.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected["sha256"]
    with np.load(path,allow_pickle=False) as d:
        result={key:np.asarray(d[key]).copy() for key in d.files}
    x=result["observations"];y=result["labels"]
    assert x.ndim==2 and x.shape[1]==259
    assert y.shape==(len(x),2)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    offsets=result["episode_offsets"]
    assert offsets[0]==0 and offsets[-1]==len(x)
    assert len(offsets)==len(result["episode_seeds"])+1
    assert len(offsets)==len(result["episode_modes"])+1
    return result


def features(dataset,variant,mean,std):
    obs=np.clip((dataset["observations"]-mean)/std,-8.,8.)
    if variant=="spatial":
        return obs.astype(np.float32)
    if variant=="temporal":
        diff=np.zeros_like(obs)
        for start,end in zip(dataset["episode_offsets"][:-1],
                             dataset["episode_offsets"][1:]):
            start=int(start);end=int(end)
            index=np.arange(start,end)
            lag=np.maximum(index-3,start)
            diff[start:end]=obs[index]-obs[lag]
        return np.concatenate([obs,diff],axis=1).astype(np.float32)
    assert variant=="time_ego"
    # A deliberately strong negative control: privilege true EGO speed,
    # but no lead-car speed, range, braking schedule, or other lead states.
    return np.stack([dataset["times"]/100.,dataset["ego_speeds"]/10.],axis=1).astype(np.float32)


class Probe(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n,128),nn.GELU(),
            nn.Linear(128,128),nn.GELU(),nn.Linear(128,2))

    def forward(self,x):
        return self.net(x)


def train_probe(variant,train,val,obs_mean,obs_std,ymean,ystd,device):
    torch.manual_seed(2410)
    xtrain=torch.from_numpy(features(train,variant,obs_mean,obs_std)).to(device)
    ytrain=torch.from_numpy(((train["labels"]-ymean)/ystd).astype(np.float32)).to(device)
    xval=torch.from_numpy(features(val,variant,obs_mean,obs_std)).to(device)
    yval=torch.from_numpy(((val["labels"]-ymean)/ystd).astype(np.float32)).to(device)
    assert xtrain.shape[1] in (2,259,518)
    model=Probe(xtrain.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    best=float("inf");best_epoch=-1;best_params=None
    for epoch in range(1,81):
        model.train()
        order=torch.randperm(len(xtrain),device=device)
        for inds in order.split(512):
            pred=model(xtrain[inds])
            loss=F.mse_loss(pred,ytrain[inds])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            error=float(F.mse_loss(model(xval),yval).item())
        if error<best:
            best=error;best_epoch=epoch
            best_params={key:value.detach().cpu().clone()
                         for key,value in model.state_dict().items()}
        if epoch%20==0:
            print("PROBE_TRAIN",variant,epoch,"best",best,
                  "val",error,flush=True)
    assert best_params is not None
    model.load_state_dict(best_params)
    model.eval()
    return model,{"best_epoch":best_epoch,"val_standardized_mse":best,
                  "train_samples":len(xtrain),
                  "train_dims":xtrain.shape[1]}


def metric(model,data,variant,mean,std,ymean,ystd,device):
    x=features(data,variant,mean,std)
    with torch.no_grad():
        y_pred=torch.cat([
            model(torch.from_numpy(q).to(device)).cpu()
            for q in np.array_split(x,max(1,int(np.ceil(len(x)/8192))))
        ],dim=0).numpy()*ystd+ymean
    truth=data["labels"]
    assert y_pred.shape==truth.shape and np.isfinite(y_pred).all()
    t=data["times"]
    event=data["event_steps"]
    masks={
        "all":np.ones(len(t),dtype=np.bool_),
        "pre_event_15":(t>=event-15)&(t<event),
        "post_event_0_15":(t>=event)&(t<event+15),
        "post_event_15_30":(t>=event+15)&(t<event+30),
    }
    result={}
    for key,mask in masks.items():
        if not mask.any():
            result[key]={"count":0}
            continue
        e=y_pred[mask]-truth[mask]
        result[key]={
            "count":int(mask.sum()),
            "gap_mae_m":float(np.mean(np.abs(e[:,0]))),
            "closing_speed_mae_mps":float(np.mean(np.abs(e[:,1]))),
            "gap_rmse_m":float(np.sqrt(np.mean(e[:,0]**2))),
            "closing_speed_rmse_mps":float(np.sqrt(np.mean(e[:,1]**2))),
            "reference_gap_std_m":float(truth[mask,0].std()),
            "reference_closing_speed_std_mps":float(truth[mask,1].std()),
        }
    return result


def main():
    assert os.path.ismount("/mnt/datasets")
    assert not OUT.exists()
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    device=torch.device("cuda:0")
    train=load("train");val=load("val")
    evals={"heldout_teacher":load("test")}
    for variant in "ABC":
        evals[f"heldout_onpolicy_{variant}"]=load(f"onpolicy-{variant}")
    assert len(train["episode_seeds"])==72 and len(val["episode_seeds"])==24
    assert len(evals["heldout_teacher"]["episode_seeds"])==36
    train_set=set(int(z) for z in train["episode_seeds"])
    val_set=set(int(z) for z in val["episode_seeds"])
    test_set=set(int(z) for z in evals["heldout_teacher"]["episode_seeds"])
    assert not (train_set&val_set or train_set&test_set or val_set&test_set)
    mean=train["observations"].mean(axis=0,dtype=np.float64).astype(np.float32)
    std=np.maximum(train["observations"].std(axis=0,dtype=np.float64).astype(np.float32),.01)
    ymean=train["labels"].mean(axis=0).astype(np.float32)
    ystd=np.maximum(train["labels"].std(axis=0).astype(np.float32),.5)
    report={
        "stage":"DLS D5 259D sensor-to-real-physics state decodability",
        "data_id":DATA_ID,
        "method":"same 2-layer MLP/80 epochs/val-selected for sensor instantaneous, sensor+3-frame difference, time+true ego speed negative control",
        "training_scope":"only 72 privileged-physics-labelled teacher episodes, not model policy actions",
        "teacher_input":"259D normal observation only; physics labels are output supervision",
        "time_control":"elapsed time and true ego speed, but NOT lead state or action schedule",
        "teacher_replay_verified":True,
        "n_training_samples":len(train["labels"]),
        "n_validation_samples":len(val["labels"]),
        "targets":["surface_gap_m","ego_speed_minus_lead_speed_mps"],
        "results":{},
        "caveats":[
            "positive decoding is a lower bound on information available, not proof that current policies decode it",
            "teacher-state testing and on-policy state shift are evaluated separately",
            "time+ego negative control uses privileged ego speed, deliberately favorable to time shortcut",
            "this probe is not deployed in vehicle control; it does not establish closed-loop safety",
        ],
    }
    for variant in ("time_ego","spatial","temporal"):
        model,info=train_probe(variant,train,val,mean,std,ymean,ystd,device)
        scores={name:metric(model,data,variant,mean,std,ymean,ystd,device)
                for name,data in evals.items()}
        report["results"][variant]={"optimization":info,"evaluation":scores}
        print("PROBE_DONE",variant,json.dumps(scores,ensure_ascii=False),flush=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("D5_OBSERVABILITY_DONE",OUT,flush=True)


if __name__=="__main__":
    main()
