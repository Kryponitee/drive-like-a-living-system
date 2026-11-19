#!/usr/bin/env python3
"""DLS D5 follow-up: matched state-coverage diagnostic and on-policy sensor probe.

This is a privileged diagnostic estimator, NOT a replacement policy. Train,
validation and test groups use disjoint physical scene seeds across A/B/C.
"""
import json,os
from pathlib import Path
import numpy as np
import torch

from dls_d5_sensor_probe import (
    BASE,OUT,load,train_probe,metric)

RESULT=BASE/"onpolicy-decoding-shift-diagnostic.json"
VARIANTS=("A","B","C")
SPLITS={
    "train":set(range(71000,71006)),
    "val":set(range(71006,71008)),
    "test":set(range(71008,71012)),
}


def repack(source,indices):
    fields=("observations","labels","times","event_steps","ego_speeds")
    dest={k:[] for k in fields}
    offsets=[0];seeds=[];modes=[]
    for i in indices:
        a,b=map(int,source["episode_offsets"][i:i+2])
        assert b>a
        for field in fields:dest[field].append(source[field][a:b])
        offsets.append(offsets[-1]+b-a)
        seeds.append(int(source["episode_seeds"][i]))
        modes.append(int(source["episode_modes"][i]))
    assert len(seeds)>0
    result={name:np.concatenate(arr,axis=0) for name,arr in dest.items()}
    result.update({
        "episode_offsets":np.asarray(offsets,dtype=np.int64),
        "episode_seeds":np.asarray(seeds,dtype=np.int32),
        "episode_modes":np.asarray(modes,dtype=np.int8),
    })
    return result


def combine(datasets,allowed):
    parts=[]
    for variant in VARIANTS:
        dataset=datasets[variant]
        idx=[i for i,x in enumerate(dataset["episode_seeds"]) if int(x) in allowed]
        assert len(idx)==len(allowed)
        subset=repack(dataset,idx)
        assert all(m==0 for m in subset["episode_modes"]) # pulse only
        parts.append(subset)
    fields=("observations","labels","times","event_steps","ego_speeds")
    full={field:np.concatenate([r[field] for r in parts],axis=0) for field in fields}
    offsets=[0];seeds=[];modes=[]
    for r in parts:
        for a,b in zip(r["episode_offsets"][:-1],r["episode_offsets"][1:]):
            offsets.append(offsets[-1]+int(b-a))
        seeds.extend(r["episode_seeds"].tolist())
        modes.extend(r["episode_modes"].tolist())
    full["episode_offsets"]=np.asarray(offsets,dtype=np.int64)
    full["episode_seeds"]=np.asarray(seeds,dtype=np.int32)
    full["episode_modes"]=np.asarray(modes,dtype=np.int8)
    assert offsets[-1]==len(full["labels"])
    return full


def describe(data):
    label=data["labels"];delta=data["times"]-data["event_steps"]
    groups={
        "all":np.ones(len(label),dtype=bool),
        "pre_event_15":(delta>=-15)&(delta<0),
        "post_event_0_15":(delta>=0)&(delta<15),
        "post_event_15_30":(delta>=15)&(delta<30),
    }
    out={}
    for k,mask in groups.items():
        if not mask.any():continue
        values=label[mask]
        out[k]={
            "samples":len(values),
            "gap_quantiles_m":np.percentile(values[:,0],[0,5,50,95,100]).tolist(),
            "closing_speed_quantiles_mps":np.percentile(values[:,1],[0,5,50,95,100]).tolist(),
            "closing_gt_1_mps_fraction":float(np.mean(values[:,1]>1)),
            "closing_gt_2_mps_fraction":float(np.mean(values[:,1]>2)),
        }
    return out


def main():
    assert os.path.ismount("/mnt/datasets")
    assert not RESULT.exists()
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    datasets={variant:load(f"onpolicy-{variant}") for variant in VARIANTS}
    train=combine(datasets,SPLITS["train"])
    val=combine(datasets,SPLITS["val"])
    test=combine(datasets,SPLITS["test"])
    expert=load("train")
    mean=train["observations"].mean(axis=0,dtype=np.float64).astype(np.float32)
    std=np.maximum(train["observations"].std(axis=0,dtype=np.float64).astype(np.float32),.01)
    ymean=train["labels"].mean(axis=0).astype(np.float32)
    ystd=np.maximum(train["labels"].std(axis=0).astype(np.float32),.5)
    device=torch.device("cuda:0")
    result={
        "stage":"D5 privileged state probe retrained on out-of-policy emergency regions",
        "data_id":"d4-dynamic-follow-v1-44481405397",
        "protocol":"train scene 71000..71005 / val 71006..71007 / test 71008..71011; all A/B/C seed42 pulse physical episodes",
        "teacher_train_episodes":len(expert["episode_seeds"]),
        "onpolicy_probe_train_episodes":len(train["episode_seeds"]),
        "onpolicy_probe_val_episodes":len(val["episode_seeds"]),
        "onpolicy_probe_test_episodes":len(test["episode_seeds"]),
        "samples_train_val_test":[len(train["labels"]),len(val["labels"]),len(test["labels"])],
        "privileged_labels_are_diagnostic_only":True,
        "distributions":{
            "teacher_train":describe(expert),
            "onpolicy_train":describe(train),
            "onpolicy_test":describe(test),
        },
        "probes":{},
        "limits":"On-policy probe is trained with hidden physical labels for diagnostic decoding only; it does NOT represent a policy that can use those labels or assure closed-loop safety.",
    }
    for variant in ("time_ego","spatial","temporal"):
        model,info=train_probe(variant,train,val,mean,std,ymean,ystd,device)
        score=metric(model,test,variant,mean,std,ymean,ystd,device)
        result["probes"][variant]={"training":info,"heldout_onpolicy_test":score}
        print("D5_ONPOLICY_PROBE",variant,json.dumps(score,ensure_ascii=False),flush=True)
    RESULT.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("D5_ONPOLICY_DIAGNOSTIC_DONE",RESULT,flush=True)


if __name__=="__main__":
    main()
