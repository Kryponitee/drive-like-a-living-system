#!/usr/bin/env python3
"""D2e read-only audit: reward/success alignment and teacher-forced imitation drift.

No simulator, no environment downloads, no optimizer updates, no checkpoint mutations.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from dls_p0 import model_info, run_sequence

ROOT=Path("/mnt/datasets/lxb/dls")
OUT=ROOT/"reports"/"d2e"
TRAIN=ROOT/"reports"/"d2b"
LOW=ROOT/"reports"/"d2d"
SEQ=40


def load_eval(phase, variant, density):
    f=LOW/f"eval-{phase}-{variant}-density{density:.2f}.json"
    x=json.loads(f.read_text())
    assert x["seeds"]==list(range(4300,4312)) and x["traffic_density"]==density
    assert len(x["episodes"])==12
    return x


def validation_sequences(mean, std):
    root=ROOT/"demonstrations"/"d1-v1"/"val"
    chunks=[]
    paths=sorted(root.glob("worker-*.json"))
    assert len(paths)==3
    for path in paths:
        manifest=json.loads(path.read_text())
        archive=Path(manifest["archive"])
        assert archive.parent==root and hashlib.sha256(archive.read_bytes()).hexdigest()==manifest["archive_sha256"]
        with np.load(archive,allow_pickle=False) as z:
            xs=np.asarray(z["observations"],dtype=np.float32)
            ys=np.asarray(z["actions"],dtype=np.float32)
            offsets=np.asarray(z["episode_offsets"],dtype=np.int64)
        assert xs.shape[1]==259 and ys.shape==(len(xs),2)
        for start,end in zip(offsets[:-1],offsets[1:]):
            for j in range(int(start),int(end),SEQ):
                n=min(SEQ,int(end)-j)
                xx=np.zeros((SEQ,259),dtype=np.float32)
                yy=np.zeros((SEQ,2),dtype=np.float32)
                mm=np.zeros((SEQ,),dtype=np.float32)
                xx[:n]=np.clip((xs[j:j+n]-mean)/std,-10,10)
                yy[:n]=ys[j:j+n]
                mm[:n]=1
                chunks.append((xx,yy,mm))
    assert sum(int(mask.sum()) for _,_,mask in chunks)==9822
    return chunks


def actor_from_checkpoint(variant,phase):
    if phase=="bc":
        path=ROOT/"checkpoints"/"d1-v1"/f"{variant}-seed42.pt"
        key="state_dict"
    else:
        path=ROOT/"checkpoints"/("d2b-v1" if phase=="d2b" else "d2d-v1")/f"{variant}-seed42.pt"
        key="actor_state_dict"
    data=torch.load(path,map_location="cpu",weights_only=False)
    assert data["variant"]==variant
    model,_=model_info(variant)
    model.load_state_dict(data[key],strict=True)
    model.eval()
    return model,data


def eval_val(model,chunks):
    total_loss=0.0
    total_mae=np.zeros((2,),dtype=np.float64)
    total_steps=0
    with torch.no_grad():
        for i in range(0,len(chunks),16):
            buf=chunks[i:i+16]
            x=torch.from_numpy(np.stack([a for a,_,_ in buf]))
            y=torch.from_numpy(np.stack([b for _,b,_ in buf]))
            m=torch.from_numpy(np.stack([mm for _,_,mm in buf]))
            pred=run_sequence(model,x)
            loss=F.smooth_l1_loss(pred,y,reduction="none",beta=0.1)
            total_loss+=float((loss*m.unsqueeze(-1)).sum())
            total_mae+=np.asarray(((pred-y).abs()*m.unsqueeze(-1)).sum(dim=(0,1)).numpy(),np.float64)
            total_steps+=int(m.sum())
    assert total_steps==9822
    return {"huber":total_loss/(total_steps*2),"steer_mae":float(total_mae[0]/total_steps),
            "pedal_mae":float(total_mae[1]/total_steps)}


def main():
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    start=time.monotonic()
    results={
        "stage":"D2e read-only objective-alignment audit; not a new scientific experiment",
        "new_training":False, "new_simulation":False,
        "paired_scenarios":list(range(4300,4312)),
        "BC_validation_steps":9822,
        "models":{},
    }
    for v in "ABC":
        models={}
        states={}
        for phase in ("bc","d2b","d2d"):
            models[phase],states[phase]=actor_from_checkpoint(v,phase)
        mean=np.asarray(states["bc"]["train_mean"],dtype=np.float32)
        std=np.asarray(states["bc"]["train_std"],dtype=np.float32)
        chunks=validation_sequences(mean,std)
        imitation={}
        for phase,model in models.items():
            imitation[phase]=eval_val(model,chunks)
            assert np.array_equal(np.asarray(states[phase]["train_mean"]),mean)
            assert np.array_equal(np.asarray(states[phase]["train_std"]),std)
        groups={}
        for d in (0.0,0.1,0.25):
            raw={phase:load_eval(phase,v,d) for phase in ("bc","d2b","d2d")}
            for phase in ("d2b","d2d"):
                assert [x["seed"] for x in raw["bc"]["episodes"]] == [x["seed"] for x in raw[phase]["episodes"]]
            groups[str(d)]={}
            baseline=raw["bc"]
            for phase,r in raw.items():
                group={
                    "success":r["success"],
                    "mean_reward":r["mean_reward"],
                    "mean_completion":r["mean_completion"],
                    "mean_reward_change_from_BC":r["mean_reward"]-baseline["mean_reward"],
                    "success_count_change_from_BC":r["success"]-baseline["success"],
                    "completion_change_from_BC":r["mean_completion"]-baseline["mean_completion"],
                }
                if phase!="bc":
                    group["num_reward_increase_success_loss"]=sum(
                        b["success"] and not x["success"] and x["reward"] > b["reward"]
                        for b,x in zip(baseline["episodes"],r["episodes"])
                    )
                groups[str(d)][phase]=group
        train={}
        for phase,root in (("d2b",TRAIN),("d2d",LOW)):
            t=json.loads((root/f"train-{v}-seed42.json").read_text())
            eps=t["completed_episodes"]
            assert t["environment_steps"]==4096
            train[phase]={
                "completed_episodes":len(eps),
                "successful_episodes":sum(e["success"] for e in eps),
                "mean_train_episode_reward":float(np.mean([e["reward"] for e in eps])),
                "mean_train_episode_length":float(np.mean([e["steps"] for e in eps])),
                "max_preupdate_ratio_error":t["max_ppo_behavior_replay_error"],
                "avg_logged_approx_kl":t["mean_approx_kl_last_epoch"],
            }
        results["models"][v]={"train":train,"validation_imitation":imitation,"closed_loop":groups}
        print(v,"training success",train["d2b"]["successful_episodes"],"->",train["d2d"]["successful_episodes"])
        for d,phase_data in groups.items():
            print(v,d,"BC/old/new reward",*[round(phase_data[p]["mean_reward"],2) for p in ("bc","d2b","d2d")],
                  "success",*[phase_data[p]["success"] for p in ("bc","d2b","d2d")])
        print(v,"val Huber BC/old/new",*[round(imitation[p]["huber"],6) for p in ("bc","d2b","d2d")])

    OUT.mkdir(parents=True,exist_ok=True)
    out=OUT/"d2e-objective-diagnostic.json"
    assert not out.exists(),"Refuse overwrite"
    results["analysis_elapsed_seconds"]=time.monotonic()-start
    out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved {out}",flush=True)


if __name__=="__main__":
    main()
