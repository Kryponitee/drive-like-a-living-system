#!/usr/bin/env python3
"""D2f: read-only original BC validation Huber after PPO anchoring, 9822 held-out steps."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from dls_p0 import model_info, nparams, run_sequence

ROOT=Path("/mnt/datasets/lxb/dls")
LEN=40

def make_chunks(mean,std):
    data=ROOT/"demonstrations"/"d1-v1"/"val"
    chunks=[]
    paths=sorted(data.glob("worker-*.json"))
    assert len(paths)==3
    for manifest_file in paths:
        m=json.loads(manifest_file.read_text())
        archive=Path(m["archive"])
        assert archive.parent==data
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==m["archive_sha256"]
        with np.load(archive,allow_pickle=False) as z:
            x=np.asarray(z["observations"],dtype=np.float32)
            y=np.asarray(z["actions"],dtype=np.float32)
            offsets=np.asarray(z["episode_offsets"],dtype=np.int64)
        assert y.shape==(len(x),2) and x.shape[1]==259
        for start,end in zip(offsets[:-1],offsets[1:]):
            for k in range(int(start),int(end),LEN):
                n=min(LEN,int(end)-k)
                normed=np.zeros((LEN,259),np.float32)
                targets=np.zeros((LEN,2),np.float32)
                mask=np.zeros((LEN,),np.float32)
                normed[:n]=np.clip((x[k:k+n]-mean)/std,-10,10)
                targets[:n]=y[k:k+n]
                mask[:n]=1.0
                chunks.append((normed,targets,mask))
    assert sum(int(m.sum()) for _,_,m in chunks)==9822
    return chunks

def score(variant,phase,chunks):
    ckdir="d1-v1" if phase=="bc" else ("d2d-v1" if phase=="d2d" else "d2f-v1")
    c=torch.load(ROOT/"checkpoints"/ckdir/f"{variant}-seed42.pt",map_location="cpu",weights_only=False)
    model,_=model_info(variant)
    model.load_state_dict(c["state_dict"] if phase=="bc" else c["actor_state_dict"],strict=True)
    model.eval()
    assert c["variant"]==variant and len(chunks)>100
    loss=0.; mae=np.zeros((2,),np.float64);steps=0
    with torch.no_grad():
        for i in range(0,len(chunks),16):
            block=chunks[i:i+16]
            x=torch.from_numpy(np.stack([row[0] for row in block]))
            y=torch.from_numpy(np.stack([row[1] for row in block]))
            m=torch.from_numpy(np.stack([row[2] for row in block]))
            pred=run_sequence(model,x)
            d=(pred-y).abs()
            loss+=float((F.smooth_l1_loss(pred,y,reduction="none",beta=0.1)*m.unsqueeze(-1)).sum())
            mae+=np.asarray((d*m.unsqueeze(-1)).sum(dim=(0,1)).numpy(),dtype=np.float64)
            steps+=int(m.sum())
    assert steps==9822
    return {"Huber":loss/(2*steps),"steer_MAE":float(mae[0]/steps),"pedal_MAE":float(mae[1]/steps)}

def main():
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    out={"stage":"D2f validation functional retention on 9822 independent D1 expert steps",
         "episodes_source":"D1 validation 2000..2047, not used in anchoring or PPO",
         "models":{}}
    for variant in "ABC":
        base=torch.load(ROOT/"checkpoints"/"d1-v1"/f"{variant}-seed42.pt",
                        map_location="cpu",weights_only=False)
        mean=np.asarray(base["train_mean"],np.float32)
        std=np.asarray(base["train_std"],np.float32)
        assert mean.shape==std.shape==(259,) and float(std.min())>0
        for phase,folder in (("d2d","d2d-v1"),("d2f","d2f-v1")):
            c=torch.load(ROOT/"checkpoints"/folder/f"{variant}-seed42.pt",
                         map_location="cpu",weights_only=False)
            assert np.array_equal(c["train_mean"],mean) and np.array_equal(c["train_std"],std)
        chunks=make_chunks(mean,std)
        out["models"][variant]={phase:score(variant,phase,chunks) for phase in ("bc","d2d","d2f")}
        print(variant,[(p,round(out["models"][variant][p]["Huber"],6)) for p in ("bc","d2d","d2f")],flush=True)
    path=ROOT/"reports"/"d2f"/"d2f-validation-drift.json"
    assert not path.exists()
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved: {path}",flush=True)

if __name__=="__main__":
    main()
