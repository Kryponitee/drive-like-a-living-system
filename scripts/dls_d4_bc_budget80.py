#!/usr/bin/env python3
"""D4 frozen-protocol A/B/C behavior cloning on identical dynamic MetaDrive data.

All variants see only 259D normal observation; privileged teacher states are never
passed to the models. Same optimizer/epochs/sequence length/validation selection.
"""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_longitudinal_motor import model_info,run_sequence

ROOT=Path("/mnt/datasets/lxb/dls")
SEQ=40
EPOCHS=80
BATCH=48


class DynamicTeacherData(Dataset):
    def __init__(self,split,data_id,mean=None,std=None):
        assert split in ("train","val")
        archive=ROOT/"demonstrations"/data_id/f"{split}.npz"
        manifest=json.loads((ROOT/"reports"/data_id/f"teacher-{split}.json").read_text())
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest["sha256"]
        with np.load(archive,allow_pickle=False) as data:
            obs=np.asarray(data["observations"],dtype=np.float32)
            acts=np.asarray(data["actions"],dtype=np.float32)
            offsets=np.asarray(data["episode_offsets"],dtype=np.int64)
            seeds=np.asarray(data["episode_seeds"],dtype=np.int32)
            modes=np.asarray(data["episode_modes"],dtype=np.int8)
        assert obs.shape[1]==259 and acts.shape==(len(obs),2)
        assert offsets[-1]==len(obs) and len(offsets)==len(seeds)+1==len(modes)+1
        assert np.isfinite(obs).all() and np.isfinite(acts).all()
        assert np.all((acts>=0)&(acts<=1))
        assert all(51000<=seed<51024 for seed in seeds) if split=="train" else all(61000<=seed<61008 for seed in seeds)
        if mean is None:
            assert split=="train"
            self.mean=obs.mean(axis=0,dtype=np.float64).astype(np.float32)
            self.std=np.maximum(obs.std(axis=0,dtype=np.float64).astype(np.float32),.01)
        else:
            assert split=="val" and mean is not None and std is not None
            self.mean=mean;self.std=std
        self.episodes=[]
        self.chunks=[]
        for idx,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
            a=int(a);b=int(b)
            assert b>a
            x=np.clip((obs[a:b]-self.mean)/self.std,-10.,10.).astype(np.float32)
            y=acts[a:b].copy()
            self.episodes.append((x,y))
            for start in range(0,len(y),SEQ):
                self.chunks.append((idx,start))
        self.num_steps=len(obs)
        self.num_episodes=len(seeds)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self,idx):
        i,start=self.chunks[idx]
        x,y=self.episodes[i]
        n=min(SEQ,len(y)-start)
        a=np.zeros((SEQ,259),dtype=np.float32)
        b=np.zeros((SEQ,2),dtype=np.float32)
        mask=np.zeros(SEQ,dtype=np.float32)
        a[:n]=x[start:start+n]
        b[:n]=y[start:start+n]
        mask[:n]=1
        return torch.from_numpy(a),torch.from_numpy(b),torch.from_numpy(mask)


def score(model,loader,device):
    model.eval()
    total=0.;steps=0.;mae=torch.zeros(2,device=device)
    with torch.no_grad():
        for x,y,m in loader:
            x=x.to(device);y=y.to(device);m=m.to(device)
            pred=run_sequence(model,x)
            loss=F.smooth_l1_loss(pred,y,beta=.1,reduction="none")
            total+=(loss*m[...,None]).sum().item()
            mae+=((pred-y).abs()*m[...,None]).sum(dim=(0,1))
            steps+=m.sum().item()
    assert steps>0
    return {"huber":total/(2*steps),
            "throttle_mae":float(mae[0].item()/steps),
            "brake_mae":float(mae[1].item()/steps)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("A","B","C"),required=True)
    p.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    p.add_argument("--data-id",required=True)
    p.add_argument("--gpu",type=int,default=0)
    a=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert a.data_id.startswith("d4-dynamic-follow-v1-")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    assert torch.cuda.is_available() and 0<=a.gpu<torch.cuda.device_count()
    torch.cuda.set_device(a.gpu)
    device=torch.device(f"cuda:{a.gpu}")
    train=DynamicTeacherData("train",a.data_id)
    val=DynamicTeacherData("val",a.data_id,train.mean,train.std)
    assert train.num_episodes==72 and val.num_episodes==24
    model,architecture=model_info(a.variant)
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-5)
    train_loader=DataLoader(train,batch_size=BATCH,shuffle=True,num_workers=0)
    val_loader=DataLoader(val,batch_size=BATCH,shuffle=False,num_workers=0)

    folder=ROOT/"checkpoints"/a.data_id/"budget80"
    report_dir=ROOT/"reports"/a.data_id/"budget80"
    file=folder/f"{a.variant}-seed{a.seed}.pt"
    report=report_dir/f"train-{a.variant}-seed{a.seed}.json"
    tb=ROOT/"tensorboard"/a.data_id/"budget80"/f"{a.variant}-seed{a.seed}"
    assert not file.exists() and not report.exists()
    folder.mkdir(parents=True,exist_ok=True)
    report_dir.mkdir(parents=True,exist_ok=True)
    writer=SummaryWriter(str(tb))
    started=time.monotonic();best=float("inf");best_epoch=-1;history=[]
    for epoch in range(1,EPOCHS+1):
        model.train();train_loss=0.;n=0.
        for x,y,mask in train_loader:
            x=x.to(device);y=y.to(device);mask=mask.to(device)
            pred=run_sequence(model,x)
            loss=F.smooth_l1_loss(pred,y,beta=.1,reduction="none")
            loss=(loss*mask[...,None]).sum()/(2*mask.sum())
            assert torch.isfinite(loss).item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            q=float(mask.sum().item());n+=q;train_loss+=float(loss.item())*q
        metrics=score(model,val_loader,device)
        train_loss/=n
        writer.add_scalar("d4/train_huber",train_loss,epoch)
        writer.add_scalar("d4/val_huber",metrics["huber"],epoch)
        writer.add_scalar("d4/val_brake_mae",metrics["brake_mae"],epoch)
        history.append({"epoch":epoch,"train_huber":train_loss,**metrics})
        if metrics["huber"]<best:
            best=metrics["huber"];best_epoch=epoch
            torch.save({
                "state_dict":model.state_dict(),"architecture":architecture,
                "variant":a.variant,"seed":a.seed,"data_id":a.data_id,
                "train_mean":train.mean,"train_std":train.std,
                "best_epoch":epoch,"val_metrics":metrics,
                "epochs":EPOCHS,"sequence_length":SEQ,"batch_size":BATCH,
                "action_interface":"independent absolute [T,B] in [0,1]^2",
                "privileged_teacher_only":True,
                "input_features":"standard MetaDrive 259D observation ONLY",
            },file)
        print(f"TRAIN {a.variant} seed={a.seed} epoch={epoch}/{EPOCHS} "
              f"train={train_loss:.6f} val={metrics['huber']:.6f} "
              f"brake_mae={metrics['brake_mae']:.5f}",flush=True)
    writer.close()
    assert best_epoch>0 and file.is_file()
    summary={
        "stage":"DLS D4 80-epoch optimization-budget extension, same training data and 259D inputs",
        "data_id":a.data_id,"variant":a.variant,"seed":a.seed,
        "train_episodes":train.num_episodes,"val_episodes":val.num_episodes,
        "train_steps":train.num_steps,"val_steps":val.num_steps,
        "epochs":EPOCHS,"batch_size":BATCH,"sequence_length":SEQ,
        "optimizer":"AdamW lr=6e-4 weight_decay=1e-5",
        "best_epoch":best_epoch,"best_val_huber":best,
        "checkpoint":str(file),"checkpoint_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
        "cuda_device":str(device),"wall_seconds":time.monotonic()-started,
        "history":history,
    }
    report.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print("TRAIN_DONE",json.dumps({k:v for k,v in summary.items() if k!="history"},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
