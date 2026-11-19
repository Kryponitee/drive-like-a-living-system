#!/usr/bin/env python3
"""D7 A-only strong-brake target weighting: same D6 learner-visited states.

The two arms share architecture, initialization, epochs, number of episodes,
batch size, optimizer and validation. Only NEW visited-state samples differ.
This is a capacity/coverage feasibility experiment, not motor grouping evidence.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_longitudinal_motor import model_info,run_sequence

ROOT=Path("/mnt/datasets/lxb/dls")
DATA_ID="d4-dynamic-follow-v1-44481405397"
N_EPOCH=80
BATCH=16
MAX_STEPS=210


def fetch_episodes(archive,expected_sha):
    assert archive.exists()
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected_sha
    with np.load(archive,allow_pickle=False) as data:
        x=np.asarray(data["observations"],dtype=np.float32)
        y=np.asarray(data["actions"],dtype=np.float32)
        offsets=np.asarray(data["episode_offsets"],dtype=np.int64)
    assert x.shape==(len(y),259) and y.shape[1]==2
    assert offsets[0]==0 and offsets[-1]==len(y)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.all((y>=0)&(y<=1))
    return [(x[a:b].copy(),y[a:b].copy()) for a,b in zip(offsets[:-1],offsets[1:])]


class Episodes(Dataset):
    def __init__(self,raw,mean,std):
        self.rows=[]
        for x,y in raw:
            assert 0<len(x)==len(y)<=MAX_STEPS
            self.rows.append((
                torch.from_numpy(np.clip((x-mean)/std,-10.,10.).astype(np.float32)),
                torch.from_numpy(y.copy()),
            ))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]


def collate(batch):
    n=len(batch);steps=max(len(y) for _,y in batch)
    x=torch.zeros((n,steps,259),dtype=torch.float32)
    y=torch.zeros((n,steps,2),dtype=torch.float32)
    mask=torch.zeros((n,steps),dtype=torch.float32)
    for idx,(xx,yy) in enumerate(batch):
        t=len(yy);x[idx,:t]=xx;y[idx,:t]=yy;mask[idx,:t]=1
    return x,y,mask


def validate(model,loader,device):
    model.eval();total=0.;count=0.
    with torch.no_grad():
        for x,y,mask in loader:
            x=x.to(device);y=y.to(device);mask=mask.to(device)
            pred=run_sequence(model,x)
            error=F.smooth_l1_loss(pred,y,beta=.1,reduction="none")
            total+=float((error*mask[...,None]).sum().item())
            count+=float(mask.sum().item())
    assert count>0
    return total/(2*count)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--arm",choices=("riskweighted",),required=True)
    parser.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    parser.add_argument("--gpu",type=int,required=True)
    a=parser.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    assert torch.cuda.is_available() and 0<=a.gpu<torch.cuda.device_count()
    device=torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(a.gpu)
    ref=ROOT/"checkpoints"/DATA_ID/"full-seq80"/"A-seed42.pt"
    weights=torch.load(ref,map_location="cpu",weights_only=False)
    mean=np.asarray(weights["train_mean"],dtype=np.float32)
    std=np.asarray(weights["train_std"],dtype=np.float32)
    assert mean.shape==std.shape==(259,) and (std>0).all()
    reports=ROOT/"reports"/DATA_ID
    orig_train=json.loads((reports/"teacher-train.json").read_text())
    orig_val=json.loads((reports/"teacher-val.json").read_text())
    policy_info=json.loads((reports/"d6-online-coverage-v0"/"teacher-labelled-sample-metadata.json").read_text())
    prior=fetch_episodes(ROOT/"demonstrations"/DATA_ID/"train.npz",orig_train["sha256"])
    validation=fetch_episodes(ROOT/"demonstrations"/DATA_ID/"val.npz",orig_val["sha256"])
    online=fetch_episodes(ROOT/"demonstrations"/DATA_ID/"d6-online-coverage-v0.npz",policy_info["archive_sha256"])
    assert len(prior)==72 and len(validation)==24 and len(online)==108
    rng=np.random.default_rng(17100)
    repetition=[prior[int(i)] for i in np.concatenate([
        np.arange(len(prior)),rng.permutation(len(prior))[:36]])]
    assert len(repetition)==108
    raw=prior+online
    assert len(raw)==180
    train=Episodes(raw,mean,std)
    val=Episodes(validation,mean,std)
    train_loader=DataLoader(train,batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=0)
    val_loader=DataLoader(val,batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=0)
    model,architecture=model_info("A")
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-5)
    outdir=ROOT/"checkpoints"/DATA_ID/"d7-a-riskweight"
    reportdir=reports/"d7-a-riskweight"
    ckpt=outdir/f"{a.arm}-seed{a.seed}.pt"
    result=reportdir/f"train-{a.arm}-seed{a.seed}.json"
    assert not ckpt.exists() and not result.exists()
    outdir.mkdir(parents=True,exist_ok=True)
    reportdir.mkdir(parents=True,exist_ok=True)
    writer=SummaryWriter(str(ROOT/"tensorboard"/DATA_ID/"d7-a-riskweight"/f"{a.arm}-seed{a.seed}"))
    best=float("inf");best_epoch=-1;history=[]
    start=time.monotonic()
    for epoch in range(1,N_EPOCH+1):
        model.train();loss_sum=0.;nstep=0.
        for x,y,m in train_loader:
            x=x.to(device);y=y.to(device);m=m.to(device)
            output=run_sequence(model,x)
            raw_loss=F.smooth_l1_loss(output,y,beta=.1,reduction="none")
            weight=1.+6.*(y[...,1]>0.5).float()
            wm=m*weight
            loss=(raw_loss*wm[...,None]).sum()/(2*wm.sum())
            assert torch.isfinite(loss).item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            samples=float(m.sum().item())
            loss_sum+=float(loss.item())*samples
            nstep+=samples
        v=validate(model,val_loader,device)
        train_val=loss_sum/nstep
        writer.add_scalar("train/huber",train_val,epoch)
        writer.add_scalar("val/huber",v,epoch)
        history.append((epoch,train_val,v))
        if v<best:
            best=v;best_epoch=epoch
            torch.save({
                "state_dict":model.state_dict(),"variant":"A","seed":a.seed,
                "data_id":DATA_ID,"arm":a.arm,
                "train_mean":mean,"train_std":std,
                "epoch":epoch,"best_val_huber":v,
                "action_interface":"independent physical throttle and brake",
                "episode_count":len(raw),"epochs":N_EPOCH,
                "student_input":"MetaDrive 259D only",
                "teacher_extra_physics":"used ONLY for labels",
            },ckpt)
        if epoch%20==0:
            print("D6_TRAIN",a.arm,a.seed,epoch,
                  "train",train_val,"val",v,"best",best,flush=True)
    writer.close()
    report={
        "stage":"D7 A-only strong-brake weighted supervision on identical D6 on-policy data",
        "arm":a.arm,"seed":a.seed,
        "train_episodes":len(raw),"val_episodes":len(validation),
        "critical_brake_target_threshold":0.5,"critical_brake_weight":7.0,
        "loss_only_change_from_D6_onpolicy":True,
        "train_time_steps":sum(len(x) for x,y in raw),
        "val_time_steps":sum(len(x) for x,y in validation),
        "optimizer_updates_per_epoch":len(train_loader),"batch_size":BATCH,
        "epochs":N_EPOCH,"initialization_seed":a.seed,
        "best_epoch":best_epoch,"best_val_huber":best,
        "checkpoint":str(ckpt),
        "checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "source_learner_data_seeds":[52000,52011],
        "original_teacher_data_seeds":[51000,51023],
        "validation_teacher_seeds":[61000,61007],
        "test_scenes_not_accessed_in_training":True,
        "wall_seconds":time.monotonic()-start,
        "history":history,
    }
    result.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("D6_TRAIN_COMPLETE",json.dumps({
        k:v for k,v in report.items() if k!="history"},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
