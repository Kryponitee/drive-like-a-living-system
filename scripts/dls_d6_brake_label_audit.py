#!/usr/bin/env python3
"""DLS D6 pre-objective-change audit: quantify the imbalanced critical brake labels.

Read-only from immutable physical demonstration archives. Distinguishes the
fraction of severe supervised events from actual controller safety outcomes.
"""
import hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path("/mnt/datasets/lxb/dls")
ID="d4-dynamic-follow-v1-44481405397"
BASE=ROOT/"reports"/ID
OUT=BASE/"d6-a-pilot"/"braking-target-frequency.json"
MODES=("pulse","mild","sham")

def inspect(actions,modes,sources=None):
    assert actions.ndim==2 and actions.shape[1]==2
    y=actions[:,1]
    x=actions[:,0]
    result={
        "num_steps":len(actions),
        "brake_positive_fraction":float(np.mean(y>0.1)),
        "brake_over_0_3_fraction":float(np.mean(y>0.3)),
        "brake_over_0_5_fraction":float(np.mean(y>0.5)),
        "brake_over_0_8_fraction":float(np.mean(y>0.8)),
        "throttle_brake_positive_fraction":float(np.mean((x>.1)&(y>.1))),
        "brake_mean":float(y.mean()),
        "brake_p95":float(np.percentile(y,95)),
        "brake_p99":float(np.percentile(y,99)),
    }
    return result

def main():
    assert os.path.ismount("/mnt/datasets") and not OUT.exists()
    manifest=json.loads((BASE/"teacher-train.json").read_text())
    train=ROOT/"demonstrations"/ID/"train.npz"
    assert hashlib.sha256(train.read_bytes()).hexdigest()==manifest["sha256"]
    doc=json.loads((BASE/"d6-online-coverage-v0"/"teacher-labelled-sample-metadata.json").read_text())
    online=ROOT/"demonstrations"/ID/"d6-online-coverage-v0.npz"
    assert hashlib.sha256(online.read_bytes()).hexdigest()==doc["archive_sha256"]
    groups={}
    for name,path in (("original_teacher",train),("student_visited",online)):
        with np.load(path,allow_pickle=False) as d:
            actions=d["actions"].copy()
            offsets=d["episode_offsets"].copy()
            modes=d["episode_modes"].copy()
            sources=d["source_policy"].copy() if name=="student_visited" else None
        records=[]
        for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
            records.append((actions[a:b],int(modes[i]),int(sources[i]) if sources is not None else -1))
        group={"overall":inspect(actions,modes)}
        for k,mode in enumerate(MODES):
            sub=np.concatenate([r[0] for r in records if r[1]==k])
            group[mode]=inspect(sub,None)
        if sources is not None:
            group["source_policy"]={
                key:inspect(np.concatenate([r[0] for r in records if r[2]==i]),None)
                for i,key in enumerate("ABC")
            }
        groups[name]=group
    out={
        "stage":"D6 critical action density diagnostic, prior to adjusting objective",
        "data_id":ID,
        "source":"immutable original physical teacher vs new policy-visited states labelled by same teacher",
        "groups":groups,
        "conclusion_scope":"only label prevalence and severity, not an action learning mechanism or closed-loop proof",
    }
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print("D6_BRAKE_FREQUENCY",json.dumps(out,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
