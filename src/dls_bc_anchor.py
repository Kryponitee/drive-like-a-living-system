"""D2f: reusable, independent BC-space functional anchor batches.

Uses existing D1 expert observations only. No new simulator or expert data.
Episode chunks start at each 40-step BC TBPTT boundary; zero padding is masked.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

class FrozenBCAnchorBatches:
    def __init__(self, root:Path, mean:np.ndarray, std:np.ndarray, seed:int, length:int=40):
        self.rng=np.random.default_rng(seed)
        self.length=length
        self.mean=np.asarray(mean,dtype=np.float32)
        self.std=np.asarray(std,dtype=np.float32)
        assert self.mean.shape==self.std.shape==(259,)
        assert np.all(self.std>0)
        data_dir=root/"demonstrations"/"d1-v1"/"train"
        manifest_paths=sorted(data_dir.glob("worker-*.json"))
        assert len(manifest_paths)==24, f"Need the 24 verified D1 shards, found {len(manifest_paths)}"
        self.episodes=[]
        self.chunks=[]
        self.episode_seeds=set()
        for manifest_path in manifest_paths:
            m=json.loads(manifest_path.read_text())
            assert m["split"]=="train"
            archive=Path(m["archive"])
            assert archive.parent==data_dir and archive.exists()
            assert hashlib.sha256(archive.read_bytes()).hexdigest()==m["archive_sha256"]
            with np.load(archive,allow_pickle=False) as data:
                xs=np.asarray(data["observations"],dtype=np.float32)
                ends=np.asarray(data["episode_offsets"],dtype=np.int64)
                seeds=np.asarray(data["episode_seeds"],dtype=np.int32)
            assert xs.ndim==2 and xs.shape[1]==259
            assert ends.ndim==1 and ends[0]==0 and ends[-1]==len(xs)
            assert len(ends)==len(seeds)+1
            assert np.isfinite(xs).all()
            for start,end,episode_seed in zip(ends[:-1],ends[1:],seeds):
                assert 1000<=int(episode_seed)<1600 and int(episode_seed) not in self.episode_seeds
                self.episode_seeds.add(int(episode_seed))
                assert end>start
                ep=np.clip((xs[int(start):int(end)]-self.mean)/self.std,-10.0,10.0).astype(np.float32)
                epi=len(self.episodes)
                self.episodes.append(ep)
                for i in range(0,len(ep),length):
                    self.chunks.append((epi,i))
        assert len(self.episodes)>100 and len(self.chunks)>1000
        self.steps=sum(len(e) for e in self.episodes)
        assert self.steps==91717

    def sample(self,batch_size:int,device):
        assert batch_size>=1
        indices=self.rng.integers(0,len(self.chunks),size=batch_size)
        x=np.zeros((batch_size,self.length,259),dtype=np.float32)
        mask=np.zeros((batch_size,self.length),dtype=np.float32)
        for i,ind in enumerate(indices):
            ep_i,start=self.chunks[int(ind)]
            episode=self.episodes[ep_i]
            n=min(self.length,len(episode)-start)
            x[i,:n]=episode[start:start+n]
            mask[i,:n]=1.0
        assert mask.sum()>0
        return torch.from_numpy(x).to(device),torch.from_numpy(mask).to(device)
