#!/usr/bin/env python3
"""D2g: frozen critic predictions versus actual discounted future return, no training."""

import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from metadrive.envs.metadrive_env import MetaDriveEnv
from dls_p0 import model_info, nparams
from dls_recurrent_ppo import RecurrentSquashedPolicy, IdenticalCritic


ROOT=Path("/mnt/datasets/lxb/dls")
GAMMA=0.99

def correlate(a,b):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    assert len(a)==len(b) and len(a)>=2
    if np.std(a)<1e-12 or np.std(b)<1e-12:
        return None
    return float(np.corrcoef(a,b)[0,1])

def normalize(obs,mean,std):
    x=np.asarray(obs,dtype=np.float32)
    assert x.shape==(259,) and np.isfinite(x).all()
    return torch.from_numpy(np.clip((x-mean)/std,-10.0,10.0).copy()).unsqueeze(0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("A","B","C"),required=True)
    p.add_argument("--phase",choices=("d2d","d2f"),required=True)
    a=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)

    ckpt=ROOT/"checkpoints"/f"{a.phase}-v1"/f"{a.variant}-seed42.pt"
    c=torch.load(ckpt,map_location="cpu",weights_only=False)
    assert c["variant"]==a.variant and c["train_density"]==0.1
    assert c["training_environment_steps"]==4096
    model,_=model_info(a.variant)
    model.load_state_dict(c["actor_state_dict"],strict=True)
    assert nparams(model)==c["num_actor_params"]
    policy=RecurrentSquashedPolicy(model)
    with torch.no_grad():
        policy.log_std.copy_(c["log_std"])
    policy.eval()
    critic=IdenticalCritic()
    critic.load_state_dict(c["critic_state_dict"],strict=True)
    critic.eval()
    mean=np.asarray(c["train_mean"],np.float32)
    std=np.asarray(c["train_std"],np.float32)
    assert mean.shape==std.shape==(259,) and float(std.min())>0
    sigma=policy.log_std.exp().detach().numpy().tolist()
    assert len(sigma)==2 and all(0.01<q<1 for q in sigma)

    env=MetaDriveEnv({
        "use_render":False,"image_observation":False,
        "traffic_density":0.1,"map":3,
        "start_seed":4500,"num_scenarios":24,
        "horizon":800,"log_level":50,
    })
    episodes=[]
    V_all=[]
    G_all=[]
    TD_all=[]
    t0=time.monotonic()
    with torch.no_grad():
        for seed in range(4500,4524):
            obs,_=env.reset(seed=seed)
            state=policy.initial_state(1,torch.device("cpu"))
            rng=torch.Generator(device="cpu").manual_seed(500000+seed)
            values=[]
            rewards=[]
            action_abs_dev=[]
            last_info={}
            end_terminated=False
            end_truncated=False
            next_obs=obs
            for t in range(800):
                x=normalize(obs,mean,std)
                v=float(critic(x)[0].item())
                loc,state=policy.step_parameters(x,state,t)
                normal=torch.randn((1,2),generator=rng,dtype=loc.dtype)
                act=torch.tanh(loc+policy.log_std.exp()*normal)
                mean_action=torch.tanh(loc)
                action_abs_dev.append(float((act-mean_action).abs().mean()))
                next_obs,reward,terminated,truncated,last_info=env.step(
                    act[0].numpy().astype(np.float32)
                )
                values.append(v)
                rewards.append(float(reward))
                end_terminated=bool(terminated)
                end_truncated=bool(truncated)
                obs=next_obs
                if terminated or truncated:
                    break
            # Artificial 800-step cap counted as time limit and bootstrap enabled.
            timeout=(not end_terminated) and (end_truncated or t==799)
            bootstrap=float(critic(normalize(next_obs,mean,std))[0].item()) if timeout else 0.
            returns=np.empty(len(rewards),np.float64)
            target=bootstrap
            for k in reversed(range(len(rewards))):
                target=rewards[k]+GAMMA*target
                returns[k]=target
            v_arr=np.asarray(values,np.float64)
            r_arr=np.asarray(rewards,np.float64)
            # One-step TD residual against the critic's own next-state value.
            next_values=np.concatenate([v_arr[1:],np.asarray([bootstrap],np.float64)])
            td=r_arr+GAMMA*next_values-v_arr
            assert len(v_arr)==len(returns)==len(td) and np.isfinite(returns).all()
            V_all.append(v_arr)
            G_all.append(returns)
            TD_all.append(td)
            episodes.append({
                "scene_seed":seed,"steps":len(rewards),
                "arrive_dest":bool(last_info.get("arrive_dest",False)),
                "crash_vehicle":bool(last_info.get("crash_vehicle",False)),
                "off_road":bool(last_info.get("out_of_road",False)),
                "time_limit_bootstrap":timeout,
                "first_value":float(v_arr[0]),"first_discounted_return":float(returns[0]),
                "final_bootstrap":float(bootstrap),
                "mean_abs_TD":float(np.mean(np.abs(td))),
                "mean_abs_action_deviation":float(np.mean(action_abs_dev)),
            })
    env.close()
    value=np.concatenate(V_all)
    returns=np.concatenate(G_all)
    td=np.concatenate(TD_all)
    assert len(value)==sum(e["steps"] for e in episodes)
    assert len(episodes)==24
    variance=float(np.var(returns))
    assert variance>1e-8
    explained_variance=1.0-float(np.var(returns-value))/variance
    result={
        "stage":"D2g frozen actor+critic calibration, no training",
        "variant":a.variant,"phase":a.phase,
        "model_seed":42,"training_steps":4096,
        "gamma":GAMMA,"density":0.1,
        "scenarios":list(range(4500,4524)),
        "pre_tanh_sigma_final":sigma,
        "steps":int(len(value)),
        "successes":sum(e["arrive_dest"] for e in episodes),
        "time_limit_bootstrap_count":sum(e["time_limit_bootstrap"] for e in episodes),
        "critic_value_return_pearson_over_transitions":correlate(value,returns),
        "critic_explained_variance_over_transitions":float(explained_variance),
        "critic_value_return_MAE":float(np.mean(np.abs(returns-value))),
        "critic_value_return_RMSE":float(np.sqrt(np.mean((returns-value)**2))),
        "mean_absolute_one_step_TD_residual":float(np.mean(np.abs(td))),
        "critic_value_mean":float(np.mean(value)),
        "critic_value_std":float(np.std(value)),
        "realized_return_mean":float(np.mean(returns)),
        "realized_return_std":float(np.std(returns)),
        "first_state_value_return_pearson_over_episodes":correlate(
            [e["first_value"] for e in episodes],
            [e["first_discounted_return"] for e in episodes]),
        "episodes":episodes,
        "no_training":True,
        "wall_seconds":time.monotonic()-t0,
        "interpretation_limits":"Single sampled stochastic future per state, same episode steps correlated, different policies occupy different states. Correlation alone is not a causal advantage-gradient test.",
    }
    out=ROOT/"reports"/"d2g"/f"critic-{a.phase}-{a.variant}-seed42.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"{a.variant}/{a.phase} steps={len(value)} success={result['successes']}/24 "
          f"EV={explained_variance:.4f} Pearson={result['critic_value_return_pearson_over_transitions']} "
          f"MAE={result['critic_value_return_MAE']:.3f} "
          f"Vstd={result['critic_value_std']:.3f} Gstd={result['realized_return_std']:.3f}",flush=True)

if __name__=="__main__":
    main()
