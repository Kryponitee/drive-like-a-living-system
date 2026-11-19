#!/usr/bin/env python3
"""DLS受控前车急刹中各模块logit变化是否相互抵消：冻结轨迹只读审计。"""
import json
from pathlib import Path

import numpy as np

ROOT=Path("/mnt/datasets/lxb/dls/reports/d1-longitudinal-v1-44117653285/dynamic-follow-v2")
OUT=ROOT/"readout-cancellation-audit.json"


def main():
    assert not OUT.exists()
    results={}
    for variant in ("B","C"):
        blocks=[]
        for seed in (42,43,44):
            run=json.loads((ROOT/f"{variant}-seed{seed}.json").read_text())
            for scene in (41001,41002):
                sham=next(r for r in run["results"] if r["scene_seed"]==scene and r["condition"]=="sham")
                pulse=next(r for r in run["results"] if r["scene_seed"]==scene and r["condition"]=="pulse")
                assert min(sham["steps"],pulse["steps"])>=100
                shifts=[]
                action_diffs=[]
                for t in range(90,100):
                    z0=sham["steps_trace"][t]
                    z1=pulse["steps_trace"][t]
                    c0=np.asarray(z0["unit_logit_contributions"],dtype=np.float64)
                    c1=np.asarray(z1["unit_logit_contributions"],dtype=np.float64)
                    assert c0.shape==c1.shape==(8,2)
                    shift=c1-c0
                    actions0=np.asarray([z0["throttle"],z0["brake"]])
                    actions1=np.asarray([z1["throttle"],z1["brake"]])
                    for a,c in ((actions0,c0),(actions1,c1)):
                        expected=1/(1+np.exp(-c.sum(axis=0)))
                        assert np.allclose(a,expected,rtol=1e-5,atol=1e-6)
                    shifts.append(shift)
                    action_diffs.append(actions1-actions0)
                blocks.append({
                    "seed":seed,"scene":scene,
                    "shift":np.stack(shifts).tolist(),
                    "action_diff":np.stack(action_diffs).tolist(),
                })
        assert len(blocks)==6
        delta=np.concatenate([np.asarray(z["shift"]) for z in blocks],axis=0)
        act=np.concatenate([np.asarray(z["action_diff"]) for z in blocks],axis=0)
        assert delta.shape==(60,8,2) and act.shape==(60,2)
        total=delta.sum(axis=1)
        component_abs=np.abs(delta).sum(axis=1)
        # 0 means all modules move together, ~1 means large module movements cancel.
        cancellation=1-np.abs(total).sum(axis=0)/np.maximum(component_abs.sum(axis=0),1e-12)
        results[variant]={
            "n_matched_steps":60,
            "mean_abs_module_logit_shift":np.abs(delta).mean(axis=0).tolist(),
            "mean_abs_total_logit_shift":np.abs(total).mean(axis=0).tolist(),
            "mean_signed_total_logit_shift":total.mean(axis=0).tolist(),
            "mean_signed_pedal_action_shift":act.mean(axis=0).tolist(),
            "total_module_movement_magnitude":component_abs.mean(axis=0).tolist(),
            "cross_module_logit_cancellation_fraction":cancellation.tolist(),
            "note":"Counterfactual readout decomposition only: cancellation is not automatically compensatory or beneficial.",
        }
    record={
        "stage":"DLS matched-prefix module output cancellation audit",
        "window":"physical sudden lead deceleration step70, readout steps90..99",
        "scope":"6 paired episodes per architecture, two independent real MetaDrive scene seeds and three trained models",
        "physical_meaning":"exact module contributions in the 2-channel sigmoid preactivation; each module shift is not an isolated physical cause",
        "results":results,
    }
    OUT.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps(record,ensure_ascii=False,indent=2))
    print("PASS: exact module-logit cancellation audit of real closed-loop lead-braking contrast",flush=True)


if __name__=="__main__":
    main()
