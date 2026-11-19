#!/usr/bin/env python3
"""只读：已完成MetaDrive动态跟车的严格配对刹车效应与模块响应审计。"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path("/mnt/datasets/lxb/dls/reports/d1-longitudinal-v1-44117653285/dynamic-follow-v2")
OUTPUT=ROOT/"paired-mechanism-audit.json"


def extract(t, key):
    return next(a[key] for a in t if a["t"]==t_index) # unused


def summary():
    assert not OUTPUT.exists()
    paired=[]
    for variant in ("A","B","C"):
        for model_seed in (42,43,44):
            record=json.loads((ROOT/f"{variant}-seed{model_seed}.json").read_text())
            assert record["same_prefix_counterfactual_validated"]
            for scene in (41001,41002):
                a=next(r for r in record["results"] if r["scene_seed"]==scene and r["condition"]=="sham")
                b=next(r for r in record["results"] if r["scene_seed"]==scene and r["condition"]=="pulse")
                assert min(a["steps"],b["steps"])>=100
                before_a=a["steps_trace"][:70]
                before_b=b["steps_trace"][:70]
                assert all(np.isclose(x["ego_speed_mps"],y["ego_speed_mps"],atol=5e-5)
                           and np.isclose(x["lead_speed_mps"],y["lead_speed_mps"],atol=5e-5)
                           and np.isclose(x["gap_m"],y["gap_m"],atol=5e-5)
                           for x,y in zip(before_a,before_b))
                window=range(90,100)
                lead_difference=float(np.mean([
                    a["steps_trace"][t]["lead_speed_mps"]-
                    b["steps_trace"][t]["lead_speed_mps"] for t in window]))
                assert lead_difference>0.4,("Insufficient physical braking",variant,model_seed,scene,lead_difference)
                brake_shift=float(np.mean([
                    b["steps_trace"][t]["brake"]-a["steps_trace"][t]["brake"]
                    for t in window]))
                throttle_shift=float(np.mean([
                    b["steps_trace"][t]["throttle"]-a["steps_trace"][t]["throttle"]
                    for t in window]))
                responsibility_change=None
                if variant!="A":
                    responsibility_change=np.mean([
                        np.abs(np.asarray(b["steps_trace"][t]["unit_logit_contributions"])
                               -np.asarray(a["steps_trace"][t]["unit_logit_contributions"]))
                        for t in window],axis=0).tolist()
                paired.append({
                    "variant":variant,"model_seed":model_seed,"scene_seed":scene,
                    "frozen_weights":True,"same_first_70_steps":True,
                    "lead_speed_sham_minus_brake_mps_at_t90_99":lead_difference,
                    "brake_pulse_minus_sham_at_t90_99":brake_shift,
                    "throttle_pulse_minus_sham_at_t90_99":throttle_shift,
                    "module_mean_abs_logit_shift_90_99":responsibility_change,
                    "sham_collision":a["collision"],"pulse_collision":b["collision"],
                    "sham_success":a["success"],"pulse_success":b["success"],
                    "sham_steps":a["steps"],"pulse_steps":b["steps"],
                    "sham_min_surface_gap_m":a["min_surface_gap_m"],
                    "pulse_min_surface_gap_m":b["min_surface_gap_m"],
                })
    assert len(paired)==18
    results={}
    for variant in ("A","B","C"):
        d=[a for a in paired if a["variant"]==variant]
        assert len(d)==6
        results[variant]={
            "paired_cases":6,
            "lead_braking_physical_speed_difference_mps_mean":float(np.mean([
                a["lead_speed_sham_minus_brake_mps_at_t90_99"] for a in d])),
            "sham_collisions":sum(a["sham_collision"] for a in d),
            "pulse_collisions":sum(a["pulse_collision"] for a in d),
            "sham_success":sum(a["sham_success"] for a in d),
            "pulse_success":sum(a["pulse_success"] for a in d),
            "brake_response_pulse_minus_sham_90_99":float(np.mean([
                a["brake_pulse_minus_sham_at_t90_99"] for a in d])),
            "throttle_response_pulse_minus_sham_90_99":float(np.mean([
                a["throttle_pulse_minus_sham_at_t90_99"] for a in d])),
            "mean_abs_module_logit_shift_90_99":(
                np.mean([a["module_mean_abs_logit_shift_90_99"]
                         for a in d],axis=0).tolist() if variant!="A" else None),
        }
    report={
        "stage":"DLS controlled lead matched-history physical braking audit",
        "design":"18 paired comparisons: 3 architectures x 3 model seeds x 2 MetaDrive scene seeds",
        "identical_baseline_prefix_steps":70,
        "lead_brake_trigger_step":70,
        "causal_physical_manipulation":"lead speed target 7->0 m/s vs unchanged 7 m/s; actual lead speed verified",
        "window":"steps 90..99, restricted to both models existing; no future data given to controller",
        "limits":"These are OOD original controllers; module output contributions are not unique internal causal mechanisms; no claims of generalizable synergy",
        "per_architecture":results,
        "paired_records":paired,
    }
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="paired_records"},indent=2,ensure_ascii=False))
    print("PASS: 18 paired same-history physical MetaDrive braking interventions",flush=True)


if __name__=="__main__":
    summary()
