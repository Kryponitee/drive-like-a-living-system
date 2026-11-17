#!/usr/bin/env python3
"""D2f final aggregation, read-only against original per-episode reports.

The run that generated the data contained a metadata-only bug:
rollout loop used actual seeds 4400..4423, but top-level "seeds" declared 4300..4311.
Use per-episode seed records as primary provenance; DO NOT modify old raw reports.
"""
import json
import os
from pathlib import Path

ROOT=Path("/mnt/datasets/lxb/dls/reports/d2f")
EXPECTED=list(range(4400,4424))
DENSITIES=(0.0,0.1,0.25)

def load(phase, variant, density, warnings):
    filename=ROOT/f"eval-{phase}-{variant}-density{density:.2f}.json"
    raw=json.loads(filename.read_text())
    observed=[e["seed"] for e in raw["episodes"]]
    assert observed==EXPECTED, (str(filename),observed)
    assert raw["traffic_density"]==density
    assert raw["phase"]==phase and raw["variant"]==variant
    assert raw["success"]==sum(e["success"] for e in raw["episodes"])
    assert raw["collision"]==sum(e["collision"] for e in raw["episodes"])
    assert raw["offroad"]==sum(e["out_of_road"] for e in raw["episodes"])
    if raw["seeds"]!=observed:
        assert raw["seeds"]==list(range(4300,4312)), "Unexpected metadata mismatch"
        warnings.append({
            "raw_file":str(filename),
            "mislabelled_top_level_seed_range":raw["seeds"],
            "actual_per_episode_seeds":observed,
        })
    return raw

def main():
    assert os.path.ismount("/mnt/datasets")
    assert (ROOT/"d2f-validation-drift.json").exists()
    validation=json.loads((ROOT/"d2f-validation-drift.json").read_text())
    assert set(validation["models"])==set("ABC")
    warnings=[]
    result={
        "stage":"D2f fixed coefficient BC functional anchoring, final read-only verified report",
        "source_workflow":"https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44015006851",
        "source_run_summary_status":"failed at final summarization only; all training, per-case evaluations, validation passed",
        "validation_data_steps":9822,
        "sigma":0.05,"lambda":10.0,"anchor_batch":16,
        "model_seed":42,"training_steps_per_variant":4096,
        "new_evaluation_seeds":EXPECTED,"densities":list(DENSITIES),
        "top_level_seed_metadata_bug_preserved_in_original_raw_reports":True,
        "repair_strategy":"Read actual episode[*].seed, keep original reports unchanged and publish corrected aggregate",
        "models":{},"expert":{},
    }
    for variant in "ABC":
        tr=json.loads((ROOT/f"train-{variant}-seed42.json").read_text())
        assert tr["environment_steps"]==4096 and tr["updates"]==32
        assert tr["anchor_lambda"]==10.0 and tr["initial_exploration_sigma"]==0.05
        assert tr["max_ppo_behavior_replay_error"]<3e-5
        item={
            "training_successful_episodes":tr["num_successful_completed_training_episodes"],
            "mean_anchor_MSE":tr["mean_anchor_mse_last_epoch"],
            "max_behavior_replay_ratio_error":tr["max_ppo_behavior_replay_error"],
            "BC_validation_Huber":validation["models"][variant],
            "by_density":{},
        }
        for density in DENSITIES:
            rows={phase:load(phase,variant,density,warnings)
                  for phase in ("bc","d2d","d2f")}
            ref=rows["bc"]["episodes"]
            out={}
            for phase,record in rows.items():
                episodes=record["episodes"]
                wins=sum((not a["success"]) and b["success"] for a,b in zip(ref,episodes))
                losses=sum(a["success"] and (not b["success"]) for a,b in zip(ref,episodes))
                out[phase]={
                    "success":record["success"],"collisions":record["collision"],
                    "offroad":record["offroad"],"timeouts":record["timeout"],
                    "mean_reward":record["mean_reward"],
                    "mean_completion":record["mean_completion"],
                    "success_only_this_phase_vs_BC":wins,
                    "success_only_BC":losses,
                    "net_success_vs_BC":wins-losses,
                    "mean_reward_diff_vs_BC":record["mean_reward"]-rows["bc"]["mean_reward"],
                }
            item["by_density"][str(density)]=out
        result["models"][variant]=item
        print(variant,"train successes",item["training_successful_episodes"],
              "Huber BC/D2d/D2f",
              *(round(item["BC_validation_Huber"][phase]["Huber"],6) for phase in ("bc","d2d","d2f")),flush=True)
        for d, metrics in item["by_density"].items():
            print(variant,"density",d,
                  "BC/D2d/D2f success",*(metrics[phase]["success"] for phase in ("bc","d2d","d2f")),
                  "mean reward",*(round(metrics[phase]["mean_reward"],2) for phase in ("bc","d2d","d2f")),flush=True)
    for density in DENSITIES:
        ex=load("expert","expert",density,warnings)
        result["expert"][str(density)]={
            "success":ex["success"],"collisions":ex["collision"],"offroad":ex["offroad"],
            "timeouts":ex["timeout"],"mean_reward":ex["mean_reward"],
        }
    assert len(warnings)==30, f"Unexpected metadata defects: {len(warnings)}"
    result["raw_reports_metadata_warnings"]=warnings
    summary=ROOT/"d2f-summary.json"
    assert not summary.exists(), "Refuse overwriting previously computed aggregate"
    summary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Done: validated 30 raw reports using actual episode seeds; all raw files unchanged.")
    print("Corrected aggregate:",summary)

if __name__=="__main__":
    main()
