#!/usr/bin/env python3
"""D6c: 失活单元的发送信息必要性，2×2因果交互的训练种子/场景层级分析。"""

import json
from pathlib import Path
import numpy as np


ROOT=Path("/mnt/datasets/lxb/dls/reports")
SEEDS=(42,43,44)
DENSITIES=(0.0,0.10)
SCENES=tuple(range(42000,42012))
MOTORS=range(8)


def main():
    overall={
        "stage":"D6c outgoing message relay from motor with disabled output",
        "seeds":list(SEEDS),"densities":list(DENSITIES),
        "scene_seeds":list(SCENES),"motors":list(MOTORS),
        "new_episode_records":3*2*12*25,
        "train_new_models":False,
        "causal_definition":"positive (lesion_only - lesion_and_sender) - (base - sender_only) means sender message is more helpful with its own output disabled",
        "paired_analysis_unit":"scene; 8 motor contrasts averaged within each scene, training seeds reported separately",
        "results":{},
    }
    all_seed_density=[]
    for seed in SEEDS:
        overall["results"][str(seed)]={}
        for density in DENSITIES:
            episodes=[]
            for shard in range(3):
                p=ROOT/f"d6c/eval/seed{seed}-density{density:.2f}-shard{shard}.json"
                r=json.loads(p.read_text())
                assert r["seed"]==seed and r["density"]==density
                assert r["scene_seeds"]==list(SCENES[4*shard:4*(shard+1)])
                assert r["num_episodes"]==100 and r["cases_per_scene"]==25
                episodes.extend(r["episodes"])
            assert len(episodes)==300
            per_scene=[]
            motor_records={str(i):[] for i in MOTORS}
            case_total={name:{"success":0,"reward":[]}
                        for name in ("base","sender_only","lesion_only","lesion_and_sender")}
            for scene in SCENES:
                rows=[r for r in episodes if r["scene_seed"]==scene]
                assert len(rows)==25
                base=[r for r in rows if r["case"]=="base"]
                assert len(base)==1
                B=base[0]
                case_total["base"]["success"]+=int(B["success"])
                case_total["base"]["reward"].append(B["reward"])
                scene_reward_interactions=[]
                scene_success_interactions=[]
                for i in MOTORS:
                    C={}
                    for case in ("sender_only","lesion_only","lesion_and_sender"):
                        found=[r for r in rows if r["case"]==case and r["motor"]==i]
                        assert len(found)==1
                        C[case]=found[0]
                        case_total[case]["success"]+=int(found[0]["success"])
                        case_total[case]["reward"].append(found[0]["reward"])
                    # If disabling i's outgoing message matters MORE when its output is disabled,
                    # the lesion_specific_interaction is positive.
                    isucc=(int(C["lesion_only"]["success"])-int(C["lesion_and_sender"]["success"])) - (
                        int(B["success"])-int(C["sender_only"]["success"]))
                    irwd=(C["lesion_only"]["reward"]-C["lesion_and_sender"]["reward"]) - (
                        B["reward"]-C["sender_only"]["reward"])
                    assert np.isfinite(irwd)
                    scene_success_interactions.append(isucc)
                    scene_reward_interactions.append(irwd)
                    motor_records[str(i)].append({"scene":scene,
                        "reward_interaction":float(irwd),
                        "success_interaction":int(isucc)})
                per_scene.append({
                    "scene":scene,
                    "average_across_8_senders_success_interaction":float(np.mean(scene_success_interactions)),
                    "average_across_8_senders_reward_interaction":float(np.mean(scene_reward_interactions)),
                })
            base_success=case_total["base"]["success"]
            assert len(case_total["base"]["reward"])==12
            for case in ("sender_only","lesion_only","lesion_and_sender"):
                assert len(case_total[case]["reward"])==96
            r={
                "base_success_of_12":base_success,
                "sender_only_success_of_96":case_total["sender_only"]["success"],
                "lesion_only_success_of_96":case_total["lesion_only"]["success"],
                "lesion_and_sender_success_of_96":case_total["lesion_and_sender"]["success"],
                "mean_sender_lesion_specific_success_interaction_pp":float(100*np.mean([
                    p["average_across_8_senders_success_interaction"] for p in per_scene])),
                "mean_sender_lesion_specific_reward_interaction":float(np.mean([
                    p["average_across_8_senders_reward_interaction"] for p in per_scene])),
                "scene_level_paired_interactions":per_scene,
                "per_sender_paired_interactions":motor_records,
            }
            overall["results"][str(seed)][f"{density:.2f}"]=r
            all_seed_density.append(r)
            print(
                f"D6c seed{seed} density{density:.2f}: healthy={base_success}/12; "
                f"sender-only={r['sender_only_success_of_96']}/96, "
                f"lesion-only={r['lesion_only_success_of_96']}/96, "
                f"lesion+sender-block={r['lesion_and_sender_success_of_96']}/96; "
                f"lesion-specific sender interaction(success)={r['mean_sender_lesion_specific_success_interaction_pp']:+.2f}pp, "
                f"reward={r['mean_sender_lesion_specific_reward_interaction']:+.2f}",
                flush=True,
            )
    overall["descriptive_over_6_seed_density_cells"]={
        "mean_lesion_specific_sender_success_interaction_pp":float(np.mean([
            r["mean_sender_lesion_specific_success_interaction_pp"] for r in all_seed_density])),
        "mean_lesion_specific_sender_reward_interaction":float(np.mean([
            r["mean_sender_lesion_specific_reward_interaction"] for r in all_seed_density])),
        "positive_success_interaction_cells":sum(
            r["mean_sender_lesion_specific_success_interaction_pp"]>0 for r in all_seed_density),
        "positive_reward_interaction_cells":sum(
            r["mean_sender_lesion_specific_reward_interaction"]>0 for r in all_seed_density),
    }
    out=ROOT/"d6c/d6c-summary.json"
    assert not out.exists()
    out.write_text(json.dumps(overall,ensure_ascii=False,indent=2),encoding="utf-8")
    print("D6c complete:",json.dumps(overall["descriptive_over_6_seed_density_cells"]),flush=True)


if __name__=="__main__":
    main()
