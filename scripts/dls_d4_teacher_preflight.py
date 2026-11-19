#!/usr/bin/env python3
"""DLS D4 safety and distribution preflight for diverse teacher following trajectories."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_d4_following_scene import collect_teacher_episode

ROOT=Path("/mnt/datasets/lxb/dls")
OUT=ROOT/"reports"/"d4-following-preflight"/"teacher-coverage-v1.json"


def main():
    assert not OUT.exists()
    records=[]
    for mode in ("pulse","mild","sham"):
        for seed in (51000,51001,51002,51003,51004,51005):
            obs,actions,states,info=collect_teacher_episode(seed,mode)
            assert obs.shape[0]==actions.shape[0]==states.shape[0]
            assert info["min_surface_gap_m"]>0,("reference made physical contact",info)
            assert not info["collision"],("reference crashed",info)
            assert info["event_was_reached"],("lead event occurs after episode",info)
            assert info["max_ego_speed_mps"]>0.5,("reference stalled",info)
            records.append(info)
            print("TEACHER",json.dumps({key:value for key,value in info.items()
                  if key in ("mode","seed","steps","success","collision","min_surface_gap_m",
                             "brake_fraction","event_was_reached","brake_after_event_fraction")},
                  ensure_ascii=False),flush=True)
    assert len(records)==18
    groups={}
    for mode in ("pulse","mild","sham"):
        rows=[x for x in records if x["mode"]==mode]
        assert len(rows)==6
        groups[mode]={
            "successes":sum(x["success"] for x in rows),
            "no_collisions":sum(not x["collision"] for x in rows),
            "event_reached":sum(x["event_was_reached"] for x in rows),
            "mean_brake_fraction":sum(x["brake_fraction"] for x in rows)/6,
            "mean_brake_after_event_fraction":sum(x["brake_after_event_fraction"] for x in rows)/6,
            "min_surface_gap_m":min(x["min_surface_gap_m"] for x in rows),
        }
    result={"phase":"D4 teacher preflight","n_episodes":18,
            "groups":groups,"episodes":records}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("PRECHECK",json.dumps(groups,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
