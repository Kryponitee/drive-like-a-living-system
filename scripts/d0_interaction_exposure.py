#!/usr/bin/env python3
"""D0d：只读诊断专家轨迹是否真正经历了邻车和同车道前车交互。

本脚本不修改环境、专家策略、训练数据或控制动作；特权物理量仅用于审计，
绝不作为学习策略的输入。
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert


data_root = Path(os.environ["DLS_DATA_ROOT"]).resolve()
assert data_root == Path("/mnt/datasets/lxb/dls")
assert os.path.ismount("/mnt/datasets")
report_dir = data_root / "reports" / "d0"
report_dir.mkdir(parents=True, exist_ok=True)

def physical_context(ego, traffic_vehicles):
    # 邻车曝光按欧氏距离；同车道前车按同一 lane index + 局部纵坐标定义。
    ego_xy = np.asarray(ego.position, dtype=np.float64)
    assert ego_xy.shape == (2,)
    ego_speed = float(ego.speed_km_h)
    ego_lane = ego.lane_index
    ego_long = float(ego.lane.local_coordinates(ego.position)[0])
    closest_any = float("inf")
    closest_lead = float("inf")
    lead_speed = float("nan")

    for vehicle in traffic_vehicles:
        if vehicle.id == ego.id:
            continue
        car_xy = np.asarray(vehicle.position, dtype=np.float64)
        assert car_xy.shape == (2,)
        distance = float(np.linalg.norm(car_xy - ego_xy))
        closest_any = min(closest_any, distance)
        if vehicle.lane_index == ego_lane:
            car_long = float(ego.lane.local_coordinates(vehicle.position)[0])
            gap = car_long - ego_long  # 车中心之间距离，不扣除车长
            if 0.0 < gap < closest_lead:
                closest_lead = gap
                lead_speed = float(vehicle.speed_km_h)

    return {
        "other_vehicle_within_40m": closest_any <= 40.0,
        "same_lane_lead_within_50m": closest_lead <= 50.0,
        "closing_on_lead_within_40m": closest_lead <= 40.0 and ego_speed - lead_speed > 5.0,
        "min_other_vehicle_center_dist_m": closest_any,
        "min_same_lane_lead_center_gap_m": closest_lead,
    }


groups = []
for density in (0.1, 0.25):
    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "traffic_density": density,
        "map": 3,
        "start_seed": 0,
        "num_scenarios": 32,
        "horizon": 800,
        "log_level": 50,
    })
    episodes = []

    for seed in range(12):
        obs, _ = env.reset(seed=seed)
        counters = {
            "near_other_steps": 0,
            "lead_steps": 0,
            "closing_steps": 0,
            "braking_under_lead_steps": 0,
            "braking_without_lead_steps": 0,
            "braking_total_steps": 0,
        }
        min_other = float("inf")
        min_lead = float("inf")
        end_info = {}
        for step_idx in range(800):
            context = physical_context(env.agent, env.engine.traffic_manager.traffic_vehicles)
            action = np.clip(np.asarray(expert(env.agent, deterministic=True), dtype=np.float32), -1, 1)
            assert action.shape == (2,) and np.isfinite(action).all()
            braking = bool(action[1] < -0.05)
            lead_present = context["same_lane_lead_within_50m"]
            counters["near_other_steps"] += int(context["other_vehicle_within_40m"])
            counters["lead_steps"] += int(lead_present)
            counters["closing_steps"] += int(context["closing_on_lead_within_40m"])
            counters["braking_total_steps"] += int(braking)
            counters["braking_under_lead_steps"] += int(braking and lead_present)
            counters["braking_without_lead_steps"] += int(braking and not lead_present)
            min_other = min(min_other, context["min_other_vehicle_center_dist_m"])
            min_lead = min(min_lead, context["min_same_lane_lead_center_gap_m"])

            obs, _, terminated, truncated, end_info = env.step(action)
            if terminated or truncated:
                break

        episode = {
            "seed": seed,
            "steps": step_idx + 1,
            "arrive_dest": bool(end_info.get("arrive_dest", False)),
            "crash_vehicle": bool(end_info.get("crash_vehicle", False)),
            "out_of_road": bool(end_info.get("out_of_road", False)),
            "timeout": bool(end_info.get("max_step", False)),
            "min_other_center_dist_m": min_other if np.isfinite(min_other) else None,
            "min_same_lane_lead_center_gap_m": min_lead if np.isfinite(min_lead) else None,
            **counters,
        }
        episodes.append(episode)

    env.close()
    steps = sum(e["steps"] for e in episodes)
    success = [e for e in episodes if e["arrive_dest"]]
    lead_in_success = [e for e in success if e["lead_steps"] > 0]
    groups.append({
        "traffic_density": density,
        "episode_count": len(episodes),
        "success_count": len(success),
        "collision_count": sum(e["crash_vehicle"] for e in episodes),
        "timeout_count": sum(e["timeout"] for e in episodes),
        "episodes_with_other_vehicle_under_40m": sum(e["near_other_steps"] > 0 for e in episodes),
        "episodes_with_same_lane_lead_under_50m": sum(e["lead_steps"] > 0 for e in episodes),
        "success_episodes_with_same_lane_lead_under_50m": len(lead_in_success),
        "successful_episodes_total": len(success),
        "near_other_step_fraction": sum(e["near_other_steps"] for e in episodes) / steps,
        "same_lane_lead_step_fraction": sum(e["lead_steps"] for e in episodes) / steps,
        "closing_step_fraction": sum(e["closing_steps"] for e in episodes) / steps,
        "braking_given_lead_step_fraction": (
            sum(e["braking_under_lead_steps"] for e in episodes) /
            max(1, sum(e["lead_steps"] for e in episodes))
        ),
        "braking_without_lead_step_fraction": (
            sum(e["braking_without_lead_steps"] for e in episodes) /
            max(1, steps - sum(e["lead_steps"] for e in episodes))
        ),
        "episodes": episodes,
    })
    row = groups[-1]
    print(
        f"density={density}, success={row['success_count']}/12, "
        f"lead_exposed={row['episodes_with_same_lane_lead_under_50m']}/12, "
        f"successful_lead_exposed={row['success_episodes_with_same_lane_lead_under_50m']}/{len(success)}, "
        f"closing_step_fraction={row['closing_step_fraction']:.4f}, "
        f"brake_given_lead={row['braking_given_lead_step_fraction']:.4f}, "
        f"brake_without_lead={row['braking_without_lead_step_fraction']:.4f}",
        flush=True,
    )

report = {
    "stage": "D0d privileged scenario telemetry; not part of model input",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "same_lane_definition": "ego.lane_index == traffic_vehicle.lane_index",
    "lead_definition": "0 < longitudinal center gap in ego lane < 50 m",
    "closing_definition": "lead within 40m and ego speed > lead speed + 5 km/h",
    "caveat": "A same-lane lead, a closing condition, and a braking action do NOT establish a specific lead-braking, lane-change, or recovery event. Sparse contacts and changing road segments can be undercounted.",
    "hold_fixed": ["expert PPO deterministic", "map=3", "seed=0..11", "horizon=800"],
    "data_root": str(data_root),
    "results": groups,
}
out = report_dir / "expert_interaction_exposure.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved to {out}")
