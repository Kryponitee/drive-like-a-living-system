#!/usr/bin/env python3
"""D0': 验证独立绝对油门/制动执行通道，不训练DLS。"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dls_dual_pedal_adapter import install_dual_pedal_adapter


DATA_ROOT = Path(os.environ["DLS_DATA_ROOT"]).resolve()
assert DATA_ROOT == Path("/mnt/datasets/lxb/dls")
assert os.path.ismount("/mnt/datasets")

REPORT_DIR = DATA_ROOT / "reports" / "d0-dual-pedal"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "actuator-smoke.json"
assert not REPORT_PATH.exists(), f"Refusing to overwrite {REPORT_PATH}"

SEEDS = (7101, 7102, 7103)
ACCEL_STEPS = 80
BRAKE_STEPS = 80


def run_one(seed):
    env = MetaDriveEnv(
        {
            "use_render": False,
            "image_observation": False,
            "map": "S",
            "traffic_density": 0.0,
            "num_scenarios": 1,
            "horizon": 600,
            "log_level": 50,
            "vehicle_config": {
                "max_engine_force": 800,
                "max_brake_force": 180,
            },
        }
    )
    obs, reset_info = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    assert obs.ndim == 1 and np.isfinite(obs).all()
    assert env.action_space.shape == (2,), env.action_space

    vehicle = env.agent
    install_dual_pedal_adapter(vehicle, steering_controller=lambda _: 0.0)
    engine_max = float(vehicle.config["max_engine_force"])
    brake_max = float(vehicle.config["max_brake_force"])
    assert np.isclose(engine_max, 800.0), engine_max
    assert np.isclose(brake_max, 180.0), brake_max

    def step(throttle, brake):
        action = np.asarray([throttle, brake], dtype=np.float32)
        assert action.shape == (2,)
        assert bool((action >= 0.0).all() and (action <= 1.0).all())
        assert env.action_space.contains(action), (env.action_space, action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated, (seed, info)
        assert np.isfinite(np.asarray(next_obs)).all()
        assert np.isclose(vehicle.dls_throttle, throttle, atol=1e-6)
        assert np.isclose(vehicle.dls_brake, brake, atol=1e-6)
        assert np.isclose(vehicle.steering, 0.0, atol=1e-8)
        assert np.isclose(vehicle.dls_steering, 0.0, atol=1e-8)
        return float(vehicle.speed)

    speed_initial = float(vehicle.speed)
    accel_speeds = [step(1.0, 0.0) for _ in range(ACCEL_STEPS)]
    speed_after_accel = accel_speeds[-1]
    assert speed_after_accel > speed_initial + 0.5, (seed, speed_initial, speed_after_accel)
    assert np.isclose(vehicle.dls_engine_force, engine_max, rtol=1e-6)
    assert np.isclose(vehicle.dls_brake_force, 0.0, atol=1e-8)

    brake_speeds = [step(0.0, 1.0) for _ in range(BRAKE_STEPS)]
    speed_after_brake = brake_speeds[-1]
    assert speed_after_brake < speed_after_accel - 0.25, (
        seed, speed_after_accel, speed_after_brake
    )
    assert np.isclose(vehicle.dls_engine_force, 0.0, atol=1e-8)
    assert np.isclose(vehicle.dls_brake_force, brake_max, rtol=1e-6)

    # 同时施加两路命令：验证适配层没有做互斥、裁剪相减或符号折叠。
    speed_after_simultaneous = step(0.6, 0.4)
    assert np.isclose(vehicle.dls_engine_force, 0.6 * engine_max, rtol=1e-6)
    assert np.isclose(vehicle.dls_brake_force, 0.4 * brake_max, rtol=1e-6)
    assert vehicle.dls_engine_force > 0.0 and vehicle.dls_brake_force > 0.0
    simultaneous = {
        "throttle": vehicle.dls_throttle,
        "brake": vehicle.dls_brake,
        "engine_force": vehicle.dls_engine_force,
        "brake_force": vehicle.dls_brake_force,
    }

    # 改变油门而固定制动、改变制动而固定油门，分别只改变对应执行器。
    step(0.8, 0.2)
    pair_a = (vehicle.dls_engine_force, vehicle.dls_brake_force)
    step(0.3, 0.2)
    pair_b = (vehicle.dls_engine_force, vehicle.dls_brake_force)
    assert np.isclose(pair_a[1], pair_b[1], rtol=1e-6)
    assert pair_a[0] > pair_b[0]

    step(0.3, 0.8)
    pair_c = (vehicle.dls_engine_force, vehicle.dls_brake_force)
    assert np.isclose(pair_b[0], pair_c[0], rtol=1e-6)
    assert pair_c[1] > pair_b[1]

    result = {
        "seed": seed,
        "map": "S",
        "traffic_density": 0.0,
        "observation_dim": int(obs.size),
        "action_space": str(env.action_space),
        "control_interface": "[absolute_throttle, absolute_brake], each in [0,1]",
        "steering_controller": "fixed_zero_on_straight_map",
        "max_engine_force": engine_max,
        "max_brake_force": brake_max,
        "acceleration_steps": ACCEL_STEPS,
        "braking_steps": BRAKE_STEPS,
        "speed_initial_mps": speed_initial,
        "speed_after_acceleration_mps": speed_after_accel,
        "speed_after_braking_mps": speed_after_brake,
        "speed_after_simultaneous_command_mps": speed_after_simultaneous,
        "simultaneous": simultaneous,
        "throttle_independence_pair": [list(pair_a), list(pair_b)],
        "brake_independence_pair": [list(pair_b), list(pair_c)],
        "passed": True,
    }
    env.close()
    return result


results = [run_one(seed) for seed in SEEDS]
assert len(results) == 3 and all(r["passed"] for r in results)
report = {
    "stage": "D0' independent absolute throttle-brake actuator smoke",
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "execution-interface verification only; no policy training or coordination claim",
    "seeds": list(SEEDS),
    "num_episodes": len(results),
    "all_passed": True,
    "checks": [
        "throttle-only command accelerates from rest",
        "brake-only command reduces speed after acceleration",
        "simultaneous positive throttle and brake reach separate physical actuator APIs",
        "changing throttle while brake is fixed changes engine force only",
        "changing brake while throttle is fixed changes brake force only",
        "steering is fixed at zero on a straight map",
    ],
    "episodes": results,
    "report_path": str(REPORT_PATH),
}
REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
