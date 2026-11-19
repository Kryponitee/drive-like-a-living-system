"""DLS D4: randomized real-physics MetaDrive following scenes and privileged teacher.

Separates *driving task* from architecture. No synthetic sensor swapping, no
hand-coded group identities. The teacher reads hidden physics and is NOT a fair
policy baseline. Hold-out scenes use disjoint seeds.
"""
from dataclasses import dataclass
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.manager import BaseManager

from dls_dual_pedal_adapter import install_dual_pedal_adapter, wrap_dual_pedal_env

MAX_STEPS = 210


def scene_spec(seed, mode):
    assert mode in ("pulse", "mild", "sham")
    rng = np.random.default_rng(int(seed) + 20261119)
    gap = float(rng.uniform(16., 25.))
    speed0 = float(rng.uniform(4.5, 7.0))
    speed1 = float(rng.uniform(6.0, 9.0))
    speed2 = (float(rng.uniform(0.0, 1.5)) if mode == "pulse"
              else float(rng.uniform(3.0, 5.0)) if mode == "mild"
              else speed1)
    speed3 = float(rng.uniform(5.0, 8.0))
    t1 = int(rng.integers(18, 30))
    t2 = t1 + int(rng.integers(28, 46))
    t3 = t2 + int(rng.integers(28, 47))
    return {
        "seed": int(seed),
        "mode": mode,
        "center_gap_m": gap,
        "targets_mps": [speed0, speed1, speed2, speed3],
        "switch_steps": [t1, t2, t3],
        "brake_start_step": t2,
    }


class D4TrafficManager(BaseManager):
    def __init__(self):
        super().__init__()
        self.lead = None
        self.spec = None
        self.tick = 0
        self.phase = 0

    def reset(self):
        self.lead = None
        self.spec = None
        self.tick = 0
        self.phase = 0

    def attach(self, ego, spec):
        assert self.lead is None
        assert self.spec is None
        gap = spec["center_gap_m"]
        direction = np.asarray(ego.heading, dtype=np.float64)
        assert direction.shape == (2,) and np.linalg.norm(direction) > 0.9
        direction /= np.linalg.norm(direction)
        position = np.asarray(ego.position, dtype=np.float64) + gap * direction
        self.lead = self.spawn_object(
            DefaultVehicle, vehicle_config=dict(), position=position,
            heading=float(ego.heading_theta))
        self.spec = spec
        assert self.lead is not ego

    def before_step(self):
        if self.lead is None:
            return {}
        t1, t2, t3 = self.spec["switch_steps"]
        self.phase = int(self.tick >= t1) + int(self.tick >= t2) + int(self.tick >= t3)
        target = self.spec["targets_mps"][self.phase]
        measured = float(self.lead.speed) / 3.6
        signed = float(np.clip(0.22 * (target - measured), -1., 1.))
        self.lead.before_step([0., signed])
        self.tick += 1
        return {}

    def after_step(self, *args, **kwargs):
        if self.lead is not None:
            self.lead.after_step()
        return {}


class D4FollowingEnv(MetaDriveEnv):
    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("traffic_manager", D4TrafficManager())


def make_env(seed):
    return wrap_dual_pedal_env(D4FollowingEnv({
        "use_render": False, "image_observation": False,
        "map": "S", "traffic_density": 0.,
        "num_scenarios": 1, "start_seed": int(seed),
        "horizon": MAX_STEPS + 1, "log_level": 50,
    }))


def physical_gap(ego, lead):
    lane = ego.lane
    e_s, _ = lane.local_coordinates(ego.position)
    l_s, l_y = lane.local_coordinates(lead.position)
    center = float(l_s - e_s)
    surface = center - (float(ego.LENGTH) + float(lead.LENGTH)) / 2.
    return center, surface, float(l_y)


def teacher_action(ego, lead):
    """Conservative longitudinal ACC with privileged physics, from DLS P0 gate."""
    center_gap, surface, _ = physical_gap(ego, lead)
    v = float(ego.speed) / 3.6
    vl = float(lead.speed) / 3.6
    closing = max(v - vl, 0.)
    desired_gap = 3.5 + 1.1 * v + 1.0 * closing
    target_v = min(9., max(0., vl + 0.52 * (surface - desired_gap)))
    accel = 1.0 * (target_v - v) - 0.65 * closing
    if surface < desired_gap:
        accel = min(accel, -0.85 * (desired_gap - surface))
    return np.asarray([float(np.clip(accel / 1.6, 0., 1.)),
                       float(np.clip(-accel / 2.5, 0., 1.))], dtype=np.float32)


def collect_teacher_episode(seed, mode):
    spec = scene_spec(seed, mode)
    env = make_env(seed)
    observations = []
    actions = []
    states = []
    try:
        obs, reset_info = env.reset(seed=int(seed))
        ego = env.agent
        install_dual_pedal_adapter(ego, steering_controller=lambda _: 0.)
        manager = env.unwrapped.engine.traffic_manager
        assert isinstance(manager, D4TrafficManager)
        manager.attach(ego, spec)
        last_info = reset_info
        for step in range(MAX_STEPS):
            x = np.asarray(obs, dtype=np.float32)
            assert x.shape == (259,) and np.isfinite(x).all()
            pedal = teacher_action(ego, manager.lead)
            center, gap, lateral = physical_gap(ego, manager.lead)
            observations.append(x.copy())
            actions.append(pedal.copy())
            states.append((gap, float(ego.speed) / 3.6,
                           float(manager.lead.speed) / 3.6,
                           float(manager.phase)))
            obs, reward, done, truncated, last_info = env.step(pedal)
            if done or truncated:
                break
        collision = bool(last_info.get("crash_vehicle", False)
                         or last_info.get("crash_object", False)
                         or last_info.get("crash_building", False))
        state = np.asarray(states, dtype=np.float32)
        acts = np.stack(actions)
        record = {
            **spec,
            "steps": len(actions),
            "success": bool(last_info.get("arrive_dest", False)),
            "collision": collision,
            "out_of_road": bool(last_info.get("out_of_road", False)),
            "min_surface_gap_m": float(state[:, 0].min()),
            "max_ego_speed_mps": float(state[:, 1].max()),
            "min_lead_speed_mps": float(state[:, 2].min()),
            "max_lead_speed_mps": float(state[:, 2].max()),
            "brake_fraction": float((acts[:, 1] > 0.1).mean()),
            "throttle_fraction": float((acts[:, 0] > 0.1).mean()),
            "brake_after_event_fraction": float(
                (acts[spec["brake_start_step"]:, 1] > 0.1).mean())
                if len(acts) > spec["brake_start_step"] else None,
            "event_was_reached": len(acts) > spec["brake_start_step"] + 5,
        }
        return np.stack(observations), acts, state, record
    finally:
        env.close()
