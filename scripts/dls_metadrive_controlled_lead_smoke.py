#!/usr/bin/env python3
"""MetaDrive controlled leader feasibility gate: no training and no coordination claims.

Reuse MetaDrive environment. Spawn exactly one same-lane scripted leader under BaseManager,
and apply physically executed speed-dependent acceleration/deceleration.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager import BaseManager
from metadrive.component.vehicle.vehicle_type import DefaultVehicle

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dls_dual_pedal_adapter import install_dual_pedal_adapter,wrap_dual_pedal_env

ROOT=Path("/mnt/datasets/lxb/dls")

# Controls are signed MetaDrive throttle/brake for the other car, not DLS's [T,B].
TARGET_SPEED_MPS=(8.,13.,2.,9.)
PHASE_STEPS=(0,45,90,135,180)


class ScriptedLeadManager(BaseManager):
    def __init__(self):
        super().__init__()
        self.lead=None
        self.tick=0
        self.phase=0
        self.spawn_s=None
        self.spawn_gap=None

    def reset(self):
        self.lead=None
        self.tick=0
        self.phase=0

    def attach_lead(self,ego,gap_m=26):
        assert self.lead is None
        lane=ego.lane
        assert lane is not None, "Ego must be assigned to a valid lane."
        longitudinal,lateral=lane.local_coordinates(ego.position)
        lane_length=float(lane.length)
        assert lane_length>longitudinal+gap_m+5, (lane_length,longitudinal,gap_m)
        lead_s=longitudinal+gap_m
        pos=lane.position(lead_s,lateral)
        heading=float(lane.heading_theta_at(lead_s))
        self.lead=self.spawn_object(
            DefaultVehicle,vehicle_config=dict(),position=pos,heading=heading)
        self.spawn_s=float(lead_s)
        self.spawn_gap=float(gap_m)
        assert self.lead is not ego
        print("LEAD_SPAWN",json.dumps({
            "ego_xy":np.asarray(ego.position).tolist(),
            "lead_xy":np.asarray(self.lead.position).tolist(),
            "ego_lane_length":lane_length,"ego_s":float(longitudinal),
            "lead_s":float(lead_s),"lateral":float(lateral),
            "initial_heading_rad":heading,
            "ego_speed_kmh":float(ego.speed),
            "lead_speed_kmh":float(self.lead.speed)
        }),flush=True)

    def before_step(self):
        if self.lead is None:
            return {}
        self.phase=min(self.tick//45,3)
        target=TARGET_SPEED_MPS[self.phase]
        measured_speed=float(self.lead.speed)/3.6  # MetaDrive vehicle.speed is km/h
        control=float(np.clip(0.22*(target-measured_speed),-1.0,1.0))
        self.lead.before_step([0.0,control])
        self.tick+=1
        return {}

    def after_step(self, *args, **kwargs):
        if self.lead is not None:
            self.lead.after_step()
        return {}


class ControlledLeadEnv(MetaDriveEnv):
    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("traffic_manager",ScriptedLeadManager())


def get_relative(ego,lead):
    # The same ego lane is straight in this experiment; report longitudinal projection.
    lane=ego.lane
    es,_=lane.local_coordinates(ego.position)
    ls,lateral=lane.local_coordinates(lead.position)
    return float(ls-es),float(lateral)


def run(seed,max_steps):
    env=wrap_dual_pedal_env(ControlledLeadEnv({
        "use_render":False,"image_observation":False,
        "map":"S","traffic_density":0.0,
        "num_scenarios":1,"start_seed":seed,
        "horizon":max_steps+10,"log_level":50,
    }))
    states=[]
    try:
        obs,_=env.reset(seed=seed)
        ego=env.agent
        install_dual_pedal_adapter(ego,steering_controller=lambda _:0.0)
        mgr=env.engine.traffic_manager
        assert isinstance(mgr,ScriptedLeadManager)
        mgr.attach_lead(ego,gap_m=26)
        for t in range(max_steps):
            # Pedal pattern for physics gate only; not a trained neural controller.
            action=np.asarray([0.48,0.0],dtype=np.float32) if t<65 else np.asarray([0.,0.12],dtype=np.float32)
            obs,reward,done,truncated,info=env.step(action)
            gap,lateral=get_relative(ego,mgr.lead)
            rec={"t":t,"phase":mgr.phase,"gap_m":gap,
                 "ego_speed_kmh":float(ego.speed),
                 "lead_speed_kmh":float(mgr.lead.speed),
                 "target_lead_speed_mps":TARGET_SPEED_MPS[mgr.phase],
                 "lateral_m":lateral,
                 "observations_finite":bool(np.isfinite(np.asarray(obs)).all()),
                 "crash_vehicle":bool(info.get("crash_vehicle",False))}
            states.append(rec)
            if t%20==0:print("STEP",json.dumps(rec),flush=True)
            if done or truncated:break
        assert len(states)>=90,("episode too short",len(states),states[-1])
        # Mandatory physically observed acceleration and braking, not merely schedule updates.
        lead_speeds=np.asarray([s["lead_speed_kmh"] for s in states])
        assert float(np.max(lead_speeds)-np.min(lead_speeds))>6.0
        assert all(np.isfinite(s["gap_m"]) for s in states)
        assert all(s["observations_finite"] for s in states)
        assert np.max(np.abs([s["lateral_m"] for s in states]))<3.0
        assert set(s["phase"] for s in states)>=set((0,1)), "No requested speed transition"
        return {
            "stage":"DLS controlled MetaDrive following scene feasibility",
            "seed":seed,"n_steps":len(states),
            "lead_speed_kmh_range":[float(np.min(lead_speeds)),float(np.max(lead_speeds))],
            "min_gap_m":min(s["gap_m"] for s in states),
            "min_lateral_m":min(s["lateral_m"] for s in states),
            "max_lateral_m":max(s["lateral_m"] for s in states),
            "phase_reached":max(s["phase"] for s in states),
            "has_ego_lead_crash":any(s["crash_vehicle"] for s in states),
            "states":states,
        }
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=41000)
    p.add_argument("--max-steps",type=int,default=180)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    assert os.path.ismount("/mnt/datasets")
    assert not args.output.exists()
    report=run(args.seed,args.max_steps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("PASS: physically controlled MetaDrive lead speed intervention and real-vehicle observation gate",flush=True)


if __name__=="__main__":
    main()
