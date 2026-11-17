"""独立绝对油门/制动适配器；不把两路执行量压缩成有符号踏板。"""

from types import MethodType

import gymnasium as gym
import numpy as np


def _dual_pedal_set_action(vehicle, action):
    values = np.asarray(action, dtype=np.float32)
    assert values.shape == (2,), values.shape
    assert np.isfinite(values).all()
    assert bool((values >= -1e-6).all() and (values <= 1.0 + 1e-6).all()), values

    throttle = float(np.clip(values[0], 0.0, 1.0))
    brake = float(np.clip(values[1], 0.0, 1.0))
    steering = float(vehicle._dls_steering_controller(vehicle))
    assert np.isfinite(steering) and -1.0 <= steering <= 1.0

    max_engine_force = float(vehicle.config["max_engine_force"])
    max_brake_force = float(vehicle.config["max_brake_force"])
    engine_force = throttle * max_engine_force
    brake_force = brake * max_brake_force

    # 转向由外部控制器负责；本轮D0'在直线路况使用恒零转向。
    vehicle.system.setSteeringValue(steering * vehicle.max_steering, 0)
    vehicle.system.setSteeringValue(steering * vehicle.max_steering, 1)

    # 两路执行器独立写入Bullet车辆。允许T>0且B>0同时出现，不做互斥或相减。
    for wheel_index in range(4):
        vehicle.system.applyEngineForce(engine_force, wheel_index)
        vehicle.system.setBrake(brake_force, wheel_index)

    vehicle.steering = steering
    vehicle.throttle_brake = throttle - brake  # 仅兼容MetaDrive旧状态字段，不作为执行输入
    vehicle.dls_throttle = throttle
    vehicle.dls_brake = brake
    vehicle.dls_engine_force = engine_force
    vehicle.dls_brake_force = brake_force
    vehicle.dls_steering = steering
    vehicle.dls_control_mode = "independent_absolute_throttle_brake"



class DualPedalActionSpace(gym.Wrapper):
    """保留MetaDrive物理执行，只把外部策略接口声明为[T, B]∈[0,1]^2。"""

    def __init__(self, env):
        assert tuple(env.action_space.shape) == (2,), env.action_space
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        assert tuple(self.action_space.shape) == (2,)
        assert np.allclose(self.action_space.low, np.zeros(2, dtype=np.float32))
        assert np.allclose(self.action_space.high, np.ones(2, dtype=np.float32))

    @property
    def agent(self):
        return self.env.agent


def wrap_dual_pedal_env(env):
    return DualPedalActionSpace(env)

def install_dual_pedal_adapter(vehicle, steering_controller):
    """将env.step([T, B])适配为独立发动机力和制动力。每次新建/重置车辆后安装。"""
    assert callable(steering_controller)
    assert not hasattr(vehicle, "_dls_original_set_action"), "Adapter already installed"
    vehicle._dls_original_set_action = vehicle._set_action
    vehicle._dls_steering_controller = steering_controller
    vehicle._set_action = MethodType(_dual_pedal_set_action, vehicle)
    vehicle.dls_control_mode = "independent_absolute_throttle_brake"
    return vehicle
