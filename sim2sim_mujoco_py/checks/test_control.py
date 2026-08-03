"""控制器测试。

保护 L5A 的混合动作语义：6 个腿部关节是位置目标，2 个轮子是速度目标；
同时验证 policy_order 到 hardware_order 的映射、PD 公式、力矩裁剪和动作延迟。
"""

from __future__ import annotations

import numpy as np
from l5a_sim2sim.control import ActionDelayBuffer, MixedPDController


def test_leg_and_wheel_targets_and_policy_to_hardware_mapping(deployment_contract: dict) -> None:
    """action 前 6 维改变腿部目标角，后 2 维改变轮子目标速度。"""
    controller = MixedPDController(deployment_contract)
    action = np.arange(1.0, 9.0)
    q_target, dq_target = controller.targets(action)
    defaults = np.asarray(deployment_contract["default_joint_positions"]["values"])
    np.testing.assert_allclose(q_target[:6], defaults[:6] + 0.25 * action[:6])
    np.testing.assert_allclose(dq_target[:6], 0.0)
    np.testing.assert_allclose(dq_target[6:], action[6:])
    np.testing.assert_array_equal(controller.to_hardware_order(action), action[[0, 1, 2, 6, 3, 4, 5, 7]])


def test_pd_formula_and_effort_clipping(deployment_contract: dict) -> None:
    """PD 输出应符合 Manifest 中的 kp/kd，并受 effort_limits 裁剪。"""
    controller = MixedPDController(deployment_contract)
    defaults = np.asarray(deployment_contract["default_joint_positions"]["values"])
    action = np.ones(8)
    torque = controller.compute_policy_torque(action, defaults, np.zeros(8))
    np.testing.assert_allclose(torque[:6], [10.0, 10.0, 20.0, 10.0, 10.0, 20.0])
    np.testing.assert_allclose(torque[6:], [1.5, 1.5])

    saturated = controller.compute_policy_torque(np.full(8, 1000.0), defaults, np.zeros(8))
    np.testing.assert_allclose(
        saturated,
        deployment_contract["joint_control"]["effort_limits"],
    )


def test_action_delay_is_measured_in_physics_steps() -> None:
    """动作延迟单位是 5 ms MuJoCo 物理步，不是 10 ms 策略步。"""
    delay = ActionDelayBuffer(2)
    first = np.ones(8, dtype=np.float32)
    second = np.full(8, 2.0, dtype=np.float32)
    third = np.full(8, 3.0, dtype=np.float32)
    np.testing.assert_array_equal(delay.apply(first), np.zeros(8))
    np.testing.assert_array_equal(delay.apply(second), np.zeros(8))
    np.testing.assert_array_equal(delay.apply(third), first)
