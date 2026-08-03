"""观测和历史缓冲测试。

保护 28 维 proprioception 的拼接/缩放、四元数到投影重力的计算，以及 10 帧
历史窗口的首帧填充和 oldest-to-newest 移位语义。
"""

from __future__ import annotations

import numpy as np
from l5a_sim2sim.history import ObservationHistory
from l5a_sim2sim.observation import ObservationBuilder, projected_gravity_from_quaternion


def test_history_first_frame_fill_and_oldest_to_newest_shift() -> None:
    """首次 append 复制填满历史，后续 append 按 oldest-to-newest 左移。"""
    history = ObservationHistory(3, 2)
    first = np.array([1.0, 2.0], dtype=np.float32)
    history.append(first)
    np.testing.assert_array_equal(history.data, np.tile(first, (3, 1)))

    second = np.array([3.0, 4.0], dtype=np.float32)
    history.append(second)
    np.testing.assert_array_equal(history.data, np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]))
    assert history.batched().shape == (1, 3, 2)


def test_projected_gravity_for_identity_and_known_roll() -> None:
    """单位姿态和已知 roll 姿态下，投影重力方向应可手算验证。"""
    np.testing.assert_allclose(projected_gravity_from_quaternion([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, -1.0])
    half_sqrt = np.sqrt(0.5)
    np.testing.assert_allclose(
        projected_gravity_from_quaternion([half_sqrt, half_sqrt, 0.0, 0.0]),
        [0.0, -1.0, 0.0],
        atol=1.0e-6,
    )


def test_observation_layout_scaling_and_previous_action(deployment_contract: dict) -> None:
    """验证 28 维观测切片顺序、缩放和 previous_action 时序字段。"""
    builder = ObservationBuilder(deployment_contract)
    default_q = np.asarray(deployment_contract["default_joint_positions"]["values"])
    action = np.arange(8, dtype=np.float32)
    observation = builder.build(
        angular_velocity=np.array([4.0, -4.0, 2.0]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        joint_position_policy=default_q + 0.1,
        joint_velocity_policy=np.full(8, 2.0),
        previous_action=action,
    )
    np.testing.assert_allclose(observation[:3], [1.0, -1.0, 0.5])
    np.testing.assert_allclose(observation[3:6], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(observation[6:12], 0.1, atol=1.0e-7)
    np.testing.assert_allclose(observation[12:20], 0.1)
    np.testing.assert_array_equal(observation[20:28], action)
