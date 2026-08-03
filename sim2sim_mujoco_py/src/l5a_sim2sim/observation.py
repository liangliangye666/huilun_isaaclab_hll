"""从 MuJoCo 状态构建策略 proprioception。

当前 L5A WF-Flat 部署契约要求 28 维观测：
角速度 3、投影重力 3、腿部相对关节位置 6、全部关节速度 8、上一帧 action 8。
这里不包含 gait phase、高度命令或横向速度命令。
"""

from __future__ import annotations

import numpy as np


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """归一化 MuJoCo sensor 给出的 wxyz 四元数。

    防止传感器数值漂移导致四元数模长偏离 1，从而影响投影重力计算精度。

    Args:
        quaternion: `(4,)` 数组，wxyz 顺序的四元数。

    Returns:
        归一化后的 `(4,)` 四元数，模长为 1。
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quaternion.shape}.")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("Quaternion must be finite and non-zero.")
    return quaternion / norm


def projected_gravity_from_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """将世界重力方向 [0, 0, -1] 旋转到机器人基座坐标系中。

    通过四元数旋转公式：g_body = R(q)^T * [0, 0, -1]，其中 R(q) 是四元数到旋转矩阵的转换。
    这等价于用旋转矩阵的第三列取负，因为世界重力在世界坐标系中是 (0, 0, -1)。

    Args:
        quaternion_wxyz: 机器人基座的姿态四元数 `(4,)`，wxyz 顺序。

    Returns:
        `(3,)` 浮点数组，基座坐标系下的重力方向分量。
    """
    w, x, y, z = normalize_quaternion_wxyz(quaternion_wxyz)
    gravity = np.array(
        [
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ],
        dtype=np.float32,
    )
    return gravity


class ObservationBuilder:
    """根据 Manifest 中的 layout 和 scale 构建 28 维 proprioception。

    观测布局固定为 5 个字段的顺序拼接：
    1. base_angular_velocity (3) × 0.25
    2. projected_gravity (3) × 1.0
    3. leg_joint_position_relative (6) × 1.0
    4. joint_velocity_relative (8) × 0.05
    5. previous_action (8) × 1.0
    """

    def __init__(self, deployment: dict) -> None:
        """从部署契约中提取默认关节位置和观测缩放因子。

        Args:
            deployment: manifest 中的 deployment 字典，包含 proprioception_layout
                        和 default_joint_positions。
        """
        self.default_q = np.asarray(deployment["default_joint_positions"]["values"], dtype=np.float32)
        layout = deployment["proprioception_layout"]
        expected_names = [
            "base_angular_velocity",
            "projected_gravity",
            "leg_joint_position_relative",
            "joint_velocity_relative",
            "previous_action",
        ]
        actual_names = [item["name"] for item in layout]
        if actual_names != expected_names:
            raise ValueError(f"Unexpected proprioception layout: {actual_names}.")
        self.scales = {item["name"]: float(item["scale"]) for item in layout}

    def build(
        self,
        angular_velocity: np.ndarray,
        quaternion_wxyz: np.ndarray,
        joint_position_policy: np.ndarray,
        joint_velocity_policy: np.ndarray,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        """返回 shape `(28,)` 的 float32 连续数组，供 Actor 和历史缓冲使用。

        Args:
            angular_velocity: 基座角速度 `(3,)`，机器人坐标系。
            quaternion_wxyz: 基座姿态四元数 `(4,)`，wxyz 顺序。
            joint_position_policy: 关节位置 `(8,)`，policy_order。
            joint_velocity_policy: 关节速度 `(8,)`，policy_order。
            previous_action: 上一策略步的动作 `(8,)`，policy_order。

        Returns:
            `(28,)` float32 C 连续数组，各字段已按对应 scale 缩放。
        """
        angular_velocity = np.asarray(angular_velocity, dtype=np.float32)
        joint_position_policy = np.asarray(joint_position_policy, dtype=np.float32)
        joint_velocity_policy = np.asarray(joint_velocity_policy, dtype=np.float32)
        previous_action = np.asarray(previous_action, dtype=np.float32)
        if angular_velocity.shape != (3,):
            raise ValueError(f"Expected angular velocity shape (3,), got {angular_velocity.shape}.")
        for name, value in (
            ("joint position", joint_position_policy),
            ("joint velocity", joint_velocity_policy),
            ("previous action", previous_action),
        ):
            if value.shape != (8,):
                raise ValueError(f"Expected {name} shape (8,), got {value.shape}.")
        # 拼接顺序必须和训练端/Manifest 完全一致，否则 Encoder 和 Actor 输入语义会错位。
        observation = np.concatenate(
            (
                angular_velocity * self.scales["base_angular_velocity"],                                        # 3维：角速度，基座角速度 (x, y, z)
                projected_gravity_from_quaternion(quaternion_wxyz) * self.scales["projected_gravity"],          # 3维：投影重力，重力在基座坐标系下的投影
                (joint_position_policy[:6] - self.default_q[:6]) * self.scales["leg_joint_position_relative"],  # 6维：腿关节相对位置，6个腿关节相对默认位置的偏差
                joint_velocity_policy * self.scales["joint_velocity_relative"],                                 # 8维：关节速度，全部8个关节的速度
                previous_action * self.scales["previous_action"],                                               # 8维：上一帧动作，上一帧的策略输出动作
            )   # 总计: 3+3+6+8+8 = 28维
        ).astype(np.float32, copy=False)
        if observation.shape != (28,):
            raise RuntimeError(f"Constructed observation has shape {observation.shape}, expected (28,).")
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("Constructed observation contains NaN or Inf.")
        return np.ascontiguousarray(observation)
