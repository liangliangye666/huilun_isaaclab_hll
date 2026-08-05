"""按 Manifest 定义构造 proprioception 和 Encoder 历史窗口。

本模块把 MuJoCo 原始状态转换成训练时使用的观测语义，包括姿态四元数对应的
投影重力、相对默认关节位置、关节速度和上一策略动作。各项的拼接顺序、关节
顺序、维度与缩放均来自 Manifest，不能按阅读习惯自行交换。

.. rubric:: 观测拼接示例（L5A 机器人）

以当前 L5A 机器人为例，``proprioception_layout`` 定义了 5 个观测项，
按 Manifest 中的顺序拼接成一个 28 维向量：

======  ====================  =========  ==============================
序号    观测项名称             维度       含义
======  ====================  =========  ==============================
1       base_angular_velocity    3       基座三轴角速度（IMU 陀螺仪）
2       projected_gravity        3       重力在基座坐标系中的投影方向
3       leg_joint_position_relative 6    腿关节相对默认位置的角度偏差
4       joint_velocity_relative  8       所有关节的速度（8 个关节）
5       previous_action          8       上一策略周期的 Actor 原始输出
======  ====================  =========  ==============================

总维度 = 3 + 3 + 6 + 8 + 8 = 28

注意第 3 和第 4 项虽然都和"关节"相关，但它们的 ``order`` 字段可以不同：
- leg_joint_position_relative 只取 6 个腿关节
- joint_velocity_relative 取全部 8 个关节

每项还有一个 ``scale`` 参数，在拼接前对数值做缩放（如除以某个范围），
使不同量纲的信号进入神经网络前处于相近的数值范围。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from model import DeploymentError


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """归一化 MuJoCo sensor 输出的 ``wxyz`` 四元数。

    MuJoCo 的姿态传感器使用 ``wxyz`` 顺序。归一化可避免数值漂移造成模长偏离
    1，进而影响投影重力。输入和输出 shape 都是 ``(4,)``。
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"四元数 shape 应为 (4,)，实际为 {quaternion.shape}。")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("四元数必须有限且模长非零。")
    return quaternion / norm


def projected_gravity_from_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """把世界坐标系重力方向 ``[0, 0, -1]`` 投影到机器人基座坐标系。

    计算等价于 ``g_body = R(q).T @ g_world``。返回 ``(3,)`` float32 数组，
    其方向会随基座 roll/pitch 改变，是策略感知机身倾斜程度的主要来源。
    """
    w, x, y, z = normalize_quaternion_wxyz(quaternion_wxyz)
    return np.array(
        [
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ],
        dtype=np.float32,
    )


class ObservationHistory:
    """固定长度、oldest-to-newest 排列的观测历史。

    reset 后尚无真实历史，因此第一次 ``append(obs0)`` 会复制首帧填满窗口，
    与训练端 CircularBuffer 的首次写入语义一致。

    滑动窗口示意图（以 4 帧为例）：

    ::

        初始（reset 后，未初始化）:
        [0, 0, 0, 0]

        第一次 append(obs0) — 复制填满:
        [obs0, obs0, obs0, obs0]

        第二次 append(obs1) — 丢弃最老帧，末尾追加:
        [obs0, obs0, obs0, obs1]

        第三次 append(obs2):
        [obs0, obs0, obs1, obs2]

        第四次 append(obs3):
        [obs0, obs1, obs2, obs3]

        第五次 append(obs4):
        [obs1, obs2, obs3, obs4]   ← 最老的 obs0 被丢弃

    最新观测始终位于最后一行（索引 -1），Encoder 因而能使用与训练时相同的
    时间方向：第 0 行是最老的历史，第 N-1 行是最新的观测。
    """

    def __init__(self, length: int, observation_dim: int) -> None:
        """分配 ``[history_samples, obs_dim]`` 的 float32 历史数组。"""
        if length <= 0 or observation_dim <= 0:
            raise ValueError("历史长度和单帧观测维度必须大于零。")
        self._data = np.zeros((length, observation_dim), dtype=np.float32)
        self._initialized = False

    def reset(self) -> None:
        """清空历史并恢复到等待首帧写入的状态。"""
        self._data.fill(0.0)
        self._initialized = False

    def append(self, observation: np.ndarray) -> None:
        """首次写入复制填满窗口，之后丢弃最老帧并追加最新帧。"""
        observation = np.asarray(observation, dtype=np.float32)
        expected_shape = (self._data.shape[1],)
        if observation.shape != expected_shape:
            raise ValueError(f"观测 shape 应为 {expected_shape}，实际为 {observation.shape}。")
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("观测历史输入包含 NaN 或 Inf。")
        if not self._initialized:
            self._data[:] = observation
            self._initialized = True
            return

        # 源和目标区域重叠，显式 copy 可避免 NumPy 赋值时覆盖尚未读取的数据。
        self._data[:-1] = self._data[1:].copy()
        self._data[-1] = observation

    def batched(self) -> np.ndarray:
        """返回 ONNX 需要的连续数组 ``[1, history_samples, obs_dim]``。"""
        if not self._initialized:
            raise RuntimeError("观测历史尚未初始化。")
        return np.ascontiguousarray(self._data[None, ...], dtype=np.float32)


class ObservationBuilder:
    """按 ``proprioception_layout`` 的条目顺序拼接一帧本体观测。

    当前实现支持基座角速度、投影重力、指定关节的相对位置、指定关节速度和
    上一动作。Manifest 中每个 layout 条目的 ``order`` 决定取哪些关节以及排列，
    ``scale`` 决定送入网络前的缩放，``size`` 用于检查该项实际输出维度。
    """

    SUPPORTED_TERMS = {
        "base_angular_velocity",
        "projected_gravity",
        "joint_position_relative",
        "leg_joint_position_relative",
        "joint_velocity_relative",
        "previous_action",
    }

    def __init__(self, deployment: dict[str, Any]) -> None:
        """解析策略关节顺序、默认关节位置和观测 layout。"""
        self.policy_order = list(deployment["policy_action_order"])
        self.action_dim = len(self.policy_order)
        if len(set(self.policy_order)) != self.action_dim:
            raise DeploymentError("policy_action_order 中存在重复关节名。")
        self.policy_index = {name: index for index, name in enumerate(self.policy_order)}
        self.proprioception_dim = int(deployment["proprioception_dim"])

        defaults = deployment["default_joint_positions"]
        # load_deployment 已验证 default_joint_positions.order 与策略顺序完全一致。
        self.default_q = np.asarray(defaults["values"], dtype=np.float32)
        if self.default_q.shape != (self.action_dim,):
            raise DeploymentError(
                f"default_joint_positions.values shape 应为 {(self.action_dim,)}，实际为 {self.default_q.shape}。"
            )

        self.layout = list(deployment["proprioception_layout"])
        unsupported = [item.get("name") for item in self.layout if item.get("name") not in self.SUPPORTED_TERMS]
        if unsupported:
            raise DeploymentError(f"当前 sim2sim 不支持这些观测项：{unsupported}")

    def _indices(self, item: dict[str, Any]) -> np.ndarray:
        """把观测项中的关节名转换成 ``policy_order`` 索引。"""
        order = item.get("order", self.policy_order)
        try:
            return np.asarray([self.policy_index[name] for name in order], dtype=np.int64)
        except KeyError as error:
            raise DeploymentError(
                f"观测项 {item['name']!r} 引用了 policy_action_order 中不存在的关节 {error.args[0]!r}。"
            ) from error

    @staticmethod
    def _scaled(values: np.ndarray, scale: Any, term_name: str) -> np.ndarray:
        """应用标量或逐维 scale，并把广播错误转换成部署配置错误。"""
        scale_array = np.asarray(scale, dtype=np.float32)
        try:
            return np.asarray(values, dtype=np.float32) * scale_array
        except ValueError as error:
            raise DeploymentError(f"观测项 {term_name!r} 的 scale 无法广播到数据 shape。") from error

    def build(
        self,
        angular_velocity: np.ndarray,
        quaternion_wxyz: np.ndarray,
        joint_position_policy: np.ndarray,
        joint_velocity_policy: np.ndarray,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        """从当前 MuJoCo 状态构造一帧连续的 float32 proprioception。

        ``joint_position_policy``、``joint_velocity_policy`` 和
        ``previous_action`` 必须都采用 ``policy_action_order``。上一动作指上一个
        策略周期的 Actor 原始输出，而不是经过动作延迟后的动作或 PD 力矩。
        """
        angular_velocity = np.asarray(angular_velocity, dtype=np.float32)
        joint_position = np.asarray(joint_position_policy, dtype=np.float32)
        joint_velocity = np.asarray(joint_velocity_policy, dtype=np.float32)
        previous_action = np.asarray(previous_action, dtype=np.float32)
        if angular_velocity.shape != (3,):
            raise ValueError(f"基座角速度 shape 应为 (3,)，实际为 {angular_velocity.shape}。")
        expected_action_shape = (self.action_dim,)
        for name, value in (
            ("joint_position", joint_position),
            ("joint_velocity", joint_velocity),
            ("previous_action", previous_action),
        ):
            if value.shape != expected_action_shape:
                raise ValueError(f"{name} shape 应为 {expected_action_shape}，实际为 {value.shape}。")

        gravity = projected_gravity_from_quaternion(quaternion_wxyz)
        # 按 Manifest 原顺序逐项追加；顺序变化即使维度相同也会改变网络语义。
        # 拼接示例（L5A, 28 维）：
        #   terms[0] = base_angular_velocity       → 3 维（角速度 × scale）
        #   terms[1] = projected_gravity           → 3 维（无需 scale）
        #   terms[2] = leg_joint_position_relative → 6 维（(q - default_q) × scale）
        #   terms[3] = joint_velocity_relative     → 8 维（dq × scale）
        #   terms[4] = previous_action             → 8 维（action × scale）
        #   → concatenate → [28] → 与 proprioception_dim 比对 → 返回
        terms: list[np.ndarray] = []
        for item in self.layout:
            name = item["name"]
            if name == "base_angular_velocity":
                values = angular_velocity
            elif name == "projected_gravity":
                values = gravity
            elif name in {"joint_position_relative", "leg_joint_position_relative"}:
                indices = self._indices(item)
                values = joint_position[indices] - self.default_q[indices]
            elif name == "joint_velocity_relative":
                values = joint_velocity[self._indices(item)]
            elif name == "previous_action":
                values = previous_action[self._indices(item)]
            else:  # __init__ 已拒绝未知项；保留分支防止运行时 layout 被外部修改。
                raise DeploymentError(f"不支持的观测项：{name}")

            values = np.ravel(self._scaled(values, item.get("scale", 1.0), name))
            expected_size = int(item["size"])
            if values.size != expected_size:
                raise DeploymentError(f"观测项 {name!r} 应产生 {expected_size} 维，实际产生 {values.size} 维。")
            terms.append(values)

        # 例如当前 L5A layout 为 3 + 3 + 6 + 8 + 8 = 28 维；其他机型动态计算。
        observation = np.concatenate(terms).astype(np.float32, copy=False)
        if observation.shape != (self.proprioception_dim,):
            raise DeploymentError(
                f"proprioception_layout 实际拼接为 {observation.size} 维，"
                f"但 proprioception_dim={self.proprioception_dim}。"
            )
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("构造的 proprioception 包含 NaN 或 Inf。")
        return np.ascontiguousarray(observation)
