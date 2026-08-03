"""动作延迟和混合 PD 控制。

策略输出始终是 policy_order 下的 8 维 action：前 6 维是腿部位置增量，
后 2 维是轮子速度目标。MuJoCo 执行器可能使用 hardware_order，因此本模块
负责把 policy_order 下的目标/力矩转换到硬件顺序。
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ActionDelayBuffer:
    """按 MuJoCo 物理步计数的共享动作延迟。

    训练中延迟范围是 0..6 个物理步；这里复现同一范围。注意该延迟作用在
    decimation 内部的 5 ms 物理步，而不是 10 ms 策略步。
    """

    def __init__(self, delay_steps: int, action_dim: int = 8) -> None:
        if not 0 <= delay_steps <= 6:
            raise ValueError("Action delay must be in the training range 0..6 physics steps.")
        self.delay_steps = delay_steps
        self.action_dim = action_dim
        self._queue: deque[np.ndarray] = deque()
        self.reset()

    def reset(self) -> None:
        self._queue.clear()
        for _ in range(self.delay_steps):
            self._queue.append(np.zeros(self.action_dim, dtype=np.float32))

    def apply(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape {(self.action_dim,)}, got {action.shape}.")
        if self.delay_steps == 0:
            return action.copy()
        self._queue.append(action.copy())
        return self._queue.popleft()
    '''
    用 Python 的 deque（双端队列）实现的 FIFO 延迟队列。
    训练时随机加了 0~6 步的延迟来模拟真实硬件的通信延迟，仿真时也要保持一致。

    deque 是 collections 模块中的双端队列，append() 在右边加，popleft() 从左边取，天然支持 FIFO。
    '''


class MixedPDController:
    """policy_order 下的 6 个腿部位置环 + 2 个轮子速度环。

    策略输出 action 的前 6 维是相对于默认关节角的腿部位置偏移（scale=0.25），
    后 2 维是轮子速度目标（scale=1.0）。轮子的 kp 为 0，因此只用速度环控制。
    """

    def __init__(self, deployment: dict) -> None:
        """从部署契约中提取 PD 参数、缩放因子和关节顺序映射。

        Args:
            deployment: manifest 中的 deployment 字典，包含 joint_control、
                        policy_action_semantics、default_joint_positions 等字段。
        """
        self.policy_order = list(deployment["policy_action_order"])
        self.hardware_order = list(deployment["hardware_dof_order"])
        self.policy_to_hardware = np.asarray(deployment["policy_actions_to_hardware_indices"], dtype=np.int64)
        self.default_q = np.asarray(deployment["default_joint_positions"]["values"], dtype=np.float64)
        control = deployment["joint_control"]
        self.kp = np.asarray(control["stiffness"], dtype=np.float64)
        self.kd = np.asarray(control["damping"], dtype=np.float64)
        self.effort_limits = np.asarray(control["effort_limits"], dtype=np.float64)
        semantics = deployment["policy_action_semantics"]
        self.leg_scale = float(semantics["leg_position"]["scale"])
        self.wheel_scale = float(semantics["wheel_velocity"]["scale"])

    def targets(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把策略 action 转成期望位置和期望速度。

        腿部：`q_target = default_q + 0.25 * action`。
        轮子：`dq_target = 1.0 * action`，位置目标保持 default_q 但 kp 为 0。

        Args:
            action: 策略输出的 8 维原始动作向量，policy_order。

        Returns:
            (position_target, velocity_target) 两个 `(8,)` 数组，均为 policy_order。
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action.shape}.")
        position_target = self.default_q.copy()
        velocity_target = np.zeros(8, dtype=np.float64)
        position_target[:6] += self.leg_scale * action[:6]      # 腿：位置控制
        velocity_target[6:] = self.wheel_scale * action[6:]     # 轮：速度控制
        return position_target, velocity_target

    def compute_policy_torque(
        self, action: np.ndarray, joint_position_policy: np.ndarray, joint_velocity_policy: np.ndarray
    ) -> np.ndarray:
        """在 policy_order 下计算并裁剪 PD 力矩。

        公式：torque = kp * (q_target - q) + kd * (dq_target - dq)，然后裁剪到 effort_limits。
        腿部的 kp/kd 非零，靠位置误差驱动；轮子的 kp=0，仅靠速度误差驱动。

        Args:
            action: 策略输出的 8 维动作（policy_order）。
            joint_position_policy: 当前关节位置 `(8,)`，policy_order。
            joint_velocity_policy: 当前关节速度 `(8,)`，policy_order。

        Returns:
            policy_order 下的 8 维力矩，已裁剪到 effort_limits 范围内。
        """
        q = np.asarray(joint_position_policy, dtype=np.float64)
        dq = np.asarray(joint_velocity_policy, dtype=np.float64)
        if q.shape != (8,) or dq.shape != (8,):
            raise ValueError(f"Expected joint state shapes (8,), got q={q.shape}, dq={dq.shape}.")
        q_target, dq_target = self.targets(action)
        # 混合控制统一写成 PD 形式：腿依赖位置误差，轮子依赖速度误差。
        torque = self.kp * (q_target - q) + self.kd * (dq_target - dq)
        torque = np.clip(torque, -self.effort_limits, self.effort_limits)
        if not np.all(np.isfinite(torque)):
            raise FloatingPointError("Controller produced NaN or Inf torque.")
        return torque

    def to_hardware_order(self, policy_values: np.ndarray) -> np.ndarray:
        """把 policy_order 下的 8 维向量重排成 MuJoCo actuator/hardware 顺序。

        Args:
            policy_values: policy_order 下的 `(8,)` 向量（如力矩）。

        Returns:
            hardware_order 下的 `(8,)` 向量，可直接写入 data.ctrl。
        """
        policy_values = np.asarray(policy_values)
        if policy_values.shape != (8,):
            raise ValueError(f"Expected policy vector shape (8,), got {policy_values.shape}.")
        return policy_values[self.policy_to_hardware]
