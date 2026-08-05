"""通用位置/速度 PD 控制、动作语义解析和物理步动作延迟。

Actor 输出的 ``policy_action_order`` 必须与 ``hardware_dof_order`` 完全相同。
控制器按这一套统一顺序把 action 解释为位置或速度目标，直接计算并输出
MuJoCo/硬件可用的力矩，不再执行任何索引重排。

.. rubric:: 核心概念：统一关节顺序

当前 L5A 契约为 ``[左三腿, 左轮, 右三腿, 右轮]``。启动时会验证
Manifest 中的策略、硬件与映射字段均为这一直通契约；任意一项不一致都会在
仿真启动时报错，避免静默地把力矩写给错误关节。

.. rubric:: PD 控制通俗解释

PD 控制就像开车时的"方向盘+油门"组合：

- **P（比例项）**：你离目标位置越远，用的力越大。就像看到前方车距很大时，你会踩更深的油门。
  公式：``kp * (q_target - q_current)``
- **D（阻尼项）**：你当前速度越快，需要反向的力越大来防止超调。就像快撞到前车时你会刹车。
  公式：``kd * (0 - dq)`` （位置模式，目标速度始终是 0）
  或：``kd * (dq_target - dq)`` （速度模式）

最终力矩 = P 项 + D 项，然后裁剪到关节的 effort_limits 范围内。
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from model import DeploymentError


class ActionDelayBuffer:
    """用 FIFO 队列复现所有关节共享的动作延迟。

    延迟单位是 MuJoCo 物理步，而不是策略步。它在 decimation 内部每个物理步调用
    一次，所以当物理周期为 5 ms 时，``delay_steps=2`` 表示约 10 ms 延迟。
    ``deque.append`` 从队尾加入新动作，``popleft`` 从队首取出旧动作，形成 FIFO。
    """

    def __init__(
        self,
        delay_steps: int,
        action_dim: int,
        allowed_range: tuple[int, int] | list[int],
    ) -> None:
        minimum, maximum = (int(value) for value in allowed_range)
        if not minimum <= delay_steps <= maximum:
            raise ValueError(f"动作延迟必须位于训练范围 [{minimum}, {maximum}]，收到 {delay_steps}。")
        self.delay_steps = int(delay_steps)
        self.action_dim = int(action_dim)
        self._queue: deque[np.ndarray] = deque()
        self.reset()

    def reset(self) -> None:
        """清空队列，并用零动作填充延迟启动阶段。"""
        self._queue.clear()
        for _ in range(self.delay_steps):
            self._queue.append(np.zeros(self.action_dim, dtype=np.float32))

    def apply(self, action: np.ndarray) -> np.ndarray:
        """压入当前动作并返回延迟指定物理步数后的动作副本。"""
        action = np.asarray(action, dtype=np.float32)
        expected_shape = (self.action_dim,)
        if action.shape != expected_shape:
            raise ValueError(f"action shape 应为 {expected_shape}，实际为 {action.shape}。")
        if self.delay_steps == 0:
            return action.copy()
        self._queue.append(action.copy())
        return self._queue.popleft()


class GenericPDController:
    """根据 Manifest 为每个关节执行位置环或速度环。

    位置模式：
        q_target = default_q + action_scale * action
        torque = kp * (q_target - q) + kd * (0 - dq)

    速度模式：
        dq_target = action_scale * action
        torque = kd * (dq_target - dq)

    最终所有力矩统一裁剪到各关节的 effort_limits。位置关节和速度关节可以任意
    排列，控制器不依赖固定索引或关节数量。
    """

    def __init__(self, deployment: dict[str, Any]) -> None:
        """按统一关节顺序读取默认位置、PD 参数、动作缩放和控制模式。

        初始化流程概述：
        1. 读取 policy_action_order 和 hardware_dof_order，验证两套列表逐项相同
        2. 从 Manifest 直接读取同序的 default_q、kp、kd、effort_limits、控制模式
        3. 解析 policy_action_semantics：每个关节的 action_scale 和是否使用默认偏移
        4. 构建 position_mask/velocity_mask 布尔数组，用于后续分支计算
        """
        self.joint_order = list(deployment["policy_action_order"])
        hardware_order = list(deployment["hardware_dof_order"])
        self.action_dim = len(self.joint_order)
        if len(set(self.joint_order)) != self.action_dim:
            raise DeploymentError("policy_action_order 中存在重复关节名。")
        if hardware_order != self.joint_order:
            raise DeploymentError("hardware_dof_order 必须与 policy_action_order 逐项相同；当前运行时不做关节重排。")

        joint_index = {name: index for index, name in enumerate(self.joint_order)}

        # ---- 直接读取同序参数 ----
        # load_deployment 已保证这些数组与 joint_order 逐项相同，不再按名字重排。
        defaults = deployment["default_joint_positions"]
        control = deployment["joint_control"]
        self.default_q = np.asarray(defaults["values"], dtype=np.float64)
        self.modes = list(control["modes"])
        self.kp = np.asarray(control["stiffness"], dtype=np.float64)
        self.kd = np.asarray(control["damping"], dtype=np.float64)
        self.effort_limits = np.asarray(control["effort_limits"], dtype=np.float64)
        for name, values in (
            ("default_joint_positions.values", self.default_q),
            ("joint_control.modes", self.modes),
            ("joint_control.stiffness", self.kp),
            ("joint_control.damping", self.kd),
            ("joint_control.effort_limits", self.effort_limits),
        ):
            if len(values) != self.action_dim:
                raise DeploymentError(f"{name} 长度必须为 {self.action_dim}，实际为 {len(values)}。")

        # ---- 验证控制模式 ----
        # 目前只支持 position（位置控制）和 velocity（速度控制）两种模式
        unsupported_modes = sorted(set(self.modes) - {"position", "velocity"})
        if unsupported_modes:
            raise DeploymentError(f"当前控制器不支持这些控制模式：{unsupported_modes}")
        # position_mask: True 表示该关节用位置模式，False 表示用速度模式
        self.position_mask = np.asarray([mode == "position" for mode in self.modes])
        self.velocity_mask = ~self.position_mask

        # ---- 解析动作语义分组 ----
        # 每个语义分组定义了一组关节的 scale（缩放系数），以及位置目标是否叠加默认角度。
        # 例如：腿关节的 action_scale=0.5 表示 Actor 输出 [-1,1] 映射为偏离默认角 ±0.5 rad。
        self.action_scale = np.full(self.action_dim, np.nan, dtype=np.float64)
        self.uses_default_offset = np.zeros(self.action_dim, dtype=bool)
        semantics = deployment["policy_action_semantics"]
        for group_name, group in semantics.items():
            scale = float(group["scale"])
            use_default = bool(group.get("uses_default_offset", True))
            for joint_name in group["joints"]:
                if joint_name not in joint_index:
                    raise DeploymentError(f"动作语义 {group_name!r} 引用了未知关节 {joint_name!r}。")
                index = joint_index[joint_name]
                if np.isfinite(self.action_scale[index]):
                    raise DeploymentError(f"关节 {joint_name!r} 被多个动作语义分组重复定义。")
                self.action_scale[index] = scale
                self.uses_default_offset[index] = use_default
        missing_semantics = [
            self.joint_order[index] for index, scale in enumerate(self.action_scale) if not np.isfinite(scale)
        ]
        if missing_semantics:
            raise DeploymentError(f"这些关节没有动作缩放定义：{missing_semantics}")

    def targets(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把原始 action 转换为统一关节顺序下的位置和速度目标。

        位置模式通常使用 ``q_target = baseline + scale * action``；baseline 由
        ``uses_default_offset`` 决定是默认关节角还是零。速度模式使用
        ``dq_target = scale * action``。未采用的目标数组分量保持为零。
        """
        action = np.asarray(action, dtype=np.float64)
        expected_shape = (self.action_dim,)
        if action.shape != expected_shape:
            raise ValueError(f"action shape 应为 {expected_shape}，实际为 {action.shape}。")

        position_target = np.zeros(self.action_dim, dtype=np.float64)
        position_baseline = np.where(self.uses_default_offset, self.default_q, 0.0)
        position_target[self.position_mask] = (
            position_baseline[self.position_mask] + self.action_scale[self.position_mask] * action[self.position_mask]
        )
        velocity_target = np.zeros(self.action_dim, dtype=np.float64)
        velocity_target[self.velocity_mask] = self.action_scale[self.velocity_mask] * action[self.velocity_mask]
        return position_target, velocity_target

    def compute_torque(
        self,
        action: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> np.ndarray:
        """按控制模式计算并裁剪统一关节顺序下的关节力矩。

        位置模式使用 ``kp*(q_target-q) + kd*(0-dq)``，速度模式使用
        ``kd*(dq_target-dq)``。计算后逐关节裁剪到 ``effort_limits``。
        """
        q = np.asarray(joint_position, dtype=np.float64)
        dq = np.asarray(joint_velocity, dtype=np.float64)
        expected_shape = (self.action_dim,)
        if q.shape != expected_shape or dq.shape != expected_shape:
            raise ValueError(f"关节状态 shape 应为 {expected_shape}，实际 q={q.shape}, dq={dq.shape}。")

        q_target, dq_target = self.targets(action)
        torque = np.zeros(self.action_dim, dtype=np.float64)
        torque[self.position_mask] = self.kp[self.position_mask] * (
            q_target[self.position_mask] - q[self.position_mask]
        ) + self.kd[self.position_mask] * (0.0 - dq[self.position_mask])
        torque[self.velocity_mask] = self.kd[self.velocity_mask] * (
            dq_target[self.velocity_mask] - dq[self.velocity_mask]
        )
        torque = np.clip(torque, -self.effort_limits, self.effort_limits)
        if not np.all(np.isfinite(torque)):
            raise FloatingPointError("PD 控制器输出包含 NaN 或 Inf。")
        return torque
