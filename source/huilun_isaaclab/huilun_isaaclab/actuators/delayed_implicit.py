# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""带逐环境随机命令延迟的隐式 PD 执行器。

延迟位于 ActionTerm 生成目标之后、ImplicitActuator 把目标交给 PhysX 之前。
它模拟通信/驱动链路让“目标晚若干物理步到达”，不延迟关节状态反馈，也不改变
上层策略的 decimation。位置、速度和力矩目标分别缓存，但同一环境共享相同 lag。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class DelayedImplicitActuator(ImplicitActuator):
    """先延迟目标，再交由 IsaacLab 隐式执行器/PhysX 计算驱动力。

    ``DelayBuffer`` 的 batch 维对应并行环境，时间延迟以调用 ``compute()`` 的
    物理步为单位。每个环境可以有不同延迟，但该环境的位置、速度和力矩目标使用
    同一个随机延迟，保证混合控制量仍来自同一控制时刻。
    """

    cfg: DelayedImplicitActuatorCfg

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # 三类目标可能为 None，分别建缓冲区可保持 ArticulationActions 的可选语义。
        # max_delay 决定环形历史的最大长度；实际 lag 在 reset() 中逐环境设置。
        self._position_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._velocity_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._effort_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int] | slice | None) -> None:
        """重置指定环境，并为其重新采样一个共享延迟。

        先调用父类重置执行器内部状态；随后设置 lag，再清空 DelayBuffer 历史，
        防止上一回合尚未送达的目标泄漏到新回合。``env_ids=None`` 表示全部环境，
        slice 和显式索引都只影响被重置的 batch 行。
        """
        # ① 父类先重置执行器内部状态
        super().reset(env_ids)
        # ② 计算需要重置的环境数量
        if env_ids is None:
            num_envs = self._num_envs
        elif isinstance(env_ids, slice):
            num_envs = len(range(*env_ids.indices(self._num_envs)))
        else:
            num_envs = len(env_ids)
        # ③ 为每个重置环境采样随机延迟（在 [min_delay, max_delay] 范围内）
        delays = torch.randint(
            self.cfg.min_delay,
            self.cfg.max_delay + 1,
            (num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # ④ 同一个 delays 写入三个缓冲区 → 位置/速度/力矩目标共享同一延迟
        for buffer in (self._position_delay, self._velocity_delay, self._effort_delay):
            buffer.set_time_lag(delays, env_ids)
            buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """延迟当前目标，并在延迟之后执行标准隐式执行器计算。

        此函数由关节体在每个物理仿真步调用。上层策略在 decimation 期间保持目标
        不变，但 DelayBuffer 仍按每个物理步推进。函数原位替换
        ``control_action`` 中存在的目标。reset 后历史尚未积满 lag 时，DelayBuffer
        返回当前最新目标而不是补零，避免新回合起步阶段接收到无意义目标。

        随后父类用当前 ``joint_pos/joint_vel`` 和增益更新近似
        ``computed_effort/applied_effort``，供奖励与诊断读取；真正的隐式 PD
        仍由 PhysX 根据延迟后的 position/velocity/effort target 执行。
        """
        # ① 位置目标：通过 DelayBuffer 延迟指定物理步数后输出
        if control_action.joint_positions is not None:
            control_action.joint_positions = self._position_delay.compute(control_action.joint_positions)
        # ② 速度目标：同上，使用独立的延迟缓冲区
        if control_action.joint_velocities is not None:
            control_action.joint_velocities = self._velocity_delay.compute(control_action.joint_velocities)
        # ③ 力矩目标：同上
        if control_action.joint_efforts is not None:
            control_action.joint_efforts = self._effort_delay.compute(control_action.joint_efforts)
        # ④ 延迟后的目标交给父类 ImplicitActuator 计算 PhysX 驱动力
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """配置 :class:`DelayedImplicitActuator` 的延迟范围。

    ``min_delay`` 与 ``max_delay`` 均包含端点，单位均为物理仿真步；实际值在每次
    环境 reset 时独立采样，而不是在每个策略步抖动。
    """

    class_type: type = DelayedImplicitActuator
    min_delay: int = 0
    max_delay: int = 0
