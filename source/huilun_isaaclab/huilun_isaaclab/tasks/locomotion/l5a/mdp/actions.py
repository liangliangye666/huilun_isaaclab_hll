# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 专用动作项。

WF 任务的动作语义是六个腿关节位置目标加两个轮关节速度目标。本模块只扩展
腿部位置动作的“关节零位误差”；轮速动作仍使用 IsaacLab 标准速度 ActionTerm。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

'''
核心区别：
    JointPositionAction（标准版）的关节零位是 USD/URDF 的出厂设计值，所有环境一样。
    RandomizedDefaultJointPositionAction（L5A 版）在标准版基础上，为每个环境的每个关节加一个持久随机偏差，模拟真机的编码器零位校准误差。
'''

class RandomizedDefaultJointPositionAction(JointPositionAction):
    """带逐环境、整次运行持久关节零位误差的位置动作。

    初始化时，每个环境和每个被控腿关节只采样一次零位误差，而不是每回合重采样。
    同一个误差同时写入：

    * ``self._offset``：把归一化腿动作映射为位置目标时使用的中心；
    * ``asset.data.default_joint_pos``：相对关节位置观测和 reset 默认姿态的基准。

    这样动作目标、策略观测和重置姿态看到的是同一个“校准后零位”，复现旧 L5A
    的关节编码器零位/装配偏差语义，也避免三套参考值悄然不一致。回合 reset 中
    额外的 joint-position 随机扰动是另一层临时初态扰动，不会覆盖这里的持久零位。
    """

    cfg: RandomizedDefaultJointPositionActionCfg

    def __init__(self, cfg: RandomizedDefaultJointPositionActionCfg, env: ManagerBasedEnv):
        # ① 安全检查：必须使用 default offset，否则动作/观测/reset 的零位会不一致
        if not cfg.use_default_offset:
            raise ValueError(
                "RandomizedDefaultJointPositionAction requires use_default_offset=True "
                "so action targets, observations, and reset defaults share one reference."
            )
        # ② 调用父类构造函数，建立基础动作映射
        super().__init__(cfg, env)
        lower, upper = cfg.default_offset_range
        if lower > upper:
            raise ValueError(f"Invalid default_offset_range: {cfg.default_offset_range}")

        # ③ 为每个环境的每个腿关节独立采样零位误差（均匀分布）
        zero_error = torch.empty_like(self._offset).uniform_(lower, upper)
        # ④ 把误差写入 _offset → 动作映射使用校准后的零位
        self._offset.add_(zero_error)
        # ⑤ 同步到 data.default_joint_pos → 相对位置观测 + reset 默认值也使用同一零位
        self._asset.data.default_joint_pos[:, self._joint_ids] = self._offset
        # ⑥ 暴露原始误差供调试/记录
        self.default_offset_error = zero_error


@configclass
class RandomizedDefaultJointPositionActionCfg(JointPositionActionCfg):
    """L5A 持久关节零位随机化配置。

    ``default_offset_range`` 以弧度为单位，并对每个环境、每个选中关节独立采样。
    """

    class_type: type[ActionTerm] = RandomizedDefaultJointPositionAction
    default_offset_range: tuple[float, float] = (-0.05, 0.05)
