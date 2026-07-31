# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 任务的额外终止条件。

终止函数只返回逐环境布尔掩码；是否计为失败终止、如何写入 rollout 以及 timeout
bootstrap 由 IsaacLab TerminationManager 和 RSL-RL wrapper 统一处理。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_out_of_limits(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """任一动作维度达到极端阈值时终止对应环境。

    这是幅值异常保护，不是正常的电机软/硬限位：它监视 ActionManager 收到的
    动作向量，用于尽早截断策略输出幅值爆炸的异常回合。RslRlVecEnvWrapper 会先
    把动作裁到同一个 threshold，因此必须使用 ``>=``；若写成 ``>``，到达
    ActionManager 的有限数值永远无法触发该条件。
    """
    # ① 对每个环境取动作向量的最大绝对值 [N]
    # ② 判断是否 ≥ threshold（必须用 >=，因为 wrapper 已裁到 threshold）
    return torch.max(torch.abs(env.action_manager.action), dim=1).values >= threshold
