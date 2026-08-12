# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 任务的额外终止条件。

终止函数只返回逐环境布尔掩码；是否计为失败终止、如何写入 rollout 以及 timeout
bootstrap 由 IsaacLab TerminationManager 和 RSL-RL wrapper 统一处理。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_out_of_limits(env: ManagerBasedRLEnv, threshold: float | Sequence[float]) -> torch.Tensor:
    """任一动作维度达到极端阈值时终止对应环境。

    这是幅值异常保护，不是正常的电机软/硬限位：它监视 ActionManager 收到的
    动作向量，用于尽早截断策略输出幅值爆炸的异常回合。``threshold`` 可以是
    标量，也可以按动作维度分别指定，后者用于同时容纳腿位置与轮速度两种语义。
    RslRlVecEnvWrapper 会先裁剪动作，因此必须使用 ``>=``。
    """
    action = env.action_manager.action
    if isinstance(threshold, (int, float)):
        return torch.max(torch.abs(action), dim=1).values >= float(threshold)

    threshold_key = tuple(float(value) for value in threshold)
    if len(threshold_key) != action.shape[1]:
        raise ValueError(f"Per-dimension action threshold has {len(threshold_key)} values, expected {action.shape[1]}.")
    cache = getattr(env, "_l5a_action_limit_threshold_cache", None)
    if cache is None or cache[0] != threshold_key:
        threshold_tensor = torch.tensor(threshold_key, device=action.device, dtype=action.dtype).unsqueeze(0)
        cache = (threshold_key, threshold_tensor)
        env._l5a_action_limit_threshold_cache = cache
    return torch.any(torch.abs(action) >= cache[1], dim=1)
