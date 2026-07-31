# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Huilun IsaacLab 扩展的 Python 入口。

导入本包并不只是加载符号，还会触发两类注册副作用：

1. ``tasks`` 会递归发现任务包，并执行各任务中的 ``gym.register``；
2. ``ui_extension_example`` 暴露 Omniverse 扩展入口，由 Isaac Sim 扩展管理器调用。

训练脚本必须先创建 ``SimulationApp``，再导入本包。原因是下列模块最终会依赖
``omni``/``pxr``，普通 Python 解释器无法独立完成这段初始化。

建议接手阅读顺序是：``l5a/__init__.py``（任务入口）→ ``wf_flat_env_cfg.py``
（完整训练配置）→ ``assets``/``mdp``（机器人和各 Manager term）→
``agents/rsl_rl_ppo_cfg.py``（算法超参数）→ ``learning/rsl_rl``（Encoder/PPO）。
"""

# 导入即注册 Gym 任务，例如 Huilun-L5A-WF-Flat-v0。
from .tasks import *  # noqa: F401, F403

# 导出 UI 扩展示例；它不参与强化学习的数据流。
from .ui_extension_example import *  # noqa: F401, F403
