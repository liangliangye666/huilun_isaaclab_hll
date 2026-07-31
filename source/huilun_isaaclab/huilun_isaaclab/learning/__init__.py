# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Huilun 项目自有的学习算法扩展。

环境/MDP 代码保持 IsaacLab Manager-Based 结构；需要扩展 RSL-RL 的部分
（速度 Encoder、独立 PPO 优化器、runner 和导出器）集中放在此包，避免把
算法细节散落到环境配置中。
"""

# 对外统一暴露项目自有的 RSL-RL 类，训练/播放脚本只需从 learning 导入。
from .rsl_rl import *  # noqa: F401, F403
