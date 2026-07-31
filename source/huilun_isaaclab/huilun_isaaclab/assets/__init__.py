# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Huilun IsaacLab 扩展维护的机器人资产配置。

对外统一重导出 L5A 的资产、关节/刚体命名和策略—硬件顺序映射，使任务配置与
部署工具可从 ``huilun_isaaclab.assets`` 使用同一份机械契约。
"""

from .robots.l5a import *  # noqa: F401, F403
