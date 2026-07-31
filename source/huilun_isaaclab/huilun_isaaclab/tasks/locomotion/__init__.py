# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion 任务命名空间。

本层不创建环境；具体任务由子包（当前为 ``l5a``）注册。保留对 Gymnasium 的
导入，使该命名空间与 IsaacLab 任务包的注册约定保持一致。
"""

import gymnasium as gym  # noqa: F401
