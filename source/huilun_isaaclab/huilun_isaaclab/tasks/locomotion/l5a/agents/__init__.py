# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 训练算法配置命名空间。

Gym 注册项通过字符串路径指向 ``rsl_rl_ppo_cfg`` 中的配置类，因此这里无需
主动导入配置；Hydra 在解析 ``rsl_rl_cfg_entry_point`` 时会按需加载它们。
"""
