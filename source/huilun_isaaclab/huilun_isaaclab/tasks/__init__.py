# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""任务自动发现入口。

``import_packages`` 会递归导入该命名空间下的任务包。L5A 的
``tasks/locomotion/l5a/__init__.py`` 正是在这个阶段执行 ``gym.register``，
而真正的环境对象要等到训练脚本调用 ``gym.make`` 时才会构造。
"""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# ``mdp`` 只保存可复用 term 函数，不应被当成独立任务包扫描；``utils`` 同理。
_BLACKLIST_PKGS = ["utils", ".mdp"]
# 这里的导入具有任务注册副作用，不能简单删除为“未使用 import”。
import_packages(__name__, _BLACKLIST_PKGS)
