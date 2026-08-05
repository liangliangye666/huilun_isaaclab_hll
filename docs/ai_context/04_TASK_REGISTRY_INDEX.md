# 任务注册索引

- 生成时间：2026-08-05T23:00:27+08:00
- 仓库根目录：/mnt/isaacdata/myproject/huilun_isaaclab
- 检索范围：source/ scripts/

## gym.register 附近代码

```text
source/huilun_isaaclab/huilun_isaaclab/__init__.py-2-# All rights reserved.
source/huilun_isaaclab/huilun_isaaclab/__init__.py-3-#
source/huilun_isaaclab/huilun_isaaclab/__init__.py-4-# SPDX-License-Identifier: BSD-3-Clause
source/huilun_isaaclab/huilun_isaaclab/__init__.py-5-
source/huilun_isaaclab/huilun_isaaclab/__init__.py-6-"""Huilun IsaacLab 扩展的 Python 入口。
source/huilun_isaaclab/huilun_isaaclab/__init__.py-7-
source/huilun_isaaclab/huilun_isaaclab/__init__.py-8-导入本包并不只是加载符号，还会触发两类注册副作用：
source/huilun_isaaclab/huilun_isaaclab/__init__.py-9-
source/huilun_isaaclab/huilun_isaaclab/__init__.py:10:1. ``tasks`` 会递归发现任务包，并执行各任务中的 ``gym.register``；
source/huilun_isaaclab/huilun_isaaclab/__init__.py-11-2. ``ui_extension_example`` 暴露 Omniverse 扩展入口，由 Isaac Sim 扩展管理器调用。
source/huilun_isaaclab/huilun_isaaclab/__init__.py-12-
source/huilun_isaaclab/huilun_isaaclab/__init__.py-13-训练脚本必须先创建 ``SimulationApp``，再导入本包。原因是下列模块最终会依赖
source/huilun_isaaclab/huilun_isaaclab/__init__.py-14-``omni``/``pxr``，普通 Python 解释器无法独立完成这段初始化。
source/huilun_isaaclab/huilun_isaaclab/__init__.py-15-
source/huilun_isaaclab/huilun_isaaclab/__init__.py-16-建议接手阅读顺序是：``l5a/__init__.py``（任务入口）→ ``wf_flat_env_cfg.py``
source/huilun_isaaclab/huilun_isaaclab/__init__.py-17-（完整训练配置）→ ``assets``/``mdp``（机器人和各 Manager term）→
source/huilun_isaaclab/huilun_isaaclab/__init__.py-18-``agents/rsl_rl_ppo_cfg.py``（算法超参数）→ ``learning/rsl_rl``（Encoder/PPO）。
--
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-1-# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-2-# All rights reserved.
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-3-#
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-4-# SPDX-License-Identifier: BSD-3-Clause
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-5-
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-6-"""任务自动发现入口。
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-7-
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-8-``import_packages`` 会递归导入该命名空间下的任务包。L5A 的
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py:9:``tasks/locomotion/l5a/__init__.py`` 正是在这个阶段执行 ``gym.register``，
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-10-而真正的环境对象要等到训练脚本调用 ``gym.make`` 时才会构造。
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-11-"""
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-12-
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-13-##
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-14-# Register Gym environments.
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-15-##
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-16-
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py-17-from isaaclab_tasks.utils import import_packages
--
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-1-# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-2-# All rights reserved.
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-3-#
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-4-# SPDX-License-Identifier: BSD-3-Clause
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-5-
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-6-"""L5A Gymnasium 任务注册入口。
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-7-
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:8:导入 ``huilun_isaaclab.tasks`` 时会递归导入本模块，下面的 ``gym.register`` 因而
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-9-以导入副作用把任务 ID 写入 Gym registry。注册阶段不会创建仿真，也不会加载
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-10-机器人资产；只有训练/播放脚本调用 ``gym.make(task_id, ...)`` 时，字符串形式的
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-11-entry point 才会被解析。
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-12-
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-13-每个 ID 同时绑定两份配置：
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-14-
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-15-* ``env_cfg_entry_point`` 决定 Scene 和所有 Manager term；
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-16-* ``rsl_rl_cfg_entry_point`` 决定 Runner、网络、观测分组和 PPO 超参数。
--
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-23-
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-24-from . import agents
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-25-
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-26-##
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-27-# Register Gym environments.
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-28-##
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-29-
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-30-# 完整 WF 平地训练：非对称 Actor-Critic + 10 帧历史线速度 Encoder。
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:31:gym.register(
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-32-    id="Huilun-L5A-WF-Flat-v0",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-33-    entry_point="isaaclab.envs:ManagerBasedRLEnv",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-34-    disable_env_checker=True,
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-35-    kwargs={
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-36-        "env_cfg_entry_point": f"{__name__}.wf_flat_env_cfg:L5AWFFlatEnvCfg",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-37-        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:L5AWFPPORunnerCfg",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-38-    },
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-39-)
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-40-
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-41-# WF 评估/导出：网络配置与训练任务相同，环境侧关闭噪声、延迟和随机化。
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:42:gym.register(
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-43-    id="Huilun-L5A-WF-Flat-Play-v0",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-44-    entry_point="isaaclab.envs:ManagerBasedRLEnv",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-45-    disable_env_checker=True,
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-46-    kwargs={
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-47-        "env_cfg_entry_point": f"{__name__}.wf_flat_env_cfg:L5AWFFlatEnvCfg_PLAY",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-48-        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:L5AWFPPORunnerCfg",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-49-    },
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py-50-)
```

## Entry point 和配置入口检索

```text
scripts/zero_agent.py:68:if __name__ == "__main__":
scripts/random_agent.py:68:if __name__ == "__main__":
scripts/list_envs.py:56:            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
scripts/list_envs.py:63:if __name__ == "__main__":
scripts/rsl_rl/cli_args.py:55:    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
scripts/rsl_rl/train.py:26:    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
scripts/rsl_rl/train.py:104:logger = logging.getLogger(__name__)
scripts/rsl_rl/train.py:242:if __name__ == "__main__":
scripts/rsl_rl/play.py:28:    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
scripts/rsl_rl/play.py:583:                    "class": actuator.__class__.__name__,
scripts/rsl_rl/play.py:603:            "policy_class": self.policy_nn.__class__.__name__,
scripts/rsl_rl/play.py:658:if __name__ == "__main__":
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py:22:import_packages(__name__, _BLACKLIST_PKGS)
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/agents/__init__.py:9:主动导入配置；Hydra 在解析 ``rsl_rl_cfg_entry_point`` 时会按需加载它们。
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/agents/rsl_rl_ppo_cfg.py:8:Gym 注册项通过 ``rsl_rl_cfg_entry_point`` 找到这里的配置类。环境配置负责产生
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:15:* ``env_cfg_entry_point`` 决定 Scene 和所有 Manager term；
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:16:* ``rsl_rl_cfg_entry_point`` 决定 Runner、网络、观测分组和 PPO 超参数。
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:33:    entry_point="isaaclab.envs:ManagerBasedRLEnv",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:36:        "env_cfg_entry_point": f"{__name__}.wf_flat_env_cfg:L5AWFFlatEnvCfg",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:37:        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:L5AWFPPORunnerCfg",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:44:    entry_point="isaaclab.envs:ManagerBasedRLEnv",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:47:        "env_cfg_entry_point": f"{__name__}.wf_flat_env_cfg:L5AWFFlatEnvCfg_PLAY",
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:48:        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:L5AWFPPORunnerCfg",
```

## 任务导入链检索

```text
scripts/zero_agent.py:38:import huilun_isaaclab.tasks  # noqa: F401
scripts/random_agent.py:38:import huilun_isaaclab.tasks  # noqa: F401
scripts/list_envs.py:37:import huilun_isaaclab.tasks  # noqa: F401
scripts/list_envs.py:53:    for task_spec in gym.registry.values():
scripts/rsl_rl/play.py:95:import huilun_isaaclab.tasks  # noqa: F401
scripts/rsl_rl/train.py:106:import huilun_isaaclab.tasks  # noqa: F401
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py:8:``import_packages`` 会递归导入该命名空间下的任务包。L5A 的
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py:17:from isaaclab_tasks.utils import import_packages
source/huilun_isaaclab/huilun_isaaclab/tasks/__init__.py:22:import_packages(__name__, _BLACKLIST_PKGS)
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py:8:导入 ``huilun_isaaclab.tasks`` 时会递归导入本模块，下面的 ``gym.register`` 因而
```
