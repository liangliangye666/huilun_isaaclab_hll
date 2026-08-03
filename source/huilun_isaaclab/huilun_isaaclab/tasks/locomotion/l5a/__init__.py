# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""L5A Gymnasium 任务注册入口。

导入 ``huilun_isaaclab.tasks`` 时会递归导入本模块，下面的 ``gym.register`` 因而
以导入副作用把任务 ID 写入 Gym registry。注册阶段不会创建仿真，也不会加载
机器人资产；只有训练/播放脚本调用 ``gym.make(task_id, ...)`` 时，字符串形式的
entry point 才会被解析。

每个 ID 同时绑定两份配置：

* ``env_cfg_entry_point`` 决定 Scene 和所有 Manager term；
* ``rsl_rl_cfg_entry_point`` 决定 Runner、网络、观测分组和 PPO 超参数。

Balance 与 WF 分别使用自己的 agent 配置；同一任务的 Train/Play ID 共用网络
配置，只替换环境配置，因此 checkpoint 的观测/动作维度保持一致。
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 旧 IsaacGym balance 任务的 Manager-Based 训练基线：
# [N, 320] 展平历史同时交给标准 PPO 的 Actor/Critic。
# gym.register(
#     id="Huilun-L5A-Balance-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     # IsaacLab 使用 GPU 批量张量/Manager 生命周期，不走标准单环境 Gym checker。
#     # 这不会关闭 IsaacLab 自己对配置、shape 或 SceneEntityCfg 的解析检查。
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.balance_env_cfg:L5ABalanceEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
#     },
# )

# 完整 WF 平地训练：非对称 Actor-Critic + 10 帧历史线速度 Encoder。
gym.register(
    id="Huilun-L5A-WF-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wf_flat_env_cfg:L5AWFFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:L5AWFPPORunnerCfg",
    },
)

# WF 评估/导出：网络配置与训练任务相同，环境侧关闭噪声、延迟和随机化。
gym.register(
    id="Huilun-L5A-WF-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wf_flat_env_cfg:L5AWFFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:L5AWFPPORunnerCfg",
    },
)

# Balance 的轻量播放入口；它只关闭部分随机化，详见 L5ABalanceEnvCfg_PLAY。
# gym.register(
#     id="Huilun-L5A-Balance-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.balance_env_cfg:L5ABalanceEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
#     },
# )
