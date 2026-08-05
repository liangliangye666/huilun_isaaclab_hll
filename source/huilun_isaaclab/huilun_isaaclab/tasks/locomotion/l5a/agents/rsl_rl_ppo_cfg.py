# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""L5A WF 环境对应的 RSL-RL Runner、网络和 PPO 配置。

Gym 注册项通过 ``rsl_rl_cfg_entry_point`` 找到这里的配置类。环境配置负责产生
TensorDict，Runner 配置负责说明哪些 observation group 交给 Actor/Critic、每次
rollout 收集多长，以及实例化标准 PPO 还是带独立速度 Encoder 的项目内 PPO。

需要区分三个层次：本文件只保存超参数和分组名称；网络在拿到环境第一批观测后
才推导真实输入宽度；优化与 checkpoint 生命周期由对应 Runner/Algorithm 实现。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from huilun_isaaclab.learning.rsl_rl import VelocityEstimatorActorCriticCfg, VelocityEstimatorPPOCfg

from ..wf_flat_env_cfg import build_l5a_wf_deployment_metadata


@configclass
class L5AWFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """适配 RSL-RL 3.x 与 L5A 张量契约的 WF PPO。

    环境原始分组及 shape 为：

    * ``policy [N, 28]``、``obs_history [N, 10, 28]``；
    * ``commands [N, 3]``、``base_lin_vel_target [N, 3]``；
    * ``critic [N, 68]``。

    项目内 Actor-Critic 将历史展平后经 ``280 -> 256 -> 128 -> 3`` Encoder 估计
    基座线速度，再构造 34 维 Actor 输入
    ``[估计速度 3, 当前本体 28, 指令 3]``。Critic 则接收 71 维输入
    ``[特权状态 68, 指令 3]``。监督真值只进入 Encoder MSE，不进入 Actor。
    """

    # train.py 根据该名称选择项目内 Runner；它负责两套 optimizer 的 checkpoint。
    class_name = "VelocityEstimatorOnPolicyRunner"
    # 每环境 24 个 50 Hz 控制步（0.48 s）组成一次 on-policy rollout。
    # 保留 24 步可保持每轮 PPO 的样本数；若改为 12 步才是保持原 0.24 s 物理时域。
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "l5a_wf_flat"
    clip_actions = 100.0
    # 这里只声明 RSL-RL 的 Actor/Critic基础组。history 和监督 target 仍保留在
    # TensorDict 中，由 VelocityEstimatorActorCritic/PPO 按下方 group 名显式读取。
    obs_groups = {
        "policy": ["policy", "commands"],
        "critic": ["critic", "commands"],
    }
    policy = VelocityEstimatorActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        proprio_group="policy",
        history_group="obs_history",
        command_group="commands",
        estimator_target_group="base_lin_vel_target",
        estimator_output_dim=3,
        estimator_hidden_dims=[256, 128],
        estimator_activation="elu",
        estimator_output_detach=True,
        estimator_orthogonal_init=False,
        # 这些数据不参与网络前向；它们由 WF 环境配置生成，并随 checkpoint/export
        # manifest 保存，防止真机侧误用观测缩放、关节顺序、动作含义或控制周期。
        deployment_metadata=build_l5a_wf_deployment_metadata(),
    )
    algorithm = VelocityEstimatorPPOCfg(
        # PPO 只更新 Actor/Critic；Encoder 使用下方独立 Adam 和 MSE 更新。
        # estimator_output_detach=True 是两条梯度路径保持独立的关键。
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        # gamma/lam 按策略步定义。这里保留常用 PPO 基线；若要严格保持
        # 原 100 Hz 设置的连续时间衰减，50 Hz 下对应 gamma=0.99**2、lam=0.95**2。
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        estimator_learning_rate=1.0e-3,
        estimator_loss_coef=1.0,
        estimator_max_grad_norm=0.1,
        estimator_num_learning_epochs=5,
    )
