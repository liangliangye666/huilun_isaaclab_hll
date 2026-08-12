# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A WF 上楼梯任务的独立 RSL-RL Runner 配置。"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from huilun_isaaclab.learning.rsl_rl import VelocityEstimatorActorCriticCfg, VelocityEstimatorPPOCfg

from ..upstairs_env_cfg import build_l5a_wf_upstairs_deployment_metadata


@configclass
class L5AWFUpstairsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """沿用 WF 网络契约，同时隔离 rollout、日志目录和 checkpoint 元数据。"""

    class_name = "VelocityEstimatorOnPolicyRunner"
    num_steps_per_env = 48
    max_iterations = 100000
    save_interval = 500
    experiment_name = "l5a_wf_upstairs"
    clip_actions = 100.0
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
        deployment_metadata=build_l5a_wf_upstairs_deployment_metadata(),
    )
    algorithm = VelocityEstimatorPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        estimator_learning_rate=1.0e-3,
        estimator_loss_coef=1.0,
        estimator_max_grad_norm=0.1,
        estimator_num_learning_epochs=5,
    )
