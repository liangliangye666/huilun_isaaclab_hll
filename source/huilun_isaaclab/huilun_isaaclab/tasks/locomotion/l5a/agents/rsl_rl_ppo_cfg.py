# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from huilun_isaaclab.assets.l5a import (
    ACTUATED_JOINT_NAMES,
    HARDWARE_DOF_NAMES,
    HARDWARE_TO_POLICY_STATE_INDICES,
    LEG_JOINT_NAMES,
    POLICY_TO_HARDWARE_ACTION_INDICES,
    WHEEL_JOINT_NAMES,
)
from huilun_isaaclab.learning.rsl_rl import VelocityEstimatorActorCriticCfg, VelocityEstimatorPPOCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 20000
    save_interval = 200
    experiment_name = "l5a_balance"
    clip_actions = 100.0
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class L5AWFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """TRON2-style WF PPO adapted to RSL-RL 3.x and the L5A data contract."""

    class_name = "VelocityEstimatorOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "l5a_wf_flat"
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
        deployment_metadata={
            "control_period_s": 0.01,
            "physics_period_s": 0.005,
            "history_samples": 10,
            "history_duration_s": 0.1,
            "shared_action_delay_physics_steps": [0, 6],
            "shared_action_delay_s": [0.0, 0.03],
            "training_joint_zero_error_rad": [-0.05, 0.05],
            "training_imu_mounting_bias_deg": [-1.2, 1.2],
            "proprioception_layout": [
                {"name": "base_angular_velocity", "size": 3, "scale": 0.25, "frame": "robot_base"},
                {"name": "projected_gravity", "size": 3, "scale": 1.0, "frame": "robot_base"},
                {"name": "leg_joint_position_relative", "size": 6, "scale": 1.0, "order": LEG_JOINT_NAMES},
                {
                    "name": "joint_velocity_relative",
                    "size": 8,
                    "scale": 0.05,
                    "order": ACTUATED_JOINT_NAMES,
                },
                {"name": "previous_action", "size": 8, "scale": 1.0, "order": ACTUATED_JOINT_NAMES},
            ],
            "command_order": ["linear_velocity_x", "linear_velocity_y", "angular_velocity_z"],
            "policy_action_order": ACTUATED_JOINT_NAMES,
            "policy_action_semantics": {
                "leg_position": {"joints": LEG_JOINT_NAMES, "scale": 0.25, "uses_default_offset": True},
                "wheel_velocity": {"joints": WHEEL_JOINT_NAMES, "scale": 0.5},
            },
            "hardware_dof_order": HARDWARE_DOF_NAMES,
            "policy_actions_to_hardware_indices": POLICY_TO_HARDWARE_ACTION_INDICES,
            "hardware_state_to_policy_indices": HARDWARE_TO_POLICY_STATE_INDICES,
        },
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
