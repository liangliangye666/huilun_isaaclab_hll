# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration schemas for the RSL-RL velocity-estimator extension."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class VelocityEstimatorActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic with a supervised history encoder."""

    class_name: str = "VelocityEstimatorActorCritic"
    proprio_group: str = "policy"
    history_group: str = "obs_history"
    command_group: str = "commands"
    estimator_target_group: str = "base_lin_vel_target"
    estimator_output_dim: int = 3
    estimator_hidden_dims: list[int] = [256, 128]
    estimator_activation: str = "elu"
    estimator_output_detach: bool = True
    estimator_orthogonal_init: bool = False
    deployment_metadata: dict = {}


@configclass
class VelocityEstimatorPPOCfg(RslRlPpoAlgorithmCfg):
    """PPO plus an independently optimized velocity-estimation objective."""

    class_name: str = "VelocityEstimatorPPO"
    estimator_learning_rate: float = 1.0e-3
    estimator_loss_coef: float = 1.0
    estimator_max_grad_norm: float = 0.1
    estimator_num_learning_epochs: int | None = None
