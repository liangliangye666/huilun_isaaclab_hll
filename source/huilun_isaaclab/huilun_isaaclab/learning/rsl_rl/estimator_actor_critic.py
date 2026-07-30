# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic whose actor consumes history-estimated base linear velocity."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from tensordict import TensorDict


class VelocityEstimatorActorCritic(ActorCritic):
    """Asymmetric actor-critic with a separately supervised MLP estimator."""

    is_velocity_estimator_policy: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        proprio_group: str = "policy",
        history_group: str = "obs_history",
        command_group: str = "commands",
        estimator_target_group: str = "base_lin_vel_target",
        estimator_output_dim: int = 3,
        estimator_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        estimator_activation: str = "elu",
        estimator_output_detach: bool = True,
        estimator_orthogonal_init: bool = False,
        deployment_metadata: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if estimator_output_dim <= 0:
            raise ValueError("estimator_output_dim must be positive.")
        if not estimator_hidden_dims or any(width <= 0 for width in estimator_hidden_dims):
            raise ValueError("estimator_hidden_dims must contain only positive widths.")
        self._validate_observation_contract(
            obs,
            obs_groups,
            proprio_group,
            history_group,
            command_group,
            estimator_target_group,
            estimator_output_dim,
        )

        self.proprio_group = proprio_group
        self.history_group = history_group
        self.command_group = command_group
        self.estimator_target_group = estimator_target_group
        self.estimator_output_dim = estimator_output_dim
        self.estimator_output_detach = estimator_output_detach
        self.deployment_metadata = dict(deployment_metadata or {})
        self.num_actions = num_actions
        self.history_length = obs[history_group].shape[-2]
        self.proprio_dim = obs[proprio_group].shape[-1]
        self.command_dim = obs[command_group].shape[-1]

        # Let the upstream ActorCritic construct its actor with the augmented
        # [estimated velocity, proprioception, commands] input dimension.
        synthetic_group = "__estimated_base_lin_vel"
        augmented_obs = obs.clone()
        augmented_obs[synthetic_group] = obs[proprio_group].new_zeros(
            *obs[proprio_group].shape[:-1],
            estimator_output_dim,
        )
        augmented_obs_groups = {name: list(groups) for name, groups in obs_groups.items()}
        augmented_obs_groups["policy"] = [synthetic_group, proprio_group, command_group]
        super().__init__(augmented_obs, augmented_obs_groups, num_actions, **kwargs)

        self.velocity_estimator = MLP(
            self.history_length * self.proprio_dim,
            estimator_output_dim,
            list(estimator_hidden_dims),
            estimator_activation,
        )
        if estimator_orthogonal_init:
            linear_layers = [module for module in self.velocity_estimator if isinstance(module, nn.Linear)]
            for layer in linear_layers[:-1]:
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(linear_layers[-1].weight, gain=0.01)
            nn.init.zeros_(linear_layers[-1].bias)
        print(f"Velocity estimator MLP: {self.velocity_estimator}")

    @staticmethod
    def _validate_observation_contract(
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        proprio_group: str,
        history_group: str,
        command_group: str,
        estimator_target_group: str,
        estimator_output_dim: int,
    ) -> None:
        required = (proprio_group, history_group, command_group, estimator_target_group)
        missing = [name for name in required if name not in obs.keys()]
        if missing:
            raise KeyError(f"Velocity estimator observation groups are missing: {missing}")
        if len(obs[proprio_group].shape) != 2:
            raise ValueError(f"'{proprio_group}' must have shape [N, D], got {tuple(obs[proprio_group].shape)}")
        if obs[proprio_group].shape[-1] <= 0:
            raise ValueError(f"'{proprio_group}' must contain at least one feature.")
        if len(obs[history_group].shape) != 3:
            raise ValueError(f"'{history_group}' must have shape [N, H, D], got {tuple(obs[history_group].shape)}")
        if obs[history_group].shape[-2] <= 0:
            raise ValueError(f"'{history_group}' must contain at least one history frame.")
        if obs[history_group].shape[-1] != obs[proprio_group].shape[-1]:
            raise ValueError("History frame width must equal the single-frame proprioception width.")
        if len(obs[command_group].shape) != 2:
            raise ValueError(f"'{command_group}' must have shape [N, C], got {tuple(obs[command_group].shape)}")
        if obs[command_group].shape[-1] <= 0:
            raise ValueError(f"'{command_group}' must contain at least one command.")
        if len(obs[estimator_target_group].shape) != 2:
            raise ValueError(
                f"'{estimator_target_group}' must have shape [N, E], got {tuple(obs[estimator_target_group].shape)}"
            )
        if obs[estimator_target_group].shape[-1] != estimator_output_dim:
            raise ValueError(
                f"Estimator target width {obs[estimator_target_group].shape[-1]} "
                f"does not match estimator output width {estimator_output_dim}."
            )
        for set_name in ("policy", "critic"):
            if set_name not in obs_groups:
                raise KeyError(f"obs_groups must explicitly define the '{set_name}' observation set.")
        expected_policy_groups = [proprio_group, command_group]
        if obs_groups["policy"] != expected_policy_groups:
            raise ValueError(f"obs_groups['policy'] must be {expected_policy_groups}; got {obs_groups['policy']}.")

    def estimate_base_lin_vel(self, obs: TensorDict, detach: bool = False) -> torch.Tensor:
        """Estimate body-frame base linear velocity from the observation history."""
        history = obs[self.history_group]
        flattened_history = history.flatten(start_dim=-2)
        estimate = self.velocity_estimator(flattened_history)
        return estimate.detach() if detach else estimate

    def get_estimator_target(self, obs: TensorDict) -> torch.Tensor:
        """Return the explicit, noise-free base-velocity supervision target."""
        return obs[self.estimator_target_group]

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        estimate = self.estimate_base_lin_vel(obs, detach=self.estimator_output_detach)
        return torch.cat((estimate, obs[self.proprio_group], obs[self.command_group]), dim=-1)
