# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO extension with an independent base-velocity estimator optimizer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from .estimator_actor_critic import VelocityEstimatorActorCritic


class _EstimatorRolloutStorage(RolloutStorage):
    """Keep rollout tensors alive until both PPO and estimator updates finish."""

    defer_clear: bool = False

    def clear(self) -> None:
        if not self.defer_clear:
            super().clear()


class VelocityEstimatorPPO(PPO):
    """Run standard RSL-RL PPO, then train the history estimator with MSE."""

    policy: VelocityEstimatorActorCritic

    def __init__(
        self,
        policy: VelocityEstimatorActorCritic,
        estimator_learning_rate: float = 1.0e-3,
        estimator_loss_coef: float = 1.0,
        estimator_max_grad_norm: float = 0.1,
        estimator_num_learning_epochs: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(policy, **kwargs)
        if not isinstance(policy, VelocityEstimatorActorCritic):
            raise TypeError("VelocityEstimatorPPO requires VelocityEstimatorActorCritic.")
        if not policy.estimator_output_detach:
            raise ValueError(
                "VelocityEstimatorPPO requires estimator_output_detach=True because "
                "PPO and the estimator use independent optimizers."
            )
        if estimator_learning_rate <= 0.0:
            raise ValueError("estimator_learning_rate must be positive.")
        if estimator_loss_coef < 0.0:
            raise ValueError("estimator_loss_coef must be non-negative.")
        if estimator_max_grad_norm <= 0.0:
            raise ValueError("estimator_max_grad_norm must be positive.")
        if estimator_num_learning_epochs is not None and estimator_num_learning_epochs <= 0:
            raise ValueError("estimator_num_learning_epochs must be positive when provided.")

        estimator_parameter_ids = {id(parameter) for parameter in policy.velocity_estimator.parameters()}
        ppo_parameters = [
            parameter for parameter in policy.parameters() if id(parameter) not in estimator_parameter_ids
        ]
        # The upstream constructor initially owns all policy parameters. Replace
        # it so PPO and estimator optimizers never manage the same parameter.
        self.optimizer = optim.Adam(ppo_parameters, lr=self.learning_rate)
        self.estimator_optimizer = optim.Adam(
            policy.velocity_estimator.parameters(),
            lr=estimator_learning_rate,
        )
        self.estimator_loss_coef = estimator_loss_coef
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.estimator_num_learning_epochs = estimator_num_learning_epochs

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        """Use storage whose clear can be deferred across the auxiliary update."""
        self.storage = _EstimatorRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def update(self) -> dict[str, float]:
        """Update actor/critic first and the velocity estimator second."""
        # Estimator gradients from the preceding iteration must not contribute
        # to PPO's global gradient-norm clipping.
        self.estimator_optimizer.zero_grad(set_to_none=True)
        if self.storage is None:
            raise RuntimeError("Rollout storage must be initialized before update().")
        self.storage.defer_clear = True
        try:
            loss_dict = super().update()
            num_epochs = (
                self.num_learning_epochs
                if self.estimator_num_learning_epochs is None
                else self.estimator_num_learning_epochs
            )
            generator = self.storage.mini_batch_generator(self.num_mini_batches, num_epochs)
            mean_estimator_loss = 0.0
            num_updates = 0

            for batch in generator:
                obs_batch = batch[0]
                estimate = self.policy.estimate_base_lin_vel(obs_batch)
                target = self.policy.get_estimator_target(obs_batch).detach()
                estimator_loss = torch.mean(torch.square(estimate - target))

                self.estimator_optimizer.zero_grad(set_to_none=True)
                (self.estimator_loss_coef * estimator_loss).backward()
                if self.is_multi_gpu:
                    self._reduce_estimator_gradients()
                nn.utils.clip_grad_norm_(
                    self.policy.velocity_estimator.parameters(),
                    self.estimator_max_grad_norm,
                )
                self.estimator_optimizer.step()

                mean_estimator_loss += estimator_loss.item()
                num_updates += 1

            loss_dict["base_lin_vel_estimator"] = mean_estimator_loss / max(num_updates, 1)
            return loss_dict
        finally:
            # Upstream PPO normally clears its rollout at the end of update().
            # The custom storage defers that clear until Encoder supervision is
            # complete, including exceptional paths where a retry is attempted.
            self.storage.defer_clear = False
            self.storage.clear()

    def _reduce_estimator_gradients(self) -> None:
        """Average estimator gradients across distributed workers."""
        for parameter in self.policy.velocity_estimator.parameters():
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
            parameter.grad /= self.gpu_world_size
