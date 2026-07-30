# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL 3.x runner for the L5A velocity-estimator policy."""

from __future__ import annotations

import warnings

import torch
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from .estimator_actor_critic import VelocityEstimatorActorCritic
from .estimator_ppo import VelocityEstimatorPPO


class VelocityEstimatorOnPolicyRunner(OnPolicyRunner):
    """Construct and checkpoint project-local estimator classes explicitly."""

    def _construct_algorithm(self, obs: TensorDict) -> VelocityEstimatorPPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "empirical_normalization is deprecated; use policy observation normalization fields.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        policy_cfg = dict(self.policy_cfg)
        policy_class_name = policy_cfg.pop("class_name")
        if policy_class_name != "VelocityEstimatorActorCritic":
            raise ValueError(f"Unsupported estimator policy class: {policy_class_name}")
        policy = VelocityEstimatorActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        algorithm_cfg = dict(self.alg_cfg)
        algorithm_class_name = algorithm_cfg.pop("class_name")
        if algorithm_class_name != "VelocityEstimatorPPO":
            raise ValueError(f"Unsupported estimator algorithm class: {algorithm_class_name}")
        algorithm = VelocityEstimatorPPO(
            policy,
            device=self.device,
            **algorithm_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "deployment_metadata": self.alg.policy.deployment_metadata,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        if self.logger_type in ("neptune", "wandb") and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        checkpoint_metadata = loaded_dict.get("deployment_metadata")
        if checkpoint_metadata is None:
            warnings.warn(
                "Checkpoint has no deployment metadata; exports will use the current task configuration.",
                RuntimeWarning,
            )
        else:
            if checkpoint_metadata != self.alg.policy.deployment_metadata:
                warnings.warn(
                    "Checkpoint deployment metadata differs from the current task configuration. "
                    "The checkpoint contract is retained for export; verify the runtime observation pipeline.",
                    RuntimeWarning,
                )
            self.alg.policy.deployment_metadata = dict(checkpoint_metadata)
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            estimator_state = loaded_dict.get("estimator_optimizer_state_dict")
            if estimator_state is None:
                warnings.warn(
                    "Checkpoint has no estimator optimizer state; estimator Adam moments start fresh.",
                    RuntimeWarning,
                )
            else:
                self.alg.estimator_optimizer.load_state_dict(estimator_state)
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
