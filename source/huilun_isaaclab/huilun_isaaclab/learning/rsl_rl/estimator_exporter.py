# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Combined Encoder + actor export for L5A deployment."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile

import torch
import torch.nn as nn

from .estimator_actor_critic import VelocityEstimatorActorCritic


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _CombinedVelocityEstimatorPolicy(nn.Module):
    def __init__(self, policy: VelocityEstimatorActorCritic) -> None:
        super().__init__()
        self.estimator = copy.deepcopy(policy.velocity_estimator)
        self.actor = copy.deepcopy(policy.actor)
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.state_dependent_std = policy.state_dependent_std

    def forward(
        self,
        proprioception: torch.Tensor,
        observation_history: torch.Tensor,
        commands: torch.Tensor,
    ) -> torch.Tensor:
        estimated_velocity = self.estimator(observation_history.flatten(start_dim=-2))
        actor_input = torch.cat((estimated_velocity, proprioception, commands), dim=-1)
        actor_output = self.actor(self.normalizer(actor_input))
        if self.state_dependent_std:
            return actor_output[..., 0, :]
        return actor_output

    @torch.jit.export
    def estimate_velocity(self, observation_history: torch.Tensor) -> torch.Tensor:
        return self.estimator(observation_history.flatten(start_dim=-2))


class _VelocityEstimatorOnly(nn.Module):
    def __init__(self, policy: VelocityEstimatorActorCritic) -> None:
        super().__init__()
        self.estimator = copy.deepcopy(policy.velocity_estimator)

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        return self.estimator(observation_history.flatten(start_dim=-2))


def export_velocity_estimator_policy(
    policy: VelocityEstimatorActorCritic,
    path: str,
    export_jit: bool = True,
    export_onnx: bool = True,
) -> None:
    """Export a deployment-ready combined policy and a diagnostic estimator."""
    os.makedirs(path, exist_ok=True)
    combined = _CombinedVelocityEstimatorPolicy(policy).cpu().eval()
    estimator = _VelocityEstimatorOnly(policy).cpu().eval()
    manifest = {
        "format_version": 1,
        "policy_type": "history_velocity_estimator_actor",
        "inputs": {
            "proprioception": ["batch", policy.proprio_dim],
            "observation_history": ["batch", policy.history_length, policy.proprio_dim],
            "commands": ["batch", policy.command_dim],
        },
        "outputs": {"actions": ["batch", policy.num_actions]},
        "history_order": "oldest_to_newest",
        "actor_input_order": ["estimated_base_linear_velocity", "proprioception", "commands"],
        "estimator_output": {
            "name": "estimated_base_linear_velocity",
            "shape": ["batch", policy.estimator_output_dim],
            "frame": "robot_base",
            "units": "m/s",
        },
        "deployment": policy.deployment_metadata,
    }
    artifact_names: list[str] = []
    # Build every artifact in a staging directory.  Publishing the manifest
    # last makes it the commit marker and prevents a failed ONNX export from
    # advertising a partially updated model set.
    with tempfile.TemporaryDirectory(dir=path, prefix=".l5a-export-") as staging_dir:
        if export_jit:
            torch.jit.script(combined).save(os.path.join(staging_dir, "policy.pt"))
            torch.jit.script(estimator).save(os.path.join(staging_dir, "velocity_estimator.pt"))
            artifact_names.extend(("policy.pt", "velocity_estimator.pt"))

        if export_onnx:
            proprioception = torch.zeros(1, policy.proprio_dim)
            observation_history = torch.zeros(1, policy.history_length, policy.proprio_dim)
            commands = torch.zeros(1, policy.command_dim)
            torch.onnx.export(
                combined,
                (proprioception, observation_history, commands),
                os.path.join(staging_dir, "policy.onnx"),
                export_params=True,
                opset_version=18,
                input_names=["proprioception", "observation_history", "commands"],
                output_names=["actions"],
                dynamic_axes={
                    "proprioception": {0: "batch"},
                    "observation_history": {0: "batch"},
                    "commands": {0: "batch"},
                    "actions": {0: "batch"},
                },
                dynamo=False,
            )
            torch.onnx.export(
                estimator,
                observation_history,
                os.path.join(staging_dir, "velocity_estimator.onnx"),
                export_params=True,
                opset_version=18,
                input_names=["observation_history"],
                output_names=["estimated_base_lin_vel"],
                dynamic_axes={
                    "observation_history": {0: "batch"},
                    "estimated_base_lin_vel": {0: "batch"},
                },
                dynamo=False,
            )
            artifact_names.extend(("policy.onnx", "velocity_estimator.onnx"))

        manifest["artifacts"] = {
            name: {"sha256": _sha256_file(os.path.join(staging_dir, name))} for name in artifact_names
        }
        staged_manifest = os.path.join(staging_dir, "policy_manifest.json")
        with open(staged_manifest, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)

        for name in artifact_names:
            os.replace(os.path.join(staging_dir, name), os.path.join(path, name))
        os.replace(staged_manifest, os.path.join(path, "policy_manifest.json"))
