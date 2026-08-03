# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export the L5A velocity estimator and actor as two independent models."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .estimator_actor_critic import VelocityEstimatorActorCritic


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ActorOnly(nn.Module):
    """Pure actor wrapper with an explicit deployment tensor interface."""

    def __init__(self, policy: VelocityEstimatorActorCritic) -> None:
        super().__init__()
        self.actor = copy.deepcopy(policy.actor)
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.state_dependent_std = policy.state_dependent_std

    def forward(
        self,
        estimated_base_linear_velocity: torch.Tensor,
        proprioception: torch.Tensor,
        commands: torch.Tensor,
    ) -> torch.Tensor:
        actor_input = torch.cat((estimated_base_linear_velocity, proprioception, commands), dim=-1)
        actor_output = self.actor(self.normalizer(actor_input))
        if self.state_dependent_std:
            return actor_output[..., 0, :]
        return actor_output


class _VelocityEstimatorOnly(nn.Module):
    """Pure encoder mapping ten proprioception frames to base linear velocity."""

    def __init__(self, policy: VelocityEstimatorActorCritic) -> None:
        super().__init__()
        self.estimator = copy.deepcopy(policy.velocity_estimator)

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        return self.estimator(observation_history.flatten(start_dim=-2))


def _tensor_signature(name: str, shape: list[str | int]) -> dict[str, Any]:
    return {"name": name, "dtype": "float32", "shape": shape}


def export_velocity_estimator_policy(
    policy: VelocityEstimatorActorCritic,
    path: str,
    export_jit: bool = True,
    export_onnx: bool = True,
    *,
    deployment_metadata: dict[str, Any] | None = None,
    source_checkpoint: str | None = None,
    training_task: str | None = None,
    export_task: str | None = None,
    action_clip: float | None = None,
) -> None:
    """Atomically export a split Encoder/Actor bundle and its format-v2 manifest."""
    if not export_jit and not export_onnx:
        raise ValueError("At least one of export_jit or export_onnx must be enabled.")

    os.makedirs(path, exist_ok=True)
    actor = _ActorOnly(policy).cpu().eval()
    estimator = _VelocityEstimatorOnly(policy).cpu().eval()

    deployment = copy.deepcopy(deployment_metadata or policy.deployment_metadata)
    if action_clip is not None:
        deployment["policy_output_clip"] = float(action_clip)

    history_shape: list[str | int] = ["batch", policy.history_length, policy.proprio_dim]
    proprio_shape: list[str | int] = ["batch", policy.proprio_dim]
    command_shape: list[str | int] = ["batch", policy.command_dim]
    estimator_shape: list[str | int] = ["batch", policy.estimator_output_dim]
    action_shape: list[str | int] = ["batch", policy.num_actions]
    manifest: dict[str, Any] = {
        "format_version": 2,
        "policy_type": "split_velocity_estimator_actor",
        "history_order": "oldest_to_newest",
        "actor_input_order": ["estimated_base_linear_velocity", "proprioception", "commands"],
        "models": {
            "velocity_estimator": {
                "inputs": [_tensor_signature("observation_history", history_shape)],
                "outputs": [_tensor_signature("estimated_base_linear_velocity", estimator_shape)],
            },
            "policy": {
                "inputs": [
                    _tensor_signature("estimated_base_linear_velocity", estimator_shape),
                    _tensor_signature("proprioception", proprio_shape),
                    _tensor_signature("commands", command_shape),
                ],
                "outputs": [_tensor_signature("actions", action_shape)],
            },
        },
        "deployment": deployment,
        "source": {
            "training_task": training_task,
            "export_task": export_task,
        },
    }
    if source_checkpoint is not None:
        checkpoint_path = Path(source_checkpoint).resolve()
        manifest["source"]["checkpoint"] = {
            "name": checkpoint_path.name,
            "sha256": _sha256_file(checkpoint_path),
        }

    artifact_names: list[str] = []
    with tempfile.TemporaryDirectory(dir=path, prefix=".l5a-export-") as staging_dir:
        if export_jit:
            torch.jit.script(actor).save(os.path.join(staging_dir, "policy.pt"))
            torch.jit.script(estimator).save(os.path.join(staging_dir, "velocity_estimator.pt"))
            artifact_names.extend(("policy.pt", "velocity_estimator.pt"))

        if export_onnx:
            estimated_velocity = torch.zeros(1, policy.estimator_output_dim)
            proprioception = torch.zeros(1, policy.proprio_dim)
            observation_history = torch.zeros(1, policy.history_length, policy.proprio_dim)
            commands = torch.zeros(1, policy.command_dim)
            torch.onnx.export(
                actor,
                (estimated_velocity, proprioception, commands),
                os.path.join(staging_dir, "policy.onnx"),
                export_params=True,
                opset_version=18,
                input_names=["estimated_base_linear_velocity", "proprioception", "commands"],
                output_names=["actions"],
                dynamic_axes={
                    "estimated_base_linear_velocity": {0: "batch"},
                    "proprioception": {0: "batch"},
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
                output_names=["estimated_base_linear_velocity"],
                dynamic_axes={
                    "observation_history": {0: "batch"},
                    "estimated_base_linear_velocity": {0: "batch"},
                },
                dynamo=False,
            )
            artifact_names.extend(("policy.onnx", "velocity_estimator.onnx"))

        manifest["artifacts"] = {
            name: {"sha256": _sha256_file(os.path.join(staging_dir, name))} for name in artifact_names
        }
        for model_name, stem in (("policy", "policy"), ("velocity_estimator", "velocity_estimator")):
            manifest["models"][model_name]["files"] = {
                suffix: f"{stem}.{suffix}"
                for suffix in ("onnx", "pt")
                if f"{stem}.{suffix}" in artifact_names
            }

        staged_manifest = os.path.join(staging_dir, "policy_manifest.json")
        with open(staged_manifest, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)

        for name in artifact_names:
            os.replace(os.path.join(staging_dir, name), os.path.join(path, name))
        os.replace(staged_manifest, os.path.join(path, "policy_manifest.json"))
