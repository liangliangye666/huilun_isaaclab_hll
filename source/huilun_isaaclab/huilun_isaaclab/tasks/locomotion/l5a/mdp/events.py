# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A-specific domain-randomization events."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_imu_mounting_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    roll_pitch_range_deg: tuple[float, float],
) -> None:
    """Sample a fixed IMU mounting rotation for every selected environment."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(env.device)

    if not hasattr(env, "_l5a_imu_mounting_bias"):
        bias = torch.zeros(env.scene.num_envs, 4, device=env.device)
        bias[:, 0] = 1.0
        env._l5a_imu_mounting_bias = bias

    lower = math.radians(roll_pitch_range_deg[0])
    upper = math.radians(roll_pitch_range_deg[1])
    roll = torch.empty(len(env_ids), device=env.device).uniform_(lower, upper)
    pitch = torch.empty(len(env_ids), device=env.device).uniform_(lower, upper)
    yaw = torch.zeros_like(roll)
    env._l5a_imu_mounting_bias[env_ids] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)


def randomize_joint_effort_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float],
) -> None:
    """Randomize per-environment motor capability and keep actuator estimates synchronized."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_device = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids_device = env_ids.to(asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = list(range(asset.num_joints))
    else:
        joint_ids = list(asset_cfg.joint_ids)
    joint_ids_device = torch.tensor(joint_ids, dtype=torch.long, device=asset.device)

    nominal_limits = asset.data.joint_effort_limits[
        env_ids_device[:, None],
        joint_ids_device[None, :],
    ].clone()
    factors = math_utils.sample_uniform(
        scale_range[0],
        scale_range[1],
        nominal_limits.shape,
        device=asset.device,
    )
    randomized_limits = nominal_limits * factors
    asset.write_joint_effort_limit_to_sim(
        randomized_limits,
        joint_ids=joint_ids,
        env_ids=env_ids_device,
    )

    selected_column = {joint_id: column for column, joint_id in enumerate(joint_ids)}
    for actuator in asset.actuators.values():
        if actuator.joint_indices == slice(None):
            actuator_joint_ids = list(range(asset.num_joints))
        else:
            actuator_joint_ids = actuator.joint_indices.tolist()
        for local_id, joint_id in enumerate(actuator_joint_ids):
            if joint_id not in selected_column:
                continue
            values = randomized_limits[:, selected_column[joint_id]]
            actuator.effort_limit_sim[env_ids_device, local_id] = values
            actuator.effort_limit[env_ids_device, local_id] = values


def scale_current_rigid_body_mass_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float],
) -> None:
    """Scale current body masses and inertias by the same random factor.

    Unlike IsaacLab's default mass randomizer, this intentionally starts from
    the values produced by earlier startup events. It therefore composes with
    the separate base-mass and link-mass randomizations used by the WF task.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    else:
        body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")

    masses = asset.root_physx_view.get_masses().clone()
    inertias = asset.root_physx_view.get_inertias().clone()
    factors = math_utils.sample_uniform(
        scale_range[0],
        scale_range[1],
        (len(env_ids), len(body_ids)),
        device="cpu",
    )
    index = (env_ids[:, None], body_ids[None, :])
    masses[index] *= factors
    inertias[index] *= factors.unsqueeze(-1)
    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)
    # Mass randomization is a startup-only event in the WF task.  Keep the
    # resulting values on the simulation device so the privileged observation
    # does not perform a PhysX CPU readback on every policy step.
    env._l5a_current_body_mass = masses.to(asset.device)


def randomize_rigid_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize each selected body's center of mass independently."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    else:
        body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")

    if distribution == "uniform":
        sample_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        sample_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        sample_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unsupported COM randomization distribution: {distribution}")

    ranges = torch.tensor(
        [com_ranges.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
        device="cpu",
    )
    samples = sample_fn(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), len(body_ids), 3),
        device="cpu",
    )
    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids[None, :], :3] += samples
    asset.root_physx_view.set_coms(coms, env_ids)
