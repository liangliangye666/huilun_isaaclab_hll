# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def velocity_height_commands(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
) -> torch.Tensor:
    """Return velocity command plus the fixed base-height target used by balance rewards."""
    velocity_command = env.command_manager.get_command(command_name)
    height_command = torch.full_like(velocity_command[:, :1], target_height)
    return torch.cat((velocity_command, height_command), dim=1)


def base_ang_vel_with_imu_bias(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return angular velocity in the biased IMU mounting frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    mounting_bias = getattr(env, "_l5a_imu_mounting_bias", None)
    if mounting_bias is None:
        return asset.data.root_ang_vel_b
    sensor_quat_w = math_utils.quat_mul(asset.data.root_quat_w, mounting_bias)
    return math_utils.quat_apply_inverse(sensor_quat_w, asset.data.root_ang_vel_w)


def projected_gravity_with_imu_bias(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return projected gravity in the same biased IMU mounting frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    mounting_bias = getattr(env, "_l5a_imu_mounting_bias", None)
    if mounting_bias is None:
        return asset.data.projected_gravity_b
    sensor_quat_w = math_utils.quat_mul(asset.data.root_quat_w, mounting_bias)
    return math_utils.quat_apply_inverse(sensor_quat_w, asset.data.GRAVITY_VEC_W)


def privileged_joint_torque(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the actuator-model torque estimate for the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def privileged_joint_acc(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint accelerations for the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def body_lin_vel_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return selected body linear velocities in the world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)


def current_body_mass(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return current simulated masses, including startup randomization."""
    asset: Articulation = env.scene[asset_cfg.name]
    masses = getattr(env, "_l5a_current_body_mass", None)
    if masses is None:
        masses = asset.root_physx_view.get_masses().to(env.device)
    return masses[:, asset_cfg.body_ids]


def body_contact_force_w(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return current world-frame contact forces for selected bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return forces.flatten(start_dim=1)
