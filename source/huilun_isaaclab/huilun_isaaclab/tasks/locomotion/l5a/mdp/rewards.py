# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _body_pos_b(asset: Articulation, body_ids: list[int] | slice) -> torch.Tensor:
    """Return body positions expressed in the robot base frame."""
    body_pos_w = asset.data.body_pos_w[:, body_ids, :]
    rel_pos_w = body_pos_w - asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, rel_pos_w.shape[1], -1)
    return quat_apply_inverse(root_quat_w, rel_pos_w)


def base_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root height error with an L1 kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 2] - target_height)


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of the x linear velocity command in the base frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-lin_vel_error / std**2)


def nominal_wheel_height_exp(
    env: ManagerBasedRLEnv,
    target_base_height: float,
    wheel_radius: float,
    std: float,
    speed_attenuation_std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward wheels staying near the nominal height in the base frame.

    The reward is attenuated by the commanded speed so that height precision
    is relaxed at higher velocities (same behaviour as the original IsaacGym task).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    target_wheel_z_b = -(target_base_height - wheel_radius)
    height_error = torch.square(target_wheel_z_b - wheel_pos_b[..., 2])
    base_reward = torch.mean(torch.exp(-height_error / std**2), dim=1)
    # speed-dependent attenuation
    vel_cmd = env.command_manager.get_command(command_name)
    vel_norm = torch.norm(vel_cmd[:, :3], dim=1)
    return base_reward * torch.exp(-torch.square(vel_norm) / speed_attenuation_std**2)


def leg_y_symmetry_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward symmetric lateral wheel placement around the base centerline."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    symmetry_error = torch.abs(wheel_pos_b[:, 0, 1]) - torch.abs(wheel_pos_b[:, 1, 1])
    return torch.exp(-torch.square(symmetry_error) / std**2)


def same_wheel_x_position_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize fore-aft mismatch between the two wheels in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    return torch.abs(wheel_pos_b[:, 0, 0] - wheel_pos_b[:, 1, 0])


def same_wheel_z_position_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize vertical mismatch between the two wheels in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    return torch.square(wheel_pos_b[:, 0, 2] - wheel_pos_b[:, 1, 2])


def wheel_distance_range_l1(
    env: ManagerBasedRLEnv,
    min_distance: float,
    max_distance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize wheel distance outside the nominal track-width range."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    wheel_distance_xy = torch.norm(wheel_pos_w[:, 0, :2] - wheel_pos_w[:, 1, :2], dim=-1)
    lower_error = torch.clamp(min_distance - wheel_distance_xy, min=0.0)
    upper_error = torch.clamp(wheel_distance_xy - max_distance, min=0.0)
    return lower_error + upper_error


def wheel_distance_alignment_exp(
    env: ManagerBasedRLEnv,
    min_distance: float,
    max_distance: float,
    desired_distance: float,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    lateral_command_scale: float = 0.8,
) -> torch.Tensor:
    """Reward a valid wheel track width and alignment around its L5A nominal value."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    distance = torch.abs(wheel_pos_b[:, 0, 1] - wheel_pos_b[:, 1, 1])
    outside_error = torch.clamp(min_distance - distance, min=0.0)
    outside_error += torch.clamp(distance - max_distance, min=0.0)
    range_reward = torch.exp(-torch.square(outside_error) / std**2)
    nominal_reward = torch.exp(-torch.square(distance - desired_distance) / std**2)

    lateral_command = torch.abs(env.command_manager.get_command(command_name)[:, 1])
    nominal_weight = 1.0 - torch.clamp(lateral_command / lateral_command_scale, 0.0, 1.0)
    return 0.5 * (range_reward + nominal_weight * nominal_reward)


def stand_still_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base motion only when the sampled command requests standing."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_standing = torch.norm(command[:, :2], dim=1) < lin_threshold
    yaw_standing = torch.abs(command[:, 2]) < ang_threshold
    lin_penalty = torch.sum(torch.abs(asset.data.root_lin_vel_b[:, :2]), dim=1) * lin_standing
    yaw_penalty = torch.abs(asset.data.root_ang_vel_b[:, 2]) * yaw_standing
    return lin_penalty + yaw_penalty


def base_projection_at_wheel_midpoint_exp(
    env: ManagerBasedRLEnv,
    std: float,
    wheel_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the base projection remaining above the two-wheel support midpoint."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_w = asset.data.body_pos_w[:, wheel_cfg.body_ids, :2]
    midpoint_xy = torch.mean(wheel_pos_w, dim=1)
    error = torch.sum(torch.square(asset.data.root_pos_w[:, :2] - midpoint_xy), dim=1)
    return torch.exp(-error / std**2)


def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute mechanical power of selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def joint_deviation_from_default_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected joints deviating from their configured default pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(error), dim=1)


def action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action rate (action smoothness).

    Computes the L2 norm of (a_t - 2*a_{t-1} + a_{t-2}), discouraging jerky
    action trajectories. Matches the ``_reward_action_smooth`` from the
    original IsaacGym L5A balance task.

    .. note::
        Maintains a persistent ``_prev_prev_action`` buffer on the environment.
        The penalty is zeroed for the first two steps after reset.
    """
    # -- allocate persistent buffer on first call -------------------------------
    if not hasattr(env, "_prev_prev_action"):
        env._prev_prev_action = torch.zeros(env.num_envs, env.action_manager.action.shape[-1], device=env.device)

    # -- early-episode masking (prev_prev is invalid for steps 0, 1) ------------
    is_early = env.episode_length_buf < 3
    prev_prev = env._prev_prev_action.clone()
    prev_prev[is_early] = 0.0

    penalty = torch.sum(
        torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + prev_prev),
        dim=1,
    )
    penalty[is_early] = 0.0

    # -- rotate buffer for next step --------------------------------------------
    env._prev_prev_action = env.action_manager.prev_action.clone()
    return penalty
