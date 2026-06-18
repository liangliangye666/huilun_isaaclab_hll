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
    lin_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0]
    )
    return torch.exp(-lin_vel_error / std**2)


def nominal_wheel_height_exp(
    env: ManagerBasedRLEnv,
    target_base_height: float,
    wheel_radius: float,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward wheels staying near the nominal height in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    target_wheel_z_b = -(target_base_height - wheel_radius)
    height_error = torch.square(target_wheel_z_b - wheel_pos_b[..., 2])
    return torch.mean(torch.exp(-height_error / std**2), dim=1)


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
