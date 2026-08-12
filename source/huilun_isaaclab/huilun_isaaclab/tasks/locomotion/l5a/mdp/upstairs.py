# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""上楼梯任务的特权观测、共享步态状态、奖励和终止条件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils import math as math_utils

from .commands import UpstairsVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _upstairs_command(env: ManagerBasedRLEnv, command_name: str) -> UpstairsVelocityCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, UpstairsVelocityCommand):
        raise TypeError(f"Command term {command_name!r} must be UpstairsVelocityCommand, got {type(command).__name__}.")
    return command


def _wheel_body_vel_b(asset: Articulation, body_ids: list[int] | slice) -> torch.Tensor:
    velocity_w = asset.data.body_lin_vel_w[:, body_ids]
    root_quat = asset.data.root_quat_w[:, None, :].expand(-1, velocity_w.shape[1], -1)
    return math_utils.quat_apply_inverse(root_quat, velocity_w)


def _wheel_contact_force_b(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    root_quat = asset.data.root_quat_w[:, None, :].expand(-1, force_w.shape[1], -1)
    return math_utils.quat_apply_inverse(root_quat, force_w)


def _nearest_terrain_height(
    env: ManagerBasedRLEnv,
    query_xy_w: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_hits = scanner.data.ray_hits_w
    distance_sq = torch.sum(torch.square(query_xy_w[:, :, None, :] - ray_hits[:, None, :, :2]), dim=-1)
    nearest_ids = torch.argmin(distance_sq, dim=-1)
    return torch.gather(ray_hits[:, :, 2], 1, nearest_ids)


def wheel_clearance(
    env: ManagerBasedRLEnv,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """轮胎最低点相对最近扫描地面的净空，shape 为 ``[N, 2]``。"""
    asset: Articulation = env.scene[wheel_cfg.name]
    wheel_pos = asset.data.body_pos_w[:, wheel_cfg.body_ids]
    terrain_height = _nearest_terrain_height(env, wheel_pos[:, :, :2], sensor_cfg)
    return torch.clamp(wheel_pos[:, :, 2] - terrain_height - wheel_radius, min=0.0)


def local_terrain_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回基座水平投影处最近的扫描地形高度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _nearest_terrain_height(env, asset.data.root_pos_w[:, None, :2], sensor_cfg).squeeze(1)


def terrain_relative_base_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - local_terrain_height(env, sensor_cfg, asset_cfg)
    return torch.abs(relative_height - target_height)


def root_height_below_local_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - local_terrain_height(env, sensor_cfg, asset_cfg)
    return relative_height < minimum_height


def upstairs_goal_features(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """当前目标在基座航向坐标系中的相对位置和单位方向，共 4 维。"""
    command = _upstairs_command(env, command_name)
    delta_w = command.current_goal_w[:, :2] - command.robot.data.root_pos_w[:, :2]
    heading = command.robot.data.heading_w
    cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
    delta_b = torch.stack(
        (
            cos_heading * delta_w[:, 0] + sin_heading * delta_w[:, 1],
            -sin_heading * delta_w[:, 0] + cos_heading * delta_w[:, 1],
        ),
        dim=1,
    )
    direction = delta_b / torch.clamp(torch.linalg.vector_norm(delta_b, dim=1, keepdim=True), min=1.0e-6)
    return torch.cat((delta_b, direction), dim=1)


def upstairs_step_height(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """返回当前地形台阶高度，平地为零，shape 为 ``[N, 1]``。"""
    return _upstairs_command(env, command_name).terrain_step_height.unsqueeze(1)


def wheel_contact_history(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    history_length: int = 3,
    horizontal_force_threshold: float = 20.0,
) -> torch.Tensor:
    """最近三物理帧两轮水平接触指示，按 ``[左3帧, 右3帧]`` 展平。"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    history = sensor.data.net_forces_w_history[:, :history_length, sensor_cfg.body_ids, :2]
    contacts = torch.linalg.vector_norm(history, dim=-1) > horizontal_force_threshold
    return contacts.transpose(1, 2).float().flatten(start_dim=1)


class _UpstairsGaitState:
    """所有楼梯奖励共享的缓存式摆腿触发状态。"""

    def __init__(self, env: ManagerBasedRLEnv):
        self.blocked_history = torch.zeros(env.num_envs, 2, 3, dtype=torch.bool, device=env.device)
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.swing_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.elapsed = torch.zeros(env.num_envs, device=env.device)
        self.last_update_step = -1

    @property
    def swing_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.active.shape[0], 2, device=self.active.device)
        mask.scatter_(1, self.swing_index.unsqueeze(1), self.active.float().unsqueeze(1))
        return mask

    @property
    def stance_mask(self) -> torch.Tensor:
        return 1.0 - self.swing_mask


def _gait_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    blocking_force_threshold: float = 10.0,
    clearance_margin: float = 0.02,
    minimum_clearance: float = 0.03,
    unlock_ratio: float = 0.9,
    swing_timeout: float = 0.5,
) -> _UpstairsGaitState:
    state = getattr(env, "_l5a_upstairs_gait_state", None)
    if state is None:
        state = _UpstairsGaitState(env)
        env._l5a_upstairs_gait_state = state
    if state.last_update_step == env.common_step_counter:
        return state
    state.last_update_step = env.common_step_counter

    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
        state.blocked_history[reset_mask] = False
        state.active[reset_mask] = False
        state.elapsed[reset_mask] = 0.0

    command = _upstairs_command(env, command_name)
    force_b = _wheel_contact_force_b(env, sensor_cfg, wheel_cfg)
    direction = torch.sign(command.command[:, 0]).unsqueeze(1)
    blocking_strength = torch.relu(-direction * force_b[:, :, 0] - blocking_force_threshold)
    blocked = (blocking_strength > 0.0) & (direction != 0.0)
    state.blocked_history = torch.roll(state.blocked_history, shifts=-1, dims=2)
    state.blocked_history[:, :, -1] = blocked
    stable_blocked = torch.all(state.blocked_history, dim=2)

    nonflat = command.terrain_step_height > 0.0
    start = ~state.active & nonflat & torch.any(stable_blocked, dim=1) & ~reset_mask
    candidate = torch.argmax(blocking_strength, dim=1)
    state.swing_index[start] = candidate[start]
    state.active[start] = True
    state.elapsed[start] = 0.0
    state.elapsed[state.active] += env.step_dt

    clearance = wheel_clearance(env, wheel_radius, wheel_cfg, height_sensor_cfg)
    target = torch.clamp(command.terrain_step_height + clearance_margin, min=minimum_clearance)
    selected_clearance = torch.gather(clearance, 1, state.swing_index.unsqueeze(1)).squeeze(1)
    finished = state.active & ((selected_clearance >= unlock_ratio * target) | (state.elapsed >= swing_timeout))
    state.active[finished] = False
    state.elapsed[finished] = 0.0
    return state


def tracking_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-error / sigma)


def tracking_lin_vel_y_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])
    return torch.exp(-error / sigma)


def tracking_ang_vel_abs_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.abs(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / sigma)


def _potential_difference(env: ManagerBasedRLEnv, value: torch.Tensor, cache_name: str) -> torch.Tensor:
    previous = getattr(env, cache_name, None)
    if previous is None:
        previous = value.detach().clone()
        setattr(env, cache_name, previous)
        return torch.zeros_like(value)
    result = (value - previous) / env.step_dt
    result = torch.where(env.episode_length_buf <= 1, torch.zeros_like(result), result)
    previous.copy_(value.detach())
    return result


def tracking_lin_vel_x_potential(env: ManagerBasedRLEnv, command_name: str, sigma: float = 0.05) -> torch.Tensor:
    value = tracking_lin_vel_x_exp(env, command_name, sigma)
    return _potential_difference(env, value, "_l5a_upstairs_lin_vel_potential")


def tracking_ang_vel_potential(env: ManagerBasedRLEnv, command_name: str, sigma: float = 0.1) -> torch.Tensor:
    value = tracking_ang_vel_abs_exp(env, command_name, sigma)
    return _potential_difference(env, value, "_l5a_upstairs_ang_vel_potential")


def tracking_goal_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = _upstairs_command(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    delta = command.current_goal_w[:, :2] - asset.data.root_pos_w[:, :2]
    direction = delta / torch.clamp(torch.linalg.vector_norm(delta, dim=1, keepdim=True), min=1.0e-6)
    return torch.sign(command.command[:, 0]) * torch.sum(asset.data.root_lin_vel_w[:, :2] * direction, dim=1)


def opposite_base_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    direction = torch.sign(env.command_manager.get_command(command_name)[:, 0])
    return torch.relu(-direction * asset.data.root_lin_vel_b[:, 0])


def opposite_wheel_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    direction = torch.sign(env.command_manager.get_command(command_name)[:, 0]).unsqueeze(1)
    return torch.sum(torch.relu(-direction * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def feet_contact_number(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    state = _gait_state(env, command_name, wheel_radius, wheel_cfg, sensor_cfg, height_sensor_cfg)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = (sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0).float()
    score = state.stance_mask * contact - 2.0 * state.swing_mask * contact
    return torch.mean(score, dim=1) * state.active.float()


def feet_clearance_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    sigma: float = 0.05,
) -> torch.Tensor:
    state = _gait_state(env, command_name, wheel_radius, wheel_cfg, sensor_cfg, height_sensor_cfg)
    target = _upstairs_command(env, command_name).terrain_step_height.unsqueeze(1)
    error = torch.relu(target - wheel_clearance(env, wheel_radius, wheel_cfg, height_sensor_cfg))
    return torch.sum((1.0 - torch.exp(-error / sigma)) * state.swing_mask, dim=1)


def swing_foot_lift(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    state = _gait_state(env, command_name, wheel_radius, wheel_cfg, sensor_cfg, height_sensor_cfg)
    target = torch.clamp(_upstairs_command(env, command_name).terrain_step_height.unsqueeze(1) + 0.02, min=0.03)
    normalized = torch.clamp(wheel_clearance(env, wheel_radius, wheel_cfg, height_sensor_cfg) / target, 0.0, 1.0)
    return torch.sum(normalized * state.swing_mask, dim=1)


def triggered_leg_up_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    state = _gait_state(env, command_name, wheel_radius, wheel_cfg, sensor_cfg, height_sensor_cfg)
    asset: Articulation = env.scene[wheel_cfg.name]
    clearance = wheel_clearance(env, wheel_radius, wheel_cfg, height_sensor_cfg)
    target = torch.clamp(_upstairs_command(env, command_name).terrain_step_height.unsqueeze(1) + 0.02, min=0.03)
    not_lifted = 1.0 - torch.clamp(clearance / target, 0.0, 1.0)
    up_velocity = torch.clamp(torch.relu(asset.data.body_lin_vel_w[:, wheel_cfg.body_ids, 2]) / 0.45, 0.0, 1.0)
    return torch.sum(state.swing_mask * not_lifted * up_velocity, dim=1)


def wrong_leg_lift(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    sigma: float = 0.05,
) -> torch.Tensor:
    state = _gait_state(env, command_name, wheel_radius, wheel_cfg, sensor_cfg, height_sensor_cfg)
    clearance = wheel_clearance(env, wheel_radius, wheel_cfg, height_sensor_cfg)
    return torch.sum(state.stance_mask * clearance / sigma, dim=1) * state.active.float()


def triggered_leg_action_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    state = _gait_state(env, command_name, wheel_radius, wheel_cfg, sensor_cfg, height_sensor_cfg)
    delta = env.action_manager.action - env.action_manager.prev_action
    left_lift = torch.relu(delta[:, 1]) + torch.relu(-delta[:, 2])
    right_lift = torch.relu(delta[:, 5]) + torch.relu(-delta[:, 6])
    return state.swing_mask[:, 0] * left_lift + state.swing_mask[:, 1] * right_lift


def wheel_zero_velocity_during_swing(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    wheel_joint_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    state = _gait_state(env, command_name, wheel_radius, wheel_cfg, sensor_cfg, height_sensor_cfg)
    asset: Articulation = env.scene[wheel_joint_cfg.name]
    velocity_sq = torch.square(asset.data.joint_vel[:, wheel_joint_cfg.joint_ids])
    return torch.exp(-torch.sum(velocity_sq * state.swing_mask, dim=1)) * state.active.float()


def air_wheel_velocity(
    env: ManagerBasedRLEnv,
    wheel_joint_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 10.0,
    velocity_scale: float = 3.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[wheel_joint_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    light_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] < contact_threshold
    penalty = torch.square(asset.data.joint_vel[:, wheel_joint_cfg.joint_ids] / velocity_scale)
    return torch.sum(penalty * light_contact.float(), dim=1)


def wheel_slip(
    env: ManagerBasedRLEnv,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    wheel_joint_cfg: SceneEntityCfg,
    slip_tolerance: float = 0.1,
) -> torch.Tensor:
    asset: Articulation = env.scene[wheel_cfg.name]
    angular_speed = torch.abs(asset.data.joint_vel[:, wheel_joint_cfg.joint_ids])
    actual_forward_speed = torch.abs(_wheel_body_vel_b(asset, wheel_cfg.body_ids)[:, :, 0])
    return torch.sum(torch.relu(angular_speed * wheel_radius - actual_forward_speed - slip_tolerance), dim=1)


def wheel_contact_impact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_force: float = 250.0,
    contact_force_scale: float = 100.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    excess = torch.relu(sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] - max_contact_force)
    return torch.sum(torch.clamp(excess / contact_force_scale, 0.0, 2.0), dim=1)


def foot_landing_velocity(
    env: ManagerBasedRLEnv,
    wheel_radius: float,
    wheel_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    landing_height_threshold: float = 0.08,
    landing_time_threshold: float = 0.12,
    safe_landing_velocity: float = 0.1,
) -> torch.Tensor:
    asset: Articulation = env.scene[wheel_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    clearance = wheel_clearance(env, wheel_radius, wheel_cfg, height_sensor_cfg)
    down_velocity = torch.relu(-asset.data.body_lin_vel_w[:, wheel_cfg.body_ids, 2])
    time_to_contact = clearance / (down_velocity + 1.0e-6)
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    about_to_land = (clearance < landing_height_threshold) & (time_to_contact < landing_time_threshold) & ~contact
    excess = torch.relu(down_velocity - safe_landing_velocity)
    return torch.sum(torch.square(excess) * about_to_land.float(), dim=1)
