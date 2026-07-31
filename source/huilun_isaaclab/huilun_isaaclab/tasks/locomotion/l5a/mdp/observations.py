# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 的传感器观测与 critic 特权观测。

每个函数只负责生成一个形状为 ``[num_envs, feature_dim]`` 的张量；最终进入
policy、10 帧 history、velocity-estimator target 还是 critic，由环境配置中的
``ObservationGroupCfg`` 决定。策略可见项应尽量接近真机可测信号，质量、执行器
力矩估计和仿真关节加速度等仅仿真可得量应只放入 critic，避免部署时产生信息缺口。
"""

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
    """拼接速度指令与固定基座高度目标。

    速度部分保持 CommandManager 给出的原始列顺序，最后追加一列
    ``target_height``。该辅助项供 balance 任务使用，不会在此函数内缩放或裁剪。
    """
    # ① 取出速度指令 [N, 3]（通常为 vx, vy, omega_z）
    velocity_command = env.command_manager.get_command(command_name)
    # ② 追加固定高度目标 [N, 1]
    height_command = torch.full_like(velocity_command[:, :1], target_height)
    # ③ 拼接：[vx, vy, omega_z, target_height] → [N, 4]
    return torch.cat((velocity_command, height_command), dim=1)


def base_ang_vel_with_imu_bias(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回带安装误差的 IMU 坐标系角速度。

    仿真真实角速度先以世界系表示，再由 ``sensor_quat_w`` 反旋转到虚拟 IMU
    坐标系。若 startup 没有创建安装偏差缓冲，则直接退化为标准基座系角速度。
    该信号适合策略/history；critic 若需要真实状态应使用无偏版本。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # ① 读取 IMU 安装偏差四元数（不存在则退化为标准基座系）
    mounting_bias = getattr(env, "_l5a_imu_mounting_bias", None)
    if mounting_bias is None:
        return asset.data.root_ang_vel_b
    # ② 计算 IMU 在世界系中的姿态：base_quat * mounting_bias
    sensor_quat_w = math_utils.quat_mul(asset.data.root_quat_w, mounting_bias)
    # ③ 把世界系角速度旋转到 IMU 坐标系返回
    return math_utils.quat_apply_inverse(sensor_quat_w, asset.data.root_ang_vel_w)


def projected_gravity_with_imu_bias(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回投影到同一带偏差 IMU 坐标系的重力方向。

    投影重力提供机体倾斜信息而不显式暴露世界姿态。它与
    :func:`base_ang_vel_with_imu_bias` 使用完全相同的安装四元数，保证两类 IMU
    信号处在一致坐标系；无偏差缓冲时返回 IsaacLab 标准 ``projected_gravity_b``。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # ① 读取同一份 IMU 安装偏差（与角速度共享）
    mounting_bias = getattr(env, "_l5a_imu_mounting_bias", None)
    if mounting_bias is None:
        return asset.data.projected_gravity_b
    # ② 计算 IMU 在世界系中的姿态
    sensor_quat_w = math_utils.quat_mul(asset.data.root_quat_w, mounting_bias)
    # ③ 把世界系重力方向旋转到 IMU 坐标系返回
    return math_utils.quat_apply_inverse(sensor_quat_w, asset.data.GRAVITY_VEC_W)


def privileged_joint_torque(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回选中关节的执行器模型施加力矩，通常仅供 critic 使用。

    ``applied_torque`` 是 IsaacLab actuator model 经过限制后的输出估计，并非策略
    动作本身。它在真机上通常无法无偏直接获得，因此属于特权观测。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def privileged_joint_acc(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回选中关节加速度，通常仅供 critic 使用。

    该量由仿真状态差分/缓存提供，噪声特性与真机传感器不同，不应默认放入部署
    policy 观测。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def body_lin_vel_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回选中刚体的世界系线速度，并展平 body 与 xyz 两维。

    输出列按 ``[body_0(x,y,z), body_1(x,y,z), ...]`` 排列。世界系与基座系不可
    混用；该函数主要为 critic 提供轮/连杆运动真值。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)


def current_body_mass(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回包含 startup 随机化结果的当前刚体质量。

    训练时优先读取质量随机化事件写入的 GPU 缓存
    ``env._l5a_current_body_mass``，避免每个策略步从 PhysX CPU 属性接口回读。
    若缓存不存在（例如关闭随机化的 play 环境），才现场读取。最后再按
    ``asset_cfg.body_ids`` 选列，因此输出顺序由 SceneEntityCfg 决定。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # ① 优先使用 GPU 缓存（startup 随机化写入的），避免 CPU readback
    masses = getattr(env, "_l5a_current_body_mass", None)
    # ② 缓存不存在时（play 环境）从 PhysX 现场读取
    if masses is None:
        masses = asset.root_physx_view.get_masses().to(env.device)
    # ③ 按 body_ids 选列并返回
    return masses[:, asset_cfg.body_ids]


def body_contact_force_w(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """返回选中刚体的当前世界系净接触力，并展平 body 与 xyz 两维。

    输出列为 ``[body_0(Fx,Fy,Fz), body_1(Fx,Fy,Fz), ...]``。这里读取当前
    ``net_forces_w``，不是 contact sensor 的历史窗口；通常作为 critic 的地面
    接触真值，policy 侧若真机无对应传感器则不应使用。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return forces.flatten(start_dim=1)
