# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 专用域随机化（Domain Randomization）事件。

这些函数由 IsaacLab ``EventManager`` 按环境配置指定的 mode 调用。WF 配置把
传感器安装误差、质量/惯量、质心、驱动能力等放在 ``startup`` 阶段：每个并行
环境启动时得到一套固定但不同的"机器人个体"，回合 reset 不会持续抖动这些
物理参数。质量随机化的多层组合依赖事件配置中的执行顺序，详见对应函数说明。
"""

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
    """为指定环境采样固定的 IMU 安装姿态误差。

    只随机 roll/pitch，yaw 固定为零；角度范围由度转换为弧度。保存的四元数表示
    IMU 坐标系相对机器人基座坐标系的安装旋转。观测函数通过
    ``root_quat_w * mounting_bias`` 得到 IMU 在世界系中的姿态，再把世界系角速度
    与重力投影到带偏差的 IMU 坐标系。该误差作用于策略侧传感器观测，不会修改
    仿真中机器人真实姿态。
    """
    # ① 解析 env_ids：None → 全环境
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(env.device)

    # ② 首次调用时初始化缓冲区（未选中的环境保持单位四元数）
    if not hasattr(env, "_l5a_imu_mounting_bias"):
        bias = torch.zeros(env.scene.num_envs, 4, device=env.device)
        bias[:, 0] = 1.0
        env._l5a_imu_mounting_bias = bias

    # ③ 为每个环境独立采样 roll/pitch（yaw=0）
    lower = math.radians(roll_pitch_range_deg[0])
    upper = math.radians(roll_pitch_range_deg[1])
    roll = torch.empty(len(env_ids), device=env.device).uniform_(lower, upper)
    pitch = torch.empty(len(env_ids), device=env.device).uniform_(lower, upper)
    yaw = torch.zeros_like(roll)
    # ④ 欧拉角 → 四元数，写入缓冲区
    env._l5a_imu_mounting_bias[env_ids] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)


def randomize_joint_effort_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float],
) -> None:
    """逐环境随机电机力矩能力，并同步 PhysX 与执行器内部限制。

    对每个选中环境/关节，以当前 effort limit 为名义值独立乘以随机系数。写入
    PhysX 后还必须同步 ``actuator.effort_limit_sim`` 和 ``effort_limit``：
    前者参与仿真侧限制，后者参与执行器模型的力矩裁剪/估计。若只改其中一处，
    ``applied_torque`` 等特权观测与真实仿真能力会不一致。

    此事件按 WF 配置只在 startup 执行一次；若手动重复调用，会以当时已有的
    limit 为基准继续相乘，而不是自动回到 USD 名义值。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # ① 解析 env_ids 和 joint_ids
    if env_ids is None:
        env_ids_device = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids_device = env_ids.to(asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = list(range(asset.num_joints))
    else:
        joint_ids = list(asset_cfg.joint_ids)
    joint_ids_device = torch.tensor(joint_ids, dtype=torch.long, device=asset.device)

    # ② 读取当前名义力矩限制
    nominal_limits = asset.data.joint_effort_limits[
        env_ids_device[:, None],
        joint_ids_device[None, :],
    ].clone()
    # ③ 逐环境/关节采样缩放因子，乘以名义限制
    factors = math_utils.sample_uniform(
        scale_range[0],
        scale_range[1],
        nominal_limits.shape,
        device=asset.device,
    )
    randomized_limits = nominal_limits * factors
    # ④ 写回 PhysX 仿真侧限制
    asset.write_joint_effort_limit_to_sim(
        randomized_limits,
        joint_ids=joint_ids,
        env_ids=env_ids_device,
    )

    # ⑤ 同步到所有 actuator 对象的 effort_limit_sim 和 effort_limit
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
    """用同一逐刚体系数缩放"当前"质量和惯量。

    与 IsaacLab 常见的"从默认值重新采样"不同，本函数先读取前序 startup 事件
    已经写入 PhysX 的当前值，再继续相乘。因此 WF 中的随机化是有意组合的：

    1. 前序事件分别改变基座质量和连杆质量；
    2. 本事件再对每个选中刚体同时缩放质量及惯量。

    对同一刚体使用相同系数缩放质量和完整惯量张量，可在增加模型差异的同时维持
    两者的基本一致性。PhysX 刚体属性接口在 CPU 上读写，最终质量会缓存到仿真
    device，供 critic 特权观测使用，避免每个策略步进行 CPU readback。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # ① 解析 env_ids 和 body_ids（PhysX 属性接口在 CPU 上操作）
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    else:
        body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")

    # ② 读取 PhysX 当前质量和惯量（含前序随机化结果）
    masses = asset.root_physx_view.get_masses().clone()
    inertias = asset.root_physx_view.get_inertias().clone()
    # ③ 逐环境/刚体采样缩放因子
    factors = math_utils.sample_uniform(
        scale_range[0],
        scale_range[1],
        (len(env_ids), len(body_ids)),
        device="cpu",
    )
    index = (env_ids[:, None], body_ids[None, :])
    # ④ 乘以因子（同一刚体的质量×f，惯量×f）
    masses[index] *= factors
    inertias[index] *= factors.unsqueeze(-1)
    # ⑤ 写回 PhysX
    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)
    # ⑥ 缓存到 GPU，供 critic 特权观测使用（避免每策略步 CPU readback）
    env._l5a_current_body_mass = masses.to(asset.device)


def randomize_rigid_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """独立随机每个选中刚体的质心偏移。

    ``com_ranges`` 使用 x/y/z 三轴键，未提供的轴默认偏移为零。采样值是叠加到
    当前 PhysX COM 上的相对偏移，而不是绝对质心坐标，因此可与 USD 中原始质心
    以及其他 startup 随机化组合。每个环境、每个刚体、每个轴均独立采样。

    ``distribution`` 选择采样分布；范围及 PhysX 属性读写位于 CPU，因为刚体属性
    API 不走常规的逐步 GPU 状态缓冲。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # ① 解析 env_ids 和 body_ids（CPU 侧操作）
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    else:
        body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")

    # ② 选择采样分布
    if distribution == "uniform":
        sample_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        sample_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        sample_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unsupported COM randomization distribution: {distribution}")

    # ③ 按 x/y/z 轴读取范围（缺省轴偏移为零）
    ranges = torch.tensor(
        [com_ranges.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
        device="cpu",
    )
    # ④ 采样偏移量：[envs, bodies, 3]
    samples = sample_fn(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), len(body_ids), 3),
        device="cpu",
    )
    # ⑤ 读取当前 COM → 叠加偏移 → 写回
    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids[None, :], :3] += samples
    asset.root_physx_view.set_coms(coms, env_ids)
