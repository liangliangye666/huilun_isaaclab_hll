# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""L5A balance/WF 任务的奖励项。

所有函数都返回逐环境的一维张量 ``[num_envs]``，这里只计算"未加权指标"。
它究竟是奖励还是惩罚由环境配置中对应 ``RewardTermCfg.weight`` 的正负决定：
名称含 ``exp`` 的项通常位于 0--1、配正权重；名称含 ``l1/l2`` 的误差或能耗项
通常为非负数、配负权重。调参时应同时检查函数数值尺度与配置权重，不能只看名称。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _body_pos_b(asset: Articulation, body_ids: list[int] | slice) -> torch.Tensor:
    """把选中刚体位置从世界系转换到机器人基座坐标系。

    先减去 root 世界位置得到相对向量，再用 root 四元数的逆旋转消除机体姿态。
    后续轮距/轮高奖励因此描述"相对机身的腿轮构型"，不会把机器人全局位置或
    朝向误当成几何误差。
    """
    # ① 取出世界系刚体位置 [N, num_bodies, 3]
    body_pos_w = asset.data.body_pos_w[:, body_ids, :]
    # ② 减去基座世界位置 → 世界系中的相对向量
    rel_pos_w = body_pos_w - asset.data.root_pos_w[:, None, :]
    # ③ 用基座四元数的逆消除机体旋转 → 基座系坐标
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, rel_pos_w.shape[1], -1)
    return quat_apply_inverse(root_quat_w, rel_pos_w)


'''
基座高度 L1 惩罚
'''
def base_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """计算基座世界系高度相对目标高度的 L1 误差。

    输出为 ``|z_base - target_height|``，通常配置负权重作为高度惩罚。与轮高奖励
    不同，这里直接约束基座相对地面的绝对高度。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 2] - target_height)


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """指数奖励基座系前向速度对 x 指令的跟踪。

    指标为 ``exp(-(v_cmd_x-v_base_x)^2/std^2)``：完全跟踪时为 1，误差相对
    ``std`` 增大时快速衰减。使用基座系速度，使指令含义始终是机器人自身前向。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # ① 计算前向速度误差的平方：δ² = (v_cmd_x - v_base_x)²
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    # ② 指数核：exp(-δ² / σ²)，误差越大奖励越接近 0
    return torch.exp(-lin_vel_error / std**2)


'''
基座和轮子之间的相对腿长
在 balance_env_cfg.py 中使用，可删除
'''
def nominal_wheel_height_exp(
    env: ManagerBasedRLEnv,
    target_base_height: float,
    wheel_radius: float,
    std: float,
    speed_attenuation_std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """奖励两轮相对基座保持名义垂向位置。

    名义轮心高度为 ``-(target_base_height-wheel_radius)``；分别对左右轮的 z
    误差施加指数核后取均值。结果再按指令速度模长衰减，使高速运动时不过度要求
    静态轮高精度，延续原 IsaacGym L5A 任务的设计。这里的坐标是基座系，因此
    主要约束腿长/构型，不会重复惩罚机器人在世界系中的平移。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # ① 获取左右轮在基座系中的位置 [N, 2, 3]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    # ② 计算名义轮心高度（基座系 z 坐标）
    target_wheel_z_b = -(target_base_height - wheel_radius)
    # ③ 左右轮高度误差的指数核取均值
    height_error = torch.square(target_wheel_z_b - wheel_pos_b[..., 2])
    base_reward = torch.mean(torch.exp(-height_error / std**2), dim=1)
    # ④ 速度门控：速度指令越大，轮高奖励权重越弱
    vel_cmd = env.command_manager.get_command(command_name)
    vel_norm = torch.norm(vel_cmd[:, :3], dim=1)
    return base_reward * torch.exp(-torch.square(vel_norm) / speed_attenuation_std**2)

'''
基座系对称性
    _body_pos_b
        所有身体几何奖励（对称性、同轴、轮距、基座投影）都先用这个函数把世界系坐标转换为基座系坐标，再计算误差。这是和 Tron2 最大的技术差异。
    对比 Tron2：
        Tron2 的 leg_symmetry 用世界系坐标。（实际并不是，都是基于基坐标系计算的）
        当机器人 yaw 旋转了 45°时，世界系中左右轮的 y 坐标差会受旋转影响，而基座系中始终正确反映「相对于机器人自身，两轮是否对称」。
        基座系坐标让几何奖励对机器人全局朝向不变。
'''
def leg_y_symmetry_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励左右轮相对基座中心线具有对称的横向展开量。

    比较的是左右轮 ``|y|`` 的差，而非强制指定绝对轮距；完全镜像时返回 1。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    symmetry_error = torch.abs(wheel_pos_b[:, 0, 1]) - torch.abs(wheel_pos_b[:, 1, 1])
    return torch.exp(-torch.square(symmetry_error) / std**2)


'''
两轮前后对齐
同样是基座系坐标。确保左右轮在机器人前进方向（基座 x 轴）上位置一致，防止出现「左轮在前、右轮在后」的歪轴构型。
'''
def same_wheel_x_position_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """计算两轮基座系前后位置不一致的 L1 误差。

    通常配负权重，抑制一侧轮相对另一侧明显前伸/后缩，保持两轮轴线对齐。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    return torch.abs(wheel_pos_b[:, 0, 0] - wheel_pos_b[:, 1, 0])


def same_wheel_z_position_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """计算两轮基座系垂向位置差的平方。

    通常配负权重，抑制左右腿高度不一致导致的机身侧倾构型。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    return torch.square(wheel_pos_b[:, 0, 2] - wheel_pos_b[:, 1, 2])


def wheel_distance_range_l1(
    env: ManagerBasedRLEnv,
    min_distance: float,
    max_distance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """计算两轮平面距离超出允许轮距区间的 L1 越界量。

    距离取两轮世界系 xy 坐标的欧氏距离；位于
    ``[min_distance, max_distance]`` 内时为零，只惩罚超出安全/可用支撑宽度的
    部分。通常配负权重。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    wheel_distance_xy = torch.norm(wheel_pos_w[:, 0, :2] - wheel_pos_w[:, 1, :2], dim=-1)
    lower_error = torch.clamp(min_distance - wheel_distance_xy, min=0.0)
    upper_error = torch.clamp(wheel_distance_xy - max_distance, min=0.0)
    return lower_error + upper_error


'''
轮距
'''
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
    """同时奖励合法轮距范围与 L5A 名义轮距。

    ``range_reward`` 只关心是否越出最小/最大轮距；``nominal_reward`` 进一步
    把轮距拉向 ``desired_distance``。横向速度指令增大时，名义轮距部分逐渐降权，
    为侧向机动预留构型变化空间，而合法范围奖励始终保留。当前 WF 若固定
    ``v_y=0``，名义轮距权重将保持最大。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_b = _body_pos_b(asset, asset_cfg.body_ids)
    distance = torch.abs(wheel_pos_b[:, 0, 1] - wheel_pos_b[:, 1, 1])
    # 范围越界惩罚
    outside_error = torch.clamp(min_distance - distance, min=0.0)
    outside_error += torch.clamp(distance - max_distance, min=0.0)
    range_reward = torch.exp(-torch.square(outside_error) / std**2)
    # 名义轮距奖励
    nominal_reward = torch.exp(-torch.square(distance - desired_distance) / std**2)

    # 横向速度指令越大 → 名义轮距权重越小
    lateral_command = torch.abs(env.command_manager.get_command(command_name)[:, 1])
    nominal_weight = 1.0 - torch.clamp(lateral_command / lateral_command_scale, 0.0, 1.0)
    return 0.5 * (range_reward + nominal_weight * nominal_reward)
'''
这个函数做了三件事：
    功能	    说明
    范围约束	轮距超出 [0.27m, 0.30m] 就惩罚，不管离名义值多远
    名义吸引	轮距接近 0.28m 时额外奖励
    横向衰减	当有横向速度指令（v_y 非零）时，名义轮距奖励降权，允许腿为了横移而改变轮距
为什么 L5A 需要这个？
    Tron2 是双足轮式，轮子固定在脚底。
    L5A 是腿上有轮子，腿关节运动会改变轮距。
    L5A 需要在「保持合理轮距」和「允许腿运动改变轮距以适应机动」之间做平衡。
    横向衰减就是为此设计的——当你需要横移时，松一下轮距约束。
和Tron2 的 distance_aligned的区别有很多，其中一个就是
    使用的坐标系不同
        Tron2：yaw 对齐坐标系
            heading_aligned = yaw_quat(base_quat)
            只消除 yaw，不消除 roll/pitch。
            它测量的是近似地面水平面中的横向距离。
        L5A：完整基座系
            _body_pos_b()
            消除完整的 roll、pitch、yaw。
            它测量的是轮子相对于机身自身 y 轴的距离。
        为什么不同？
            Tron2 更关心：
                两只脚在重力水平面中的支撑宽度。
            L5A 更关心：
                两轮相对机器人本体的机械构型宽度。
            两者各有合理性。
'''


'''
线速度和角速度独立门控
当指令是「只旋转不平移」时，只惩罚多余的平移，不惩罚旋转；反过来也一样。
'''
def stand_still_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """在相应指令接近零时惩罚基座残余运动。

    线速度和偏航分别门控：平面线速度指令低于阈值时惩罚实际 xy 线速度；yaw
    指令低于阈值时惩罚实际 z 角速度。二者条件相互独立，因此可以在"只旋转"
    指令下抑制平移，也可在"只平移"指令下抑制无关偏航。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # ① 线速度门控：xy 指令接近零 → True，否则 False
    lin_standing = torch.norm(command[:, :2], dim=1) < lin_threshold
    # ② 偏航门控：omega_z 指令接近零 → True，否则 False
    yaw_standing = torch.abs(command[:, 2]) < ang_threshold
    # ③ 线速度惩罚：实际平面速度 × 线速度门
    lin_penalty = torch.sum(torch.abs(asset.data.root_lin_vel_b[:, :2]), dim=1) * lin_standing
    # ④ 偏航惩罚：实际 z 角速度 × 偏航门
    yaw_penalty = torch.abs(asset.data.root_ang_vel_b[:, 2]) * yaw_standing
    # ⑤ 二者求和（互不干扰）
    return lin_penalty + yaw_penalty
'''
与tron2的区别：
    项目	        L5A	        Tron2
    实际线速度	    基座系	    世界系
    实际角速度	    基座系	    世界系
    command 名称   可传入	   写死 "base_velocity"
    可复用性	    更高	    稍低
'''


'''
基座投影约束
鼓励基座的水平投影尽量靠近两轮支撑面的中点。对双轮足平衡至关重要——基座越靠近轮轴连线中点，抗倾覆能力越强。
'''
def base_projection_at_wheel_midpoint_exp(
    env: ManagerBasedRLEnv,
    std: float,
    wheel_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励基座水平投影靠近两轮支撑中点。

    在世界系 xy 平面计算基座到左右轮中点的平方距离并套指数核。该项把基座
    参考点的投影约束在窄支撑区域附近，有助于双轮足保持静态/动态平衡。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_w = asset.data.body_pos_w[:, wheel_cfg.body_ids, :2]
    midpoint_xy = torch.mean(wheel_pos_w, dim=1)
    error = torch.sum(torch.square(asset.data.root_pos_w[:, :2] - midpoint_xy), dim=1)
    return torch.exp(-error / std**2)


'''
使用 applied_torque（经过执行器限制裁剪后的实际输出力矩），而不是原始关节力矩。这更接近真机上电机实际消耗的功率。
'''
def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """计算选中关节绝对机械功率之和 ``sum(|tau*qdot|)``。

    通常配负权重以降低能耗和激烈驱动。这里使用 actuator model 的
    ``applied_torque``，已反映执行器限制，不是原始策略动作幅值。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def joint_deviation_from_default_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """计算选中关节相对当前默认姿态的平方偏差和。

    默认姿态可能已被 ``RandomizedDefaultJointPositionAction`` 写入持久零位误差，
    因此本项约束的是每个环境各自的校准后基准，而不是硬编码的 USD 角度。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(error), dim=1)


'''
二阶动作平滑（抑制 jerk）
    对比：action_rate_l2 惩罚一阶变化（相邻动作差），action_smooth_l2 惩罚二阶变化（变化率的变化）。
通俗理解：
    action_rate：不许突然改变动作 → 动作值不能跳变
    action_smooth：不许突然改变「改变动作的速度」 → 动作值可以匀速变化，但不能突然加速变化
'''
def action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """计算动作二阶差分的平方和，抑制控制指令突变。

    公式为 ``sum((a_t - 2*a_(t-1) + a_(t-2))^2)``，与原 IsaacGym L5A
    ``_reward_action_smooth`` 一致。相比只惩罚一阶 action rate，它允许匀速变化，
    重点惩罚动作变化率的突变，通常配负权重。

    此函数在 ``env._prev_prev_action`` 中维护跨策略步状态；它不是普通无状态奖励：

    * 第一次调用时按 ``[num_envs, action_dim]`` 创建缓存；
    * 异步 reset 后前几个 step 通过 ``episode_length_buf`` 屏蔽旧回合内容；
    * 每次奖励计算末尾把 ``prev_action`` 旋入缓存，供下一策略步使用。

    因为调用会推进缓存，同一环境每个 reward step 应只配置一个该函数实例。
    """
    # ① 首次调用：分配 t-2 时刻的动作缓存
    if not hasattr(env, "_prev_prev_action"):
        env._prev_prev_action = torch.zeros(env.num_envs, env.action_manager.action.shape[-1], device=env.device)

    # ② 回合前几步的 t-2 无效（跨回合残余），屏蔽后置零
    is_early = env.episode_length_buf < 3
    prev_prev = env._prev_prev_action.clone()
    prev_prev[is_early] = 0.0

    # ③ 计算二阶差分：(a_t - 2*a_(t-1) + a_(t-2))²
    penalty = torch.sum(
        torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + prev_prev),
        dim=1,
    )
    penalty[is_early] = 0.0

    # ④ 推进缓存：本轮 t-1 → 下轮 t-2（必须在计算后更新，否则时间错位）
    env._prev_prev_action = env.action_manager.prev_action.clone()
    return penalty
