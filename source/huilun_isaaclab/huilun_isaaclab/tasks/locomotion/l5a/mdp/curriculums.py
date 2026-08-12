# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""上楼梯任务的地形课程。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_upstairs(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    move_up_fraction: float = 0.45,
    move_down_fraction: float = 0.30,
    terrain_length: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """按回合内水平位移更新地形等级，并返回平均等级供日志记录。"""
    terrain = env.scene.terrain
    if getattr(terrain, "terrain_levels", None) is None:
        return torch.zeros((), device=env.device)
    # 初始化 reset 尚无有效轨迹，不能把首批环境误降一级。
    if env.common_step_counter == 0 or len(env_ids) == 0:
        return torch.mean(terrain.terrain_levels.float())

    asset: Articulation = env.scene[asset_cfg.name]
    distance = torch.linalg.vector_norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain_length * move_up_fraction
    move_down = (distance < terrain_length * move_down_fraction) & ~move_up
    # TerrainImporter 自身负责最高等级后的随机重分配和最低等级裁剪。
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
