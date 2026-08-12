# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 上楼梯任务使用的资产课程地形。

旧 Isaac Gym 任务先生成高度场，再转换为三角网格。本模块保留同样的几何语义，
但把生成器改写为 IsaacLab ``TerrainGeneratorCfg`` 可直接使用的纯函数和配置类。
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

UPSTAIRS_TERRAIN_SIZE = (8.0, 8.0)
UPSTAIRS_NUM_ROWS = 10
UPSTAIRS_NUM_COLS = 10
UPSTAIRS_FLAT_COLUMNS = 1
UPSTAIRS_BLOCK_COLUMNS = 2
UPSTAIRS_STAIR_COLUMNS = 7
UPSTAIRS_MAX_DIFFICULTY = 0.9
UPSTAIRS_STEP_HEIGHT_RANGE = (0.02, 0.11)
UPSTAIRS_STEP_WIDTH_RANGE = (0.66, 0.30)


def _lerp(values: tuple[float, float], difficulty: float) -> float:
    """恢复旧任务的离散 row 难度，再在线性范围内插值。

    IsaacLab 会在每行内部加入一个随机小数；先反推出 0--9 行，可让真实几何、
    command 的当前台阶高度以及 curriculum level 始终精确对应。
    """
    generator_fraction = np.clip(difficulty / UPSTAIRS_MAX_DIFFICULTY, 0.0, 1.0)
    level = min(int(np.floor(generator_fraction * UPSTAIRS_NUM_ROWS)), UPSTAIRS_NUM_ROWS - 1)
    source_difficulty = level / UPSTAIRS_NUM_ROWS
    interpolation_fraction = source_difficulty / UPSTAIRS_MAX_DIFFICULTY
    return values[0] + interpolation_fraction * (values[1] - values[0])


def _meters_to_pixels_floor(value: float, scale: float) -> int:
    """向下量化到高度场像素，并消除 ``0.30 / 0.10 -> 2.999...`` 的浮点误差。"""
    return max(1, int(np.floor(value / scale + 1.0e-6)))


@height_field_to_mesh
def variable_pyramid_stairs_terrain(
    difficulty: float,
    cfg: VariablePyramidStairsTerrainCfg,
) -> np.ndarray:
    """生成中心低、向四周逐级升高且踏面随难度缩短的金字塔楼梯。"""
    width_pixels = _meters_to_pixels_floor(cfg.size[0], cfg.horizontal_scale)
    length_pixels = _meters_to_pixels_floor(cfg.size[1], cfg.horizontal_scale)
    step_width = _meters_to_pixels_floor(_lerp(cfg.step_width_range, difficulty), cfg.horizontal_scale)
    step_height = _meters_to_pixels_floor(_lerp(cfg.step_height_range, difficulty), cfg.vertical_scale)
    platform_width = _meters_to_pixels_floor(cfg.platform_width, cfg.horizontal_scale)

    # 负高度使中心平台处于最低点。TerrainImporter 把中心平台高度作为环境原点，
    # 机器人从中心向 +/-x 方向运动时看到的都是上行台阶。
    step_height *= -1
    height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    current_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        current_height += step_height
        height_field[start_x:stop_x, start_y:stop_y] = current_height
    return height_field


@configclass
class VariablePyramidStairsTerrainCfg(HfTerrainBaseCfg):
    """可同时课程化台阶高度和踏面宽度的高度场配置。"""

    function = variable_pyramid_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    step_width_range: tuple[float, float] = MISSING
    platform_width: float = 4.0


@height_field_to_mesh
def platform_blocks_terrain(difficulty: float, cfg: PlatformBlocksTerrainCfg) -> np.ndarray:
    """复现旧任务中心平台外按网格排列的小方块地形。"""
    width_pixels = _meters_to_pixels_floor(cfg.size[0], cfg.horizontal_scale)
    length_pixels = _meters_to_pixels_floor(cfg.size[1], cfg.horizontal_scale)
    block_height = _meters_to_pixels_floor(_lerp(cfg.block_height_range, difficulty), cfg.vertical_scale)
    block_length = _meters_to_pixels_floor(cfg.block_length, cfg.horizontal_scale)
    block_width = _meters_to_pixels_floor(cfg.block_width, cfg.horizontal_scale)
    spacing_x = _meters_to_pixels_floor(cfg.spacing[0], cfg.horizontal_scale)
    spacing_y = _meters_to_pixels_floor(cfg.spacing[1], cfg.horizontal_scale)
    jitter_x = _meters_to_pixels_floor(cfg.jitter[0], cfg.horizontal_scale)
    jitter_y = _meters_to_pixels_floor(cfg.jitter[1], cfg.horizontal_scale)
    margin = _meters_to_pixels_floor(cfg.margin, cfg.horizontal_scale)

    platform_width = _meters_to_pixels_floor(cfg.platform_width, cfg.horizontal_scale)
    platform_x1 = width_pixels // 2 - platform_width // 2
    platform_x2 = width_pixels // 2 + platform_width // 2
    platform_y1 = length_pixels // 2 - platform_width // 2
    platform_y2 = length_pixels // 2 + platform_width // 2

    seed = getattr(cfg, "seed", None)
    difficulty_seed = int(round(float(difficulty) * 1_000_000))
    rng = np.random.default_rng(None if seed is None else int(seed) + difficulty_seed)
    height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    x = margin
    while x < width_pixels - margin:
        y = margin
        while y < length_pixels - margin:
            offset_x = rng.integers(-jitter_x, jitter_x + 1) if jitter_x > 0 else 0
            offset_y = rng.integers(-jitter_y, jitter_y + 1) if jitter_y > 0 else 0
            center_x = x + int(offset_x)
            center_y = y + int(offset_y)
            x1 = max(center_x - block_length // 2, 0)
            x2 = min(center_x + block_length // 2, width_pixels)
            y1 = max(center_y - block_width // 2, 0)
            y2 = min(center_y + block_width // 2, length_pixels)
            overlaps_platform = x2 > platform_x1 and x1 < platform_x2 and y2 > platform_y1 and y1 < platform_y2
            if not overlaps_platform:
                height_field[x1:x2, y1:y2] = block_height
            y += spacing_y
        x += spacing_x

    height_field[platform_x1:platform_x2, platform_y1:platform_y2] = 0
    return height_field


@configclass
class PlatformBlocksTerrainCfg(HfTerrainBaseCfg):
    """中心平台外周期布置小方块的高度场配置。"""

    function = platform_blocks_terrain

    block_height_range: tuple[float, float] = MISSING
    block_length: float = 0.4
    block_width: float = 0.4
    spacing: tuple[float, float] = (1.5, 0.5)
    jitter: tuple[float, float] = (0.6, 0.1)
    platform_width: float = 2.0
    margin: float = 1.0
    seed: int | None = 1


UPSTAIRS_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=1,
    curriculum=True,
    size=UPSTAIRS_TERRAIN_SIZE,
    border_width=25.0,
    num_rows=UPSTAIRS_NUM_ROWS,
    num_cols=UPSTAIRS_NUM_COLS,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.1,
    difficulty_range=(0.0, UPSTAIRS_MAX_DIFFICULTY),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "platform_blocks": PlatformBlocksTerrainCfg(
            proportion=0.2,
            block_height_range=UPSTAIRS_STEP_HEIGHT_RANGE,
        ),
        "upstairs": VariablePyramidStairsTerrainCfg(
            proportion=0.7,
            step_height_range=UPSTAIRS_STEP_HEIGHT_RANGE,
            step_width_range=UPSTAIRS_STEP_WIDTH_RANGE,
            platform_width=4.0,
        ),
    },
)
"""旧 Gym 上楼任务的 10% 平地、20% 小方块、70% 上行楼梯课程。"""
