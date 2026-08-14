# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""上楼梯任务的两阶段目标与速度命令。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UpstairsVelocityCommand(UniformVelocityCommand):
    """区分平地/障碍地形并实时朝向两阶段目标的三维速度命令。"""

    cfg: UpstairsVelocityCommandCfg

    def __init__(self, cfg: UpstairsVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        '''
        目标点位置
            形状为 [4096, 2, 3] 的张量：
                4096 个并行环境
                每个环境有 2 个目标点（起点附近一个 + 前方一个，形成"从 A 走到 B"的导航任务）
                每个目标点是 3 维世界坐标 (x, y, z)
            后缀 _w 表示 world frame（世界坐标系）。
        '''
        self.goal_positions_w = torch.zeros(self.num_envs, 2, 3, device=self.device)
        '''
        当前目标索引
            形状为 [4096] 的整数张量，记录每个环境当前正在追踪第几个目标点（0 或 1）。机器人到达目标 0 后，索引切换为 1，开始追踪下一个目标。
        '''
        self.current_goal_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        '''
        目标到达计时器
            形状为 [4096] 的浮点张量，记录每个环境中机器人在目标范围内停留了多久（单位：秒）。
            配合配置中的 goal_reach_delay=0.1 使用：机器人必须持续在目标 0.2 米半径内停留 0.1 秒，才算真正到达。
            这个计时器就是用来做这个判断的——防止机器人只是"路过"目标点就被误判为到达。
        '''
        self.goal_reach_time = torch.zeros(self.num_envs, device=self.device)
        '''
        台阶高度
            形状为 [4096] 的浮点张量，记录每个环境当前所处台阶的高度。这个值会随课程学习变化（低难度 2cm → 高难度 11cm）。
            这个信息最终会进入 Critic 的特权观测，让 Critic 知道当前台阶有多高，从而更准确地评估状态价值。
        '''
        self.terrain_step_height = torch.zeros(self.num_envs, device=self.device)
        # Reuse the environment index tensor when selecting each environment's
        # active goal.  Rebuilding it in every property access launches an
        # unnecessary CUDA kernel in several reward/observation terms per step.
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        '''
        目标距离指标
            记录每个环境中机器人到当前目标点的距离。
            self.metrics 是父类中定义的字典，用于存储各种监控指标。这个值会被 TensorBoard 记录，方便观察训练过程中机器人是否在逐步靠近目标。
        '''
        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        '''
        目标索引指标
            记录每个环境当前追踪的目标索引（0 还是 1）。
            用于监控——如果大量环境的 goal_index 始终为 0（没有到达过任何目标），说明策略根本没学会导航。
        '''
        self.metrics["goal_index"] = torch.zeros(self.num_envs, device=self.device)

    # 返回每个环境当前正在追踪的目标点的世界坐标。
    @property
    def current_goal_w(self) -> torch.Tensor:
        """当前阶段目标的世界坐标，shape 为 ``[N, 3]``。"""
        return self.goal_positions_w[self._all_env_ids, self.current_goal_index]
    '''
    回顾数据结构：
        self.g`oal_positions_w 形状是 [4096, 2, 3]
            维度 0：环境 ID（0~4095）
            维度 1：目标编号（0 或 1）
            维度 2：坐标 xyz
        env_ids 是 [0, 1, 2, ..., 4095]
        self.current_goa`l_index 形状是 [4096]，每个元素是 0 或 1

    PyTorch 的高级索引规则：当你用两个同形状的 1D 张量去索引一个 3D 张量时，会逐对匹配。
        goal_positions_w[ [0, 1, 2, ..., 4095], [1, 0, 1, ..., 0] ]
                                ↑                     ↑
                            环境 ID 列表          每个环境的目标编号
        结果：
        环境 0 取 goal_positions_w[0, 1] → 环境0的目标1坐标
        环境 1 取 goal_positions_w[1, 0] → 环境1的目标0坐标
        环境 2 取 goal_positions_w[2, 1] → 环境2的目标1坐标
        ...

    最终输出形状 [4096, 3]——4096 个环境，每个环境一个 3D 坐标。
    '''

    # 判断每个环境当前处于平地还是楼梯上
    # 楼梯任务的地形是一个网格：第 0 列是平地，其余列是楼梯。这个函数返回一个布尔张量，标记哪些环境当前在平地上。
    @property
    def is_flat_env(self) -> torch.Tensor:      # 返回一个形状为 [N] 的布尔张量（torch.bool），True 表示该环境在平地上，False 表示在楼梯上。
        """由地形列判断当前环境是否属于 10% 平地区域。"""
        terrain = self._env.scene.terrain
        terrain_types = getattr(terrain, "terrain_types", None)     # terrain_types 是一个形状为 [num_envs] 的张量，每个元素是该环境所在地形的列号（0 是平地，1~9 是不同难度的楼梯）。
        if terrain_types is None:   # 如果地形对象没有 terrain_types 属性（比如纯平面地形），说明没有楼梯这个概念，所有环境都是平地。返回全 True 的张量——"大家都算平地"。
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return terrain_types < self.cfg.flat_column_count   # 地形列号 < 1 → 只有第 0 列 → 平地区域
    '''
    为什么需要这个判断？
        这个函数被用于两个地方：
            速度命令：平地可以要求后退（负速度），楼梯上只能前进——因为轮足机器人倒退下楼梯非常危险
            站立命令：只在平地上要求机器人站立（flat_standing_probability=0.1），楼梯上不允许站立——因为台阶上站着本身就很难
    '''

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self.current_goal_index[env_ids] = 0    # 重置目标索引
        self.goal_reach_time[env_ids] = 0.0     # 重置到达计时器
        self._reset_goals(env_ids)              # 生成新目标点
        return super().reset(env_ids)

    # 为重置的环境生成新的目标点位置
    def _reset_goals(self, env_ids: Sequence[int] | slice) -> None:
        '''
        当环境重置时，这个方法为每个环境生成 2 个目标点：
            第一阶段目标：在机器人前方 4 米处，可能带侧向偏移（让策略学会在楼梯上转弯）
            第二阶段目标：在机器人前方 8 米处，且必须回到地形中心线（让策略学会"到达后再回来"）
        同时根据当前台阶高度，把台阶信息写入 terrain_step_height，供 Critic 使用。
        '''
        # ========== 第一阶段：设置目标点的 x 坐标（前进距离） ==============
        terrain = self._env.scene.terrain
        origins = self._env.scene.env_origins[env_ids]  # 环境的原点坐标,形状 [count, 3]
        count = origins.shape[0]    # 需要重置的环境数量
        self.goal_positions_w[env_ids] = origins[:, None, :]    # origins[:, None, :] 在形状 [count, 3] 中间插入一维变成 [count, 1, 3]，这样才能广播赋值给形状为 [count, 2, 3] 的 goal_positions_w
        self.goal_positions_w[env_ids, 0, 0] += self.cfg.goal_forward_distances[0]  # 第一个目标的 x 坐标 = 原点 x + 4 米
        self.goal_positions_w[env_ids, 1, 0] += self.cfg.goal_forward_distances[1]  # 第二个目标的 x 坐标 = 原点 x + 8 米

        # ========== 第二阶段：计算台阶高度 ==============
        levels = getattr(terrain, "terrain_levels", None)   # 获取地形难度等级
        if levels is None:
            difficulty = torch.zeros(count, device=self.device)
        else:
            # 旧任务使用 row / num_rows，因此第 0--9 行对应 difficulty 0.0--0.9。
            difficulty = levels[env_ids].float() / float(self.cfg.terrain_num_rows) # 难度 = 当前行号 / 总行数。10 行对应 0.0~0.9 的难度范围
        step_min, step_max = self.cfg.step_height_range     # 获取台阶高度范围 2cm ~ 11cm
        normalized = torch.clamp(difficulty / self.cfg.max_difficulty, 0.0, 1.0)    # 把难度归一化到 [0, 1]，除以 max_difficulty=0.9 后 clamp 防止超过 1.0
        step_height = step_min + normalized * (step_max - step_min) # 台阶高度线性插值：难度 0 → 2cm，难度 0.9 → 11cm
        flat_mask = self.is_flat_env[env_ids]   # 获取这些环境是否在平地上
        self.terrain_step_height[env_ids] = torch.where(flat_mask, 0.0, step_height)    # 平地台阶高度为 0，只有楼梯上的环境才记录实际台阶高度

        # ========== 第三阶段：计算目标点的侧向偏移（y 坐标） ==============
        if self.cfg.fixed_goal_lateral_offset is None:  # 情况 A：随机侧向偏移
            random_values = torch.rand(count, 3, device=self.device)    # 形状 [count, 3],为每个环境生成 3 个 [0,1) 的随机数，分别用于：决定方向（左/右）、决定偏移距离、决定是否在中心
            side = torch.where(random_values[:, 0] < 0.5, -1.0, 1.0)    # 50% 向左，50% 向右
            lateral_min, lateral_max = self.cfg.goal_lateral_range
            lateral = lateral_min + random_values[:, 1] * (lateral_max - lateral_min)   # 偏移距离在 [1.0, 4.0] 米之间
            lateral *= 1.0 - self.cfg.goal_lateral_difficulty_scale * difficulty    # 	课程学习缩放：难度越低，侧向偏移越大（让策略在简单台阶上练转弯）；难度越高，侧向偏移越小（高难度台阶上先保证直走）。
            lateral = torch.where(random_values[:, 2] < self.cfg.goal_center_probability, 0.0, side * lateral)  # 20% 概率目标在正前方，其余 80% 加上侧向偏移。这保证了策略既有"直走"的训练样本，也有"转弯"的训练样本
        else:                                           # 情况 B：固定侧向偏移（调试模式）
            lateral = torch.full(
                (count,), float(self.cfg.fixed_goal_lateral_offset), device=self.device, dtype=origins.dtype    # 所有目标统一向右偏移一定距离
            )
        self.goal_positions_w[env_ids, 0, 1] += lateral
        # 第二阶段重新回到地形中心线，保持旧任务的两阶段目标语义。
        self.goal_positions_w[env_ids, 1, 1] = origins[:, 1]
        '''
        目标 0: (原点.x + 4.0, 原点.y + 侧向偏移, 原点.z)
        目标 1: (原点.x + 8.0, 原点.y,          原点.z)
        为什么这样设计？ 两阶段目标语义：
            第一阶段：走到前方 4 米、可能偏左或偏右的位置（练转弯）
            第二阶段：从偏位走到前方 8 米的中心线位置（练回调）
        '''

    # 重新采样速度指令
    def _resample_command(self, env_ids: Sequence[int]) -> None:
        '''
        每个环境每隔一段时间（配置中 resampling_time=10s）会调用这个方法，为它生成一个新的速度指令。核心差异在于：
            平地：速度范围 [-1.0, 1.0] m/s，可以后退
            楼梯：速度范围 [0.1, 1.0] m/s，只能前进（倒退下楼梯太危险）
        此外，平地上还有 10% 概率下达"原地站立"指令。
        '''
        count = len(env_ids)
        random_values = torch.empty(count, 3, device=self.device)
        '''
        torch.empty 创建一块未初始化的显存，形状 [count, 3]。三列分别用于：
            列	        用途
            [:, 0]	    平地速度的随机数
            [:, 1]	    楼梯速度的随机数
            [:, 2]	    站立指令的随机数（只在平地上使用）
        '''
        flat_mask = self.is_flat_env[env_ids]   # 获取平地/楼梯掩码

        flat_speed = random_values[:, 0].uniform_(*self.cfg.ranges.lin_vel_x)
        '''
        uniform_ 是 PyTorch 的原地随机填充方法（注意末尾下划线表示原地操作）。
        *self.cfg.ranges.lin_vel_x 中的 * 是 Python 的解包运算符。lin_vel_x = (-1.0, 1.0)，* 把它解包成两个参数：
            # 等价于：
            flat_speed = random_values[:, 0].uniform_(-1.0, 1.0)
        在 [-1.0, 1.0] 范围内均匀采样前进速度。
        '''
        upstairs_speed = random_values[:, 1].uniform_(*self.cfg.nonflat_lin_vel_x)
        self.vel_command_b[env_ids, 0] = torch.where(flat_mask, flat_speed, upstairs_speed)
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = 0.0
        self.is_heading_env[env_ids] = True     # 所有楼梯任务环境都使用航向控制模式
        standing_draw = random_values[:, 2].uniform_(0.0, 1.0)  # 为每个环境生成一个 [0, 1) 的随机数，用作站立抽签
        self.is_standing_env[env_ids] = flat_mask & (standing_draw <= self.cfg.flat_standing_probability)   # 随机抽中（10% 概率，flat_standing_probability=0.1）

    # 更新监控指标
    def _update_metrics(self) -> None:
        '''
        在每个策略步（50Hz）被调用，更新两个监控指标：
            目标距离积分：累积"机器人离目标有多远"的时间积分，用于计算 episode 平均距离
            目标索引：当前在追第几个目标（0 还是 1）
        这些指标最终会被 TensorBoard 记录，让你在训练时观察策略的导航能力是否在提升。
        '''
        super()._update_metrics()
        goal_delta = self.current_goal_w[:, :2] - self.robot.data.root_pos_w[:, :2] # 从机器人指向目标点的水平向量 = 每个环境当前目标点的世界坐标 (x, y) - 每个环境机器人基座的世界坐标
        self.metrics["goal_distance"] += torch.linalg.vector_norm(goal_delta, dim=1) * self._env.step_dt    # 计算每个环境的 √(dx² + dy²)，乘以策略步的时间步长（0.02 秒）,累加到积分器中
        '''
        为什么要乘以 step_dt？ 这是把"距离"转换成"距离×时间"的积分：
            普通求平均（错误）：
            平均距离 = (d₁ + d₂ + d₃ + ... + dₙ) / n
            问题：如果 episode 因为摔倒提前终止，平均值会偏小

            积分求平均（正确）：
            距离积分 = d₁×dt + d₂×dt + d₃×dt + ... + dₙ×dt
            平均距离 = 距离积分 / (n × dt) = 距离积分 / episode时长
            好处：与 episode 长度无关，不同长度的 episode 可以公平比较
        通俗类比：
            你在开车去目的地，每分钟记录一次剩余距离。
            要计算"平均剩余距离"，不能简单地 (d₁ + d₂ + ...) / 分钟数，因为如果你开了 60 分钟和我开了 10 分钟，分母不同。
            正确做法是算"距离曲线下的面积"——这就是 += 距离 × dt 在做的事。
        += 距离 × dt 的积分模式：
            这是强化学习监控代码中的经典技巧。
            相比于直接求平均，积分法不受 episode 长度影响，在 TensorBoard 中画出的曲线更平滑、更可比较。
            如果你在训练时看到 goal_distance 指标从 3.0 逐渐降到 1.5，说明策略在 episode 中的平均目标距离在减小——机器人越来越快地到达目标点。
        '''
        self.metrics["goal_index"][:] = self.current_goal_index.float()
        '''
        为什么用 [:] 而不是直接 =？
            # 方式 A：用 [:] 原地修改（正确）
            self.metrics["goal_index"][:] = self.current_goal_index.float()

            # 方式 B：直接赋值（错误！）
            self.metrics["goal_index"] = self.current_goal_index.float()

            方式 B 会替换字典中的引用，导致后续的 logging 代码拿到的是新张量，而之前可能已经有其他代码持有了旧张量的引用——造成不一致。
            方式 A 修改的是同一个张量的内容，所有引用者都能看到最新值。
        '''

    def _update_command(self) -> None:
        # =========== 第一阶段：目标到达检测 ==========
        goal_delta = self.current_goal_w[:, :2] - self.robot.data.root_pos_w[:, :2] # 从机器人指向目标点的水平向量 = 每个环境当前目标点的世界坐标 (x, y) - 每个环境机器人基座的世界坐标
        goal_distance = torch.linalg.vector_norm(goal_delta, dim=1) # 计算每个环境机器人到当前目标点的水平距离 √(dx² + dy²)
        within_goal = goal_distance < self.cfg.goal_reach_radius    # 判断是否进入目标范围
        self.goal_reach_time = torch.where(     # 更新到达计时器
            within_goal,
            self.goal_reach_time + self._env.step_dt,   # 在范围内,累加停留时间
            torch.zeros_like(self.goal_reach_time),     # 离开范围,计时器立即清零
        )
        # 判断是否应该切换到下一目标
        '''
        两个条件同时满足才切换：
            连续在目标范围内停留了 0.1 秒
            当前还在第一阶段（追目标 0）。如果已经是第二阶段（追目标 1），到达后就结束了，不再切换
        '''
        advance = (self.goal_reach_time >= self.cfg.goal_reach_delay) & (self.current_goal_index == 0)
        self.current_goal_index[advance] = 1    # 切换到第二阶段目标
        self.goal_reach_time[advance] = 0.0     # 计时器清零，开始为第二阶段目标的到达计时

        # =========== 第二阶段：航向控制 ==========
        goal_delta = self.current_goal_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        self.heading_target[:] = torch.atan2(goal_delta[:, 1], goal_delta[:, 0])    # 计算目标方向角,atan2(y, x) 计算从机器人指向目标点的方向角（弧度），范围 [-π, π]
        heading_error = math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        '''
        部分	                        含义
        self.heading_target	            目标方向角（指向目标点的方向）
        self.robot.data.heading_w	    机器人当前的朝向角
        heading_target - heading_w	    航向误差（机器人需要转多少角度才能面向目标）
        wrap_to_pi(...)	                角度包裹到 [-π, π] 范围
        '''
        self.vel_command_b[:, 2] = torch.clamp(     # 生成角速度指令
            self.cfg.heading_control_stiffness * heading_error,
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )
        self.vel_command_b[self.is_standing_env] = 0.0  # 被标记为站立的环境，三维速度指令全部清零 [0, 0, 0]——原地不动。


@configclass
class UpstairsVelocityCommandCfg(UniformVelocityCommandCfg):
    """上楼梯速度命令配置。"""

    class_type: type = UpstairsVelocityCommand

    asset_name: str = MISSING
    nonflat_lin_vel_x: tuple[float, float] = (0.1, 1.0)     # 楼梯速度控制，非平地上前进速度的范围：0.1 ~ 1.0 m/s，注意这里只有正值——楼梯上不要求后退，只前进。
    flat_standing_probability: float = 0.1                  # 平地站立比例，在平地区域，10% 的环境会收到"原地站立"指令（速度为零）。这保证策略不会忘记怎么站稳，而不是只会一直走。
    # 地形布局
    flat_column_count: int = 1                              # 地形网格中前 1 列是平地，其余列是楼梯。机器人在平地列中学习站立和基本行走，在楼梯列中学习爬楼梯。
    terrain_num_rows: int = 10                              # 总共 10 行（难度等级），配合课程学习逐步增加难度。
    step_height_range: tuple[float, float] = (0.02, 0.11)   # 台阶高度范围，台阶高度从 2cm 到 11cm。配合课程学习，低难度时台阶矮（2cm），高难度时台阶高（11cm）。
    max_difficulty: float = 0.9                             # 课程难度上限
    '''
    地形等级 vs 难度系数
        地形等级 (terrain_level)：0, 1, 2, ..., 9（共 10 级，离散）
            ↓ 归一化
        难度系数 (difficulty)：  0.0, 0.1, 0.2, ..., 0.9（连续值）
        地形等级是离散的（你被分到了第几级台阶），但台阶高度是连续的（2cm~11cm 之间的任意值）。

    max_difficulty 控制的是台阶高度增长的上限，不是地形等级的上限。
        三种情况对比
            假设地形等级 = 9（最高级），difficulty = 0.9：
                max_difficulty	    normalized = 0.9 / max_difficulty	    台阶高度
                1.0（不设上限）	        0.9 / 1.0 = 0.9	                        2 + 0.9×9 = 10.1 cm
                0.9（当前值）	        0.9 / 0.9 = 1.0（被 clamp 住）	        2 + 1.0×9 = 11.0 cm（满额）
                0.6	                  0.9 / 0.6 = 1.5 → clamp 到 1.0	    2 + 1.0×9 = 11.0 cm（早就满额了）
    它的真正作用
        max_difficulty 控制的是**"多快达到最大台阶高度"**：
            max_difficulty = 0.9：地形等级从 0 到 9，台阶高度均匀增长，每升一级加约 1cm
            max_difficulty = 0.6：地形等级 6 以后台阶高度就不再增加了（提前到顶）
    为什么要提前到顶？
        答案是给策略更多时间在高难度台阶上训练。
    '''
    # 目标点导航
    goal_forward_distances: tuple[float, float] = (4.0, 8.0)# 目标点位于机器人前方 4~8 米处。策略需要导航到这个目标点，而不仅仅是往前走——这比纯速度跟踪更难，因为要走楼梯的同时保持方向。
    goal_center_probability: float = 0.2                    # 20% 的目标点位于正前方（零侧向偏移），让策略在简单场景下也有训练样本。
    goal_lateral_range: tuple[float, float] = (1.0, 4.0)    # 其余 80% 的目标点随机偏移 1~4 米到左侧或右侧，强制策略学会在楼梯上横向调整方向。
    goal_lateral_difficulty_scale: float = 0.5              # 课程学习时，侧向偏移的难度增速是主难度的 50%。
    # 目标到达判定
    goal_reach_radius: float = 0.2                          # 机器人基座进入目标点 0.2 米范围内就算"到达"。
    goal_reach_delay: float = 0.1                           # 需要持续在范围内停留 0.1 秒才确认到达。防止机器人只是"路过"目标点就被判定为到达——它必须真正停在那里。
    fixed_goal_lateral_offset: float | None = None          # 固定侧向偏移（调试用），None 表示使用随机偏移。如果设为具体数值（比如 1.5），所有目标点都会固定向同一侧偏移 1.5 米——这在调试和评估时很有用，可以验证策略是否能稳定地走特定方向。
