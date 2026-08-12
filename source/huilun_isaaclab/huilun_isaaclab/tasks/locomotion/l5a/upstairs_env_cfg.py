# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A WF 盲走上楼梯训练与 Play 配置。"""

from __future__ import annotations

import math
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from huilun_isaaclab.assets.robots.l5a import (
    BASE_BODY_NAME,
    L5A_CFG,
    L5A_MAX_TRACK_WIDTH,
    L5A_MIN_TRACK_WIDTH,
    L5A_NOMINAL_BASE_HEIGHT,
    L5A_NOMINAL_TRACK_WIDTH,
    L5A_WHEEL_RADIUS,
    LEG_JOINT_NAMES,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
)
from huilun_isaaclab.assets.terrains.upstairs_terrains import (
    UPSTAIRS_MAX_DIFFICULTY,
    UPSTAIRS_NUM_ROWS,
    UPSTAIRS_STEP_HEIGHT_RANGE,
    UPSTAIRS_TERRAIN_SIZE,
    UPSTAIRS_TERRAINS_CFG,
)

from . import mdp
from .wf_flat_env_cfg import (
    ActionsCfg,
    EventCfg,
    L5AWFFlatEnvCfg,
    L5AWFSceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    build_l5a_wf_deployment_metadata,
    build_l5a_wf_export_metadata,
)


@configclass
class L5AWFUpstairsSceneCfg(L5AWFSceneCfg):
    """10×10 课程地形、L5A、接触传感器和 77 点高度扫描。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",                   # 程序化生成地形
        terrain_generator=UPSTAIRS_TERRAINS_CFG,    # 地形生成器的具体配置
        max_init_terrain_level=5,                   # 课程学习的起始难度等级
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        debug_vis=False,
    )

    height_scanner = RayCasterCfg(                  # 高度扫描器
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_BODY_NAME}",   # 扫描器挂载在机器人的 base_link 上，跟随机器人移动
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),    # 射线起点在 base_link 上方 20 米处，向下发射。
        ray_alignment="yaw",    # 射线网格只跟随机器人的偏航角（yaw）旋转，不跟随 roll/pitch。这样测量的高度是相对于世界 z 轴的，不会因为机器人倾斜而产生错误读数
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.0, 0.6)),   # 网格采样模式：每 0.1 米一个采样点，覆盖机器人前后 1.0 米、左右 0.6 米的矩形区域。总共 (1.0/0.1+1) × (0.6/0.1+1) = 11 × 7 = 77 个采样点
        mesh_prim_paths=["/World/ground"],  # 射线只检测地形网格，忽略机器人自身和其他物体
        debug_vis=False,    # 不显示扫描射线的可视化
    )


@configclass
class UpstairsCommandsCfg:
    """两阶段目标驱动的平地双向/楼梯正向速度命令。"""

    base_velocity = mdp.UpstairsVelocityCommandCfg(
        # 基础参数（继承自父类）
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), # 每 10 秒重新采样一次速度指令
        rel_standing_envs=0.0,              # 父类的站立比例设为 0，因为子类用自己的 flat_standing_probability 来控制
        rel_heading_envs=1.0,               # 100% 的环境使用航向控制（不是角速度跟踪）
        heading_command=True,               # 启用航向控制模式
        heading_control_stiffness=2.0,      # 航向 P 控制器增益：误差 1 rad → 角速度指令 2 rad/s
        debug_vis=False,                    # 不显示调试可视化
        # 楼梯特有参数
        nonflat_lin_vel_x=(0.1, 1.0),       # 楼梯上前进速度 0.1~1.0 m/s，只有正值（不后退）
        flat_standing_probability=0.1,      # 平地上 10% 概率站立
        flat_column_count=1,                # 地形网格中第 0 列是平地，其余是楼梯
        terrain_num_rows=UPSTAIRS_NUM_ROWS,             # 地形行数（难度等级数）
        step_height_range=UPSTAIRS_STEP_HEIGHT_RANGE,   # 台阶高度范围
        max_difficulty=UPSTAIRS_MAX_DIFFICULTY,         # 课程学习最大难度
        # 目标点导航参数
        goal_forward_distances=(4.0, 8.0),  # 目标 0 在前方 4 米，目标 1 在前方 8 米
        goal_center_probability=0.2,        # 20% 概率目标在正前方（无侧向偏移）
        goal_lateral_range=(1.0, 4.0),      # 侧向偏移 1~4 米
        goal_lateral_difficulty_scale=0.5,  # 难度越高，侧向偏移越小（先保直走）
        goal_reach_radius=0.2,              # 进入目标 0.2 米范围内算"在目标附近"
        goal_reach_delay=0.1,               # 需连续停留 0.1 秒才算到达
        # 平地上的速度指令范围
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.0),          # 平地上可以后退 -0.6 m/s 到前进 1.0 m/s
            lin_vel_y=(0.0, 0.0),           # 不要求侧向移动
            ang_vel_z=(-1.0, 1.0),          # 角速度限制 ±1.0 rad/s（虽然实际用的是航向控制生成的角速度，这是硬限制）
            heading=(-math.pi, math.pi),    # 航向目标可以是任意方向
        ),
    )


@configclass
class UpstairsObservationsCfg(ObservationsCfg):
    """Actor 保持 28 维盲走本体观测，Critic 扩展到 156 维。"""

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        terrain_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=5.0,
        )
        target = ObsTerm(func=mdp.upstairs_goal_features, params={"command_name": "base_velocity"})
        wheel_contact_history = ObsTerm(
            func=mdp.wheel_contact_history,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True),
                "history_length": 3,
                "horizontal_force_threshold": 20.0,
            },
        )
        step_height = ObsTerm(func=mdp.upstairs_step_height, params={"command_name": "base_velocity"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class UpstairsEventCfg(EventCfg):
    """保留 WF 随机化，仅把 reset 收紧到中心平台附近。"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.35, 0.35), "yaw": (-0.15, 0.15)},
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.20, 0.20),
                "z": (-0.10, 0.10),
                "roll": (-0.20, 0.20),
                "pitch": (-0.20, 0.20),
                "yaw": (-0.25, 0.25),
            },
        },
    )


def _gait_params() -> dict[str, Any]:
    return {
        "command_name": "base_velocity",
        "wheel_radius": L5A_WHEEL_RADIUS,
        "wheel_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True),
        "height_sensor_cfg": SceneEntityCfg("height_scanner"),
    }


@configclass
class UpstairsRewardsCfg(RewardsCfg):
    """旧上楼目标/步态奖励与当前 WF 稳定性、平滑和可执行性奖励的并集。"""

    # 旧任务分别跟踪 x/y/yaw，保留其权重和核函数；覆盖 WF-Flat 的合并项。
    track_lin_vel_xy = None
    track_ang_vel_z = None
    track_lin_vel_x = RewTerm(
        func=mdp.tracking_lin_vel_x_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "sigma": 0.05},
    )
    track_lin_vel_y = RewTerm(
        func=mdp.tracking_lin_vel_y_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "sigma": 0.05},
    )
    track_ang_vel = RewTerm(
        func=mdp.tracking_ang_vel_abs_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "sigma": 0.1},
    )
    track_lin_vel_potential = RewTerm(
        func=mdp.tracking_lin_vel_x_potential,
        weight=1.0,
        params={"command_name": "base_velocity", "sigma": 0.05},
    )
    track_ang_vel_potential = RewTerm(
        func=mdp.tracking_ang_vel_potential,
        weight=1.0,
        params={"command_name": "base_velocity", "sigma": 0.1},
    )
    track_goal = RewTerm(
        func=mdp.tracking_goal_progress,
        weight=2.0,
        params={"command_name": "base_velocity"},
    )
    opposite_base_vel = RewTerm(
        func=mdp.opposite_base_velocity,
        weight=-4.0,
        params={"command_name": "base_velocity"},
    )
    opposite_wheel_vel = RewTerm(
        func=mdp.opposite_wheel_velocity,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True),
        },
    )

    feet_contact_number = RewTerm(func=mdp.feet_contact_number, weight=5.0, params=_gait_params())
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_error,
        weight=-5.0,
        params={**_gait_params(), "sigma": 0.05},
    )
    swing_foot_lift = RewTerm(func=mdp.swing_foot_lift, weight=10.0, params=_gait_params())
    triggered_leg_up_vel = RewTerm(func=mdp.triggered_leg_up_velocity, weight=10.0, params=_gait_params())
    wrong_leg_lift = RewTerm(
        func=mdp.wrong_leg_lift,
        weight=-10.0,
        params={**_gait_params(), "sigma": 0.05},
    )
    triggered_leg_action_dir = RewTerm(
        func=mdp.triggered_leg_action_direction,
        weight=10.0,
        params=_gait_params(),
    )
    wheel_zero_velocity = RewTerm(
        func=mdp.wheel_zero_velocity_during_swing,
        weight=0.5,
        params={
            **_gait_params(),
            "wheel_joint_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True),
        },
    )
    foot_landing_vel = RewTerm(
        func=mdp.foot_landing_velocity,
        weight=-5.0,
        params={
            "wheel_radius": L5A_WHEEL_RADIUS,
            "wheel_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "landing_height_threshold": 0.08,
            "landing_time_threshold": 0.12,
            "safe_landing_velocity": 0.1,
        },
    )
    air_wheel_vel = RewTerm(
        func=mdp.air_wheel_velocity,
        weight=-2.0,
        params={
            "wheel_joint_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "contact_threshold": 10.0,
            "velocity_scale": 3.0,
        },
    )
    wheel_spin = RewTerm(
        func=mdp.wheel_slip,
        weight=-20.0,
        params={
            "wheel_radius": L5A_WHEEL_RADIUS,
            "wheel_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "wheel_joint_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True),
            "slip_tolerance": 0.1,
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.wheel_contact_impact,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "max_contact_force": 250.0,
            "contact_force_scale": 100.0,
        },
    )
    default_pos = RewTerm(
        func=mdp.joint_deviation_from_default_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )

    # 保留 WF-Flat 的几何奖励实现/权重，并把绝对世界高度替换为局部地形相对高度。
    leg_symmetry = RewTerm(
        func=mdp.leg_y_symmetry_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "std": math.sqrt(0.5),
        },
    )
    same_wheel_x = RewTerm(
        func=mdp.same_wheel_x_position_l1,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True)},
    )
    wheel_distance = RewTerm(
        func=mdp.wheel_distance_alignment_exp,
        weight=0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "min_distance": L5A_MIN_TRACK_WIDTH,
            "max_distance": L5A_MAX_TRACK_WIDTH,
            "desired_distance": L5A_NOMINAL_TRACK_WIDTH,
            "std": math.sqrt(0.01),
            "command_name": "base_velocity",
        },
    )
    base_height = RewTerm(
        func=mdp.terrain_relative_base_height_l1,
        weight=-20.0,
        params={
            "target_height": L5A_NOMINAL_BASE_HEIGHT,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )


@configclass
class UpstairsTerminationsCfg(TerminationsCfg):
    """与 WF 一致，但过低判断使用局部地形高度。"""

    base_height = DoneTerm(
        func=mdp.root_height_below_local_terrain,
        params={"minimum_height": 0.35, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class UpstairsCurriculumCfg:
    """按 45%/30% 地形长度阈值升降难度。"""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_upstairs,
        params={
            "move_up_fraction": 0.45,
            "move_down_fraction": 0.30,
            "terrain_length": UPSTAIRS_TERRAIN_SIZE[0],
        },
    )


@configclass
class L5AWFUpstairsEnvCfg(L5AWFFlatEnvCfg):
    """独立的 L5A WF 上楼梯训练任务。"""

    scene: L5AWFUpstairsSceneCfg = L5AWFUpstairsSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UpstairsObservationsCfg = UpstairsObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: UpstairsCommandsCfg = UpstairsCommandsCfg()
    events: UpstairsEventCfg = UpstairsEventCfg()
    rewards: UpstairsRewardsCfg = UpstairsRewardsCfg()
    terminations: UpstairsTerminationsCfg = UpstairsTerminationsCfg()
    curriculum: UpstairsCurriculumCfg = UpstairsCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (5.0, -6.0, 3.0)
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

    def build_deployment_export_metadata(self) -> dict[str, Any]:
        return build_l5a_wf_upstairs_export_metadata()


@configclass
class L5AWFUpstairsEnvCfg_PLAY(L5AWFUpstairsEnvCfg):
    """关闭噪声、延迟和动力学随机化的确定性上楼梯评估配置。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.robot = L5A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.observations.policy.enable_corruption = False
        self.observations.obs_history.enable_corruption = False
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
        self.commands.base_velocity.nonflat_lin_vel_x = (0.3, 0.3)
        self.commands.base_velocity.flat_standing_probability = 0.0
        self.commands.base_velocity.fixed_goal_lateral_offset = 0.0

        self.events.imu_mounting_bias = None
        self.events.add_base_mass = None
        self.events.scale_link_mass = None
        self.events.scale_mass_inertia = None
        self.events.physics_material = None
        self.events.actuator_gains = None
        self.events.motor_effort_limits = None
        self.events.base_com = None
        self.events.link_com = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_leg_joints.params["position_range"] = (0.0, 0.0)
        self.actions.left_leg_pos.default_offset_range = (0.0, 0.0)
        self.actions.right_leg_pos.default_offset_range = (0.0, 0.0)
        self.curriculum.terrain_levels = None


def build_l5a_wf_upstairs_deployment_metadata() -> dict[str, Any]:
    """构造与 WF-Flat 动作/Actor 兼容、但 checkpoint 家族独立的训练元数据。"""
    env_cfg = L5AWFUpstairsEnvCfg()
    scan_cfg = env_cfg.scene.height_scanner.pattern_cfg
    scan_dim = (round(scan_cfg.size[0] / scan_cfg.resolution) + 1) * (
        round(scan_cfg.size[1] / scan_cfg.resolution) + 1
    )
    critic_layout = [
        {"name": "wf_privileged_state", "size": 68},
        {"name": "terrain_height_scan", "size": scan_dim},
        {"name": "target", "size": 4},
        {"name": "wheel_contact_history", "size": 6},
        {"name": "step_height", "size": 1},
    ]
    critic_dim = sum(item["size"] for item in critic_layout)
    if scan_dim != 77 or critic_dim != 156:
        raise ValueError(f"Unexpected upstairs critic contract: scan={scan_dim}, critic={critic_dim}.")

    metadata = build_l5a_wf_deployment_metadata()
    metadata.update(
        {
            "source_env_cfg": "L5AWFUpstairsEnvCfg",
            "task_family": "l5a_wf_upstairs",
            "actor_proprioception_dim": 28,
            "critic_privileged_dim": critic_dim,
            "critic_privileged_layout": critic_layout,
            "critic_runner_command_dim": 3,
            "terrain_scan_actor_visible": False,
        }
    )
    return metadata


def build_l5a_wf_upstairs_export_metadata() -> dict[str, Any]:
    """补充上楼梯任务的部署和训练地形契约。"""
    metadata = build_l5a_wf_export_metadata()
    metadata.update(build_l5a_wf_upstairs_deployment_metadata())
    metadata["robot_model"]["name"] = "Huilun-L5A-WF-Upstairs"
    metadata["command_limits"] = {
        "linear_velocity_x": [-0.6, 1.0],
        "nonflat_linear_velocity_x": [0.1, 1.0],
        "linear_velocity_y": [0.0, 0.0],
        "angular_velocity_z": [-1.0, 1.0],
    }
    metadata["training_terrain"] = {
        "grid": [10, 10],
        "tile_size_m": [8.0, 8.0],
        "column_counts": {"flat": 1, "platform_blocks": 2, "upstairs": 7},
        "step_height_m": [0.02, 0.11],
        "step_width_requested_m": [0.66, 0.30],
        # 旧任务 0.1 m 水平高度场会把期望踏面向下量化为实际 0.60--0.30 m。
        "step_width_quantized_m": [0.60, 0.30],
        "max_initial_level": 5,
    }
    return metadata
