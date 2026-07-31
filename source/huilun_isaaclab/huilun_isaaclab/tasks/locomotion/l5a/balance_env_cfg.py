# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""从 IsaacGym 迁移而来的 L5A 基础平衡任务配置。

该任务保留为结构较简单的基线：Actor 和 Critic 共用一组展平后的 10 帧本体
观测，不使用 WF 任务中的独立线速度 Encoder 或特权 Critic。配置类只描述
Scene 和各 Manager 的 term；``ManagerBasedRLEnv`` 在 Gym 创建环境时才会解析
配置、实例化 Manager，并在每个控制步调度动作、仿真、奖励、终止和观测。

若目标是完整轮足移动训练，应使用 ``Huilun-L5A-WF-Flat-v0``；本文件主要用于
复现旧 balance 任务、做控制链路检查和作为最小回归基线。
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from huilun_isaaclab.assets.robots.l5a import (
    ACTUATED_JOINT_NAMES,
    L5A_CFG,
    LEG_JOINT_NAMES,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
)

from . import mdp


@configclass
class L5ABalanceSceneCfg(InteractiveSceneCfg):
    """每个并行环境中的平地、L5A、接触传感器和共享照明声明。

    ``InteractiveScene`` 根据环境配置中的 ``num_envs`` 克隆机器人与传感器；
    ContactSensor 覆盖全部刚体，供基座触地终止使用。
    """

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
    )

    robot: ArticulationCfg = L5A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3)

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # 本任务是纯平地本体感知基线，因此没有创建 RayCasterCfg，也没有高度扫描
    # observation term。若以后启用复杂地形，需要同时添加 RayCaster 传感器、
    # 更新周期和对应观测，单独取消这一行注释并不能产生高度信息。


@configclass
class CommandsCfg:
    """平衡基线使用的三维速度指令 ``[v_x, v_y, omega_z]``。

    双轮 L5A 不直接跟踪横向速度，所以 ``v_y`` 固定为零。这里使用直接偏航
    角速度指令而不是“目标 heading -> 内部比例控制 -> omega_z”的模式，这与
    WF 配置的航向命令机制不同。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        # 不划分专门的零指令环境；因此该基线不会系统性训练静止站立子任务。
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        # False 表示直接采样/跟踪 ang_vel_z，而不是对目标航向做内部 P 控制。
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    """沿用旧 IsaacGym 任务的 8 维混合动作接口。

    ActionManager 按声明顺序先取 6 维腿关节位置动作、再取 2 维轮关节速度动作。
    腿目标为默认角度加 ``0.25 * action``，轮目标为默认速度加
    ``0.5 * action``；``preserve_order=True`` 保证策略维度按项目常量中的
    关节顺序映射，而不是按资产内部查找顺序重排。
    """

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,  # 策略动作维度与 LEG_JOINT_NAMES 的声明顺序一致。
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """平衡基线的单组历史观测。

    每帧特征按 term 声明顺序组成 32 维：
    ``角速度 3 + 投影重力 3 + 速度/高度指令 4 + 腿位置 6 + 全关节速度 8
    + 上一动作 8``。ObservationManager 保存最近 10 帧并在环境侧展平，所以
    ``policy`` 的最终 shape 是 ``[N, 320]``。

    该组同时被标准 PPO 配置映射给 Actor 和 Critic，不是 WF 的非对称观测。
    在 ``dt=0.005``、``decimation=2`` 下策略频率为 100 Hz，10 个连续采样按
    训练约定构成约 0.10 s 的窗口。term 的声明顺序也是网络输入/部署顺序，
    修改时必须同步 checkpoint 或重新训练。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """带训练噪声的本体感知、指令和上一动作历史。"""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_height_commands = ObsTerm(
            func=mdp.velocity_height_commands,
            params={"command_name": "base_velocity", "target_height": 0.645},
            scale=(2.0, 2.0, 0.25, 5.0),
        )
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            # 先连接每帧 32 维特征，再将 10 帧历史展平为 320 维网络输入。
            self.history_length = 10
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """平衡任务的 EventManager 生命周期配置。

    ``startup`` term 在仿真启动时执行一次，并在后续 episode 中保持该环境的
    材质参数；``reset`` term 在对应环境超时/失败后重置基座与关节；``interval``
    term 使用各环境独立计时器，每 3--8 秒通过根速度突变模拟一次外部推搡。
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True),
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.05, 0.05),
        },
    )
    push_robot = EventTerm(
        # 瞬时修改根速度用于恢复训练，不是持续施加的物理外力。
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-1.0, 1.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """从旧 L5A balance 任务迁移的奖励组合。

    RewardManager 每个 100 Hz 控制步计算所有 term；函数输出乘 ``weight`` 后
    求和（框架按控制步长积分）。正权重用于跟踪/存活目标，负权重用于惩罚失败、
    几何误差、能耗、越限和动作不平滑。
    """

    # 生存与失败：为持续保持可控姿态提供最直接的 episode 信号。
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # 任务跟踪：前向速度、偏航速度以及与目标基座高度一致的轮高。
    tracking_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    nominal_wheel_height = RewTerm(
        func=mdp.nominal_wheel_height_exp,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "target_base_height": 0.645,
            "wheel_radius": 0.127,
            "std": math.sqrt(0.005),
            "speed_attenuation_std": math.sqrt(0.5),
            "command_name": "base_velocity",
        },
    )
    leg_symmetry = RewTerm(
        func=mdp.leg_y_symmetry_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "std": math.sqrt(0.001),
        },
    )

    # 轮足构型：让两轮同轴、同高，并把轮距保持在 L5A 可行区间。
    same_wheel_x_position = RewTerm(
        func=mdp.same_wheel_x_position_l1,
        weight=-50.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True)},
    )
    same_wheel_z_position = RewTerm(
        func=mdp.same_wheel_z_position_l2,
        weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True)},
    )
    wheel_distance = RewTerm(
        func=mdp.wheel_distance_range_l1,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "min_distance": 0.27,
            "max_distance": 0.30,
        },
    )
    # 稳定性与可执行性：抑制非任务方向运动、姿态误差、冲击、能耗和越限。
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.3)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.00016)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    action_smooth = RewTerm(func=mdp.action_smooth_l2, weight=-0.03)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-12.0)
    base_height = RewTerm(
        func=mdp.base_height_l1,
        weight=-20.0,
        params={"target_height": 0.645},
    )


@configclass
class TerminationsCfg:
    """平衡任务的截断与失败边界。

    ``time_out`` 是达到时长后的正常截断；基座触地、倾角过大和高度过低是真正
    失败。RSL-RL 据此决定回报是否可从超时状态 bootstrap。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    base_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.35})


@configclass
class L5ABalanceEnvCfg(ManagerBasedRLEnvCfg):
    """汇总 Scene 与各 Manager 配置的 L5A 平衡训练环境。

    ``gym.make()`` 读取该类后，``ManagerBasedRLEnv`` 才会构造 Scene、解析
    ``SceneEntityCfg`` 中的关节/刚体索引并实例化 Manager。每个策略步由
    ActionManager 处理 8 维动作，推进两个物理子步后，再生成 done、reward 和
    下一帧 ``[N, 320]`` 观测。
    """

    scene: L5ABalanceSceneCfg = L5ABalanceSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # ① 设置策略控制频率：physics dt=0.005, decimation=2 → 100 Hz 策略步
        #    L5A 固定使用 200 Hz 物理仿真和 100 Hz 策略控制：
        #    control_dt = sim.dt * decimation = 0.005 * 2 = 0.01 s。
        #    decimation 不能孤立修改，否则会改变动作保持时间、10 帧历史窗口、
        #    reward 积分和 interval 事件时间语义，并破坏与实机 100 Hz 接口的对应。
        self.decimation = 2
        self.episode_length_s = 20.0

        # ② 设置可视化相机位置
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.55)

        # ③ PhysX 求解器配置
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_type = 1                # TGS 求解器
        self.sim.physx.min_position_iteration_count = 4
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.gpu_max_rigid_contact_count = 2**23

        # ④ 接触传感器更新频率 = 物理步长（每物理步更新一次）
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class L5ABalanceEnvCfg_PLAY(L5ABalanceEnvCfg):
    """用于播放/调试的较小并行版本，但并非完全确定性评估。

    当前覆盖项仅包括环境数量、policy 观测噪声和 startup 材质随机化；命令采样、
    reset 姿态/关节随机化以及 interval 推搡仍继承自训练配置。需要严格可重复
    回放时，应先理解这些继承项，而不能仅凭 ``PLAY`` 名称判断其已全部关闭。
    """

    def __post_init__(self) -> None:
        # ① 继承训练配置的物理参数和时间离散
        super().__post_init__()
        # ② 缩小环境数量（训练 4096 → 播放 10）
        self.scene.num_envs = 10
        # ③ 关闭策略观测噪声（观测布局和维度不变，checkpoint 可直接加载）
        self.observations.policy.enable_corruption = False
        # ④ 开启指令可视化
        self.commands.base_velocity.debug_vis = True
        # ⑤ 关闭材质随机化
        self.events.physics_material = None
