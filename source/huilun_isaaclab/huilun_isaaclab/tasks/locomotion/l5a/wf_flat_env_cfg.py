# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""适配 Huilun L5A 的 WF（Wheel-Foot）平地盲走任务。

这个文件只负责声明 Manager-Based 环境的“数据契约”：场景中有什么资产，各
Manager 有哪些 term，以及 term 使用什么参数。真正的运行对象由
``ManagerBasedRLEnv`` 在 ``gym.make()`` 时创建：配置先被解析为 Scene，随后
构造 Command/Action/Observation/Event/Reward/Termination 等 Manager。训练循环
每个策略步提交一次动作，环境在内部推进若干物理步，再由各 Manager 产出奖励、
终止信号和下一帧观测。

任务借鉴 TRON2 WF 的训练结构，但机器人模型、关节限制、PD 参数、动作缩放、
轮半径、轮距以及 100 Hz 的 L5A 控制周期仍以本项目为准。尤其不要把本文件当作
TRON2 参数的逐项复制：所有张量维度和随机化范围都对应 L5A 当前资产。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from huilun_isaaclab.assets.robots.l5a import (
    ACTUATED_JOINT_NAMES,
    BASE_BODY_NAME,
    HARDWARE_DOF_NAMES,
    HARDWARE_TO_POLICY_STATE_INDICES,
    L5A_CFG,
    L5A_MAX_TRACK_WIDTH,
    L5A_MIN_TRACK_WIDTH,
    L5A_NOMINAL_BASE_HEIGHT,
    L5A_NOMINAL_TRACK_WIDTH,
    L5A_WF_CFG,
    LEG_BODY_NAMES,
    LEG_JOINT_NAMES,
    POLICY_TO_HARDWARE_ACTION_INDICES,
    PROJECT_ROOT,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
)

from . import mdp


@configclass
class L5AWFSceneCfg(InteractiveSceneCfg):
    """平地训练场景。

    ``InteractiveScene`` 会按 ``num_envs`` 克隆这份声明。每个环境包含一台
    L5A、覆盖机器人全部刚体的接触传感器和共享地面；灯光只影响显示，不进入
    策略观测。接触传感器既服务于非法接触终止，也服务于接触类奖励和 Critic
    的特权观测。
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            # multiply 模式下地面设为 1.0，才能保留机器人侧随机 restitution；
            # 若这里为 0，任意机器人恢复系数与其相乘后都会失效。
            restitution=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = L5A_WF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",    # 把这个传感器挂在整个机器人身上的每一个碰撞体上
        history_length=4,                       # 在 buffer 中保留最近 4 帧的接触力数据（4 个物理步 × 5ms = 20ms 窗口）
        track_air_time=True,                    # 记录每个刚体「离开地面」的时间。用于奖励函数中惩罚腾空时间过长。
        update_period=0.0,                      # 配置文件中先设为 0（占位），实际值在 __post_init__ 中被覆盖为 self.sim.dt，每个物理步（5ms）更新一次，保证不丢任何接触事件
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


@configclass
class CommandsCfg:
    """定义策略要跟踪的机体速度指令。

    ``base_velocity`` 最终向观测提供 ``[v_x, v_y, omega_z]`` 三维指令，每
    10 秒重采样一次。10% 的环境生成站立指令；其余环境采样前进/后退速度和
    目标航向，并由 command term 将航向误差转换为偏航角速度目标。

    L5A 是非完整约束的双轮平台，不能像麦克纳姆轮一样直接横移，因此
    ``v_y`` 固定为零，避免策略通过不真实的轮胎侧滑来追踪横向速度。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """L5A 的 8 维混合控制接口。

    ActionManager 按 term 的声明顺序拼接动作，所以策略动作始终是
    ``[6 个 LEG_JOINT_NAMES 位置动作, 2 个 WHEEL_JOINT_NAMES 速度动作]``。
    ``preserve_order=True`` 又保证每一段内部严格沿常量中的关节顺序映射，部署
    时不能按 USD/PhysX 返回的关节顺序重新排列。

    腿动作被解释为“每环境随机化后的默认关节角 + 0.25 * action”，轮动作被
    解释为“默认轮速 + 1.0 * action”。前者是位置目标，后者是速度目标，二者
    虽在同一策略向量中，进入执行器后的物理含义不同。
    """

    leg_pos = mdp.RandomizedDefaultJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        default_offset_range=(-0.05, 0.05),
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=1,      # TODO 原先设计是0.5,但是L5A 轮半径是 0.127 m，如果命令 lin_vel_x=1.0 m/s，理想轮速量级约为：1.0 / 0.127 ≈ 7.87 rad/s，tron设置为 1
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """WF 策略、速度估计器和 Critic 之间的观测契约。

    ObservationManager 按内部类中 term 的声明顺序连接特征。当前 L5A 资产下，
    各组输出（``N`` 为并行环境数）为：

    * ``policy``：``[N, 28]``，由 3 维角速度、3 维投影重力、6 维腿关节位置、
      8 维关节速度和 8 维上一动作组成；
    * ``obs_history``：``[N, 10, 28]``，与 ``policy`` 同源但保留时间维；
    * ``commands``：``[N, 3]``，顺序是 ``[v_x, v_y, omega_z]``；
    * ``base_lin_vel_target``：``[N, 3]``，只作为 Encoder 的监督真值；
    * ``critic``：``[N, 68]``，包含无噪声本体状态及仿真中才能可靠获得的
      力矩、加速度、轮体速度、刚体质量和接触力。

    RSL-RL 配置会把 10 帧历史展平为 280 维送入 Encoder，得到 3 维基座线速度
    估计；Actor 输入是 ``3 + 28 + 3 = 34`` 维，Critic 输入则是
    ``68 + 3 = 71`` 维。真值线速度和其余特权量不会泄漏给 Actor。

    物理步长是 0.005 s，``decimation=2`` 后策略频率为 100 Hz。因此 10 帧表示
    10 个连续的 100 Hz 采样，训练契约中记作约 0.10 s 的历史窗口，而不是
    TRON2 原控制频率下的 0.20 s。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 当前帧本体感知；训练时注入传感器噪声。"""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_with_imu_bias,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_with_imu_bias,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            scale=1.0,
        )
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            # corruption 由 ObservationManager 统一开关；Play 配置会关闭它。
            self.enable_corruption = True
            # 按上方 term 声明顺序连接为单个 28 维向量。
            self.concatenate_terms = True

    @configclass
    class HistoryCfg(ObsGroup):
        """供 Encoder 使用的时序本体观测，单帧布局与 ``policy`` 完全一致。"""

        '''
        额外模拟了 IMU 安装偏差，所以 policy/history 的角速度和重力方向不是直接用标准 base_ang_vel/projected_gravity
        标准版 base_ang_vel 直接返回 asset.data.root_ang_vel_b（基座系角速度），假设 IMU 完美对齐机器人基座坐标系。
        但真机上 IMU 是物理焊接/螺丝固定在基座上的，总有 ±1~2° 的安装角度偏差。
        '''
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_with_imu_bias,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_with_imu_bias,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            scale=1.0,
        )
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            # 保留 10 帧且不在环境侧展平，便于网络显式检查时间维和单帧维度。
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CommandsObsCfg(ObsGroup):
        """独立保存命令，便于 Actor/Critic 共用且不混入 Encoder 历史。"""

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EstimatorTargetCfg(ObsGroup):
        """仿真监督信号；仅用于训练 Encoder，部署时不需要外部线速度传感器。"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """非对称 Actor-Critic 中仅训练 Critic 可见的无噪声/特权状态。"""

        # robot base measurements
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=1.0,
        )

        # robot joint measurements
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # Privileged observation
        '''
        与标准版的joint_effort一致，只是改了各名字
        '''
        joint_torque = ObsTerm(
            func=mdp.privileged_joint_torque,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        '''
        标准版没有读取joint_acc的函数
        '''
        joint_acc = ObsTerm(
            func=mdp.privileged_joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
            scale=0.0025,
        )
        '''
        单独读取轮速
        '''
        wheel_lin_vel = ObsTerm(
            func=mdp.body_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True)},
            scale=1.0,
        )
        '''
        标准版没有
        L5A 有自己的随机化缓存，例如质量随机化后的 current_body_mass
        L5A 的做法：质量随机化在每回合开始时（reset_idx 事件中）通过事件系统一次性写入 GPU 缓存 env._l5a_current_body_mass，之后每帧从 GPU 直接读，零 CPU 回读。
        避免每帧从 PhysX CPU 接口 get_masses() 回读一次。
        '''
        body_mass = ObsTerm(
            func=mdp.current_body_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
            scale=1.0,
        )
        '''
        单独读取轮接触力
        '''
        wheel_contact_force = ObsTerm(
            func=mdp.body_contact_force_w,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=WHEEL_BODY_NAMES,
                    preserve_order=True,
                )
            },
            scale=1.0,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    obs_history: HistoryCfg = HistoryCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    base_lin_vel_target: EstimatorTargetCfg = EstimatorTargetCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """按生命周期组织的 L5A 动力学随机化和扰动。

    EventManager 根据 ``mode`` 决定 term 的执行时机：

    * ``startup``：仿真启动后对所有并行环境执行一次，参数在随后的多个
      episode 中保持不变；这里用于质量、惯量、质心、摩擦、执行器和 IMU 安装
      偏差等“机器个体差异”；
    * ``reset``：某个环境终止或超时后，只重置相应 ``env_ids`` 的基座与关节；
    * ``interval``：按每个环境自己的计时器周期触发，不要求所有环境同步，
      用于训练期间随机推搡。

    这种划分很重要：若把 startup 误认为每个 episode 重采样，会错误估计策略
    在一次 rollout 中面对的动力学是否稳定。
    """

    imu_mounting_bias = EventTerm(      # IMU 安装偏差
        func=mdp.randomize_imu_mounting_bias,
        mode="startup",
        params={"roll_pitch_range_deg": (-1.2, 1.2)},
    )
    add_base_mass = EventTerm(          # 基座质量
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
        },
    )
    scale_link_mass = EventTerm(        # 连杆质量
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEG_BODY_NAMES + WHEEL_BODY_NAMES),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    '''
    核心区别：
        Tron2 的标准函数直接以 USD 默认值为基准做缩放；
        L5A 的自定义函数 scale_current_rigid_body_mass_inertia 先读取前序事件已经随机化后的当前值，再继续乘。
    通俗类比：
        Tron2 像「每道菜用同一个配方重新做」（从默认值重采样）。
        L5A 像「先用事件 1 加料、事件 2 加料，最后事件 3 整体乘以 0.8~1.2」（叠加式随机化）。
        叠加式的优势是质量分布更分散：
            基座可能 +2kg（事件 1），连杆 ×0.8（事件 2），全局 ×1.2（事件 3），最终一个机器人可能基座很重但腿很轻——更接近真机制造公差。
    '''
    scale_mass_inertia = EventTerm(     # 质量+惯量缩放
        func=mdp.scale_current_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "scale_range": (0.8, 1.2),
        },
    )
    physics_material = EventTerm(       # 摩擦/弹性
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
            "make_consistent": True,
        },
    )
    actuator_gains = EventTerm(         # 执行器刚度阻尼
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    '''
    这个函数随机化每个关节的力矩上限（0.8~1.2 倍），模拟不同电机的最大输出能力差异。
    关键难点在于：
        改了力矩限制后必须同时同步三处——PhysX 仿真侧限制、actuator 的 effort_limit_sim、actuator 的 effort_limit。
        漏掉任何一处，applied_torque 等特权观测就会和真实仿真能力脱节。
    Isaac Lab 没有这个标准函数，因为一般项目的执行器模型不会用到 effort_limit_sim 和 effort_limit 的分离设计。
    L5A 使用的 DelayedImplicitActuatorCfg 有独立的力矩裁剪逻辑，所以需要这个额外同步。
    '''
    motor_effort_limits = EventTerm(    # 电机力矩限制，为什么这个也要限制？
        func=mdp.randomize_joint_effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True),
            "scale_range": (0.8, 1.2),
        },
    )
    '''
    # Tron2: 所有刚体统一范围，三轴对称（不区分基座和连杆）
        robot_center_of_mass = EventTerm(
            func=mdp.randomize_rigid_body_coms,  # Isaac Lab 标准版
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "com_distribution_params": ((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)),
            },
        )
    L5A 这样做的原因：
        基座（大质量）的质心偏移对整体动力学影响更大，所以给了更大的随机范围；
        连杆的质心偏移影响小，给了更小的范围。
        而且 y 轴（左右方向）的偏移范围故意比 x/z 小，因为 L5A 是双轮结构，左右不对称对姿态影响最敏感。
        这些都是针对 L5A 机械结构的特点调优的。
    '''
    base_com = EventTerm(               # 基座质心
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "com_ranges": {
                "x": (-0.03, 0.03),
                "y": (-0.02, 0.02),
                "z": (-0.03, 0.03),
            },
        },
    )
    link_com = EventTerm(               # 连杆质心
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEG_BODY_NAMES + WHEEL_BODY_NAMES),
            "com_ranges": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
            },
        },
    )

    reset_base = EventTerm(             # 基座重置
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_leg_joints = EventTerm(       # 关节重置
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
            "position_range": (-0.10, 0.10),
            "velocity_range": (0.0, 0.0),
        },
    )
    '''
    为什么轮子没有偏移还要写重置？
        因为 (0,0) 表示偏移量为零，不表示不执行。
        该事件会把轮子确定性地写回默认位置和默认速度。
        它主要防止轮速度以及可能不断累积的轮角从上一个 episode 遗留到下一个 episode。
    '''
    reset_wheel_joints = EventTerm(     # 轮重置（轮子重置时不偏移）
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(             # 外部推搡
        # 通过瞬时修改根速度模拟外部冲击，强化恢复能力；不是持续外力。
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
    """按“任务目标—轮足构型—稳定性—可执行性”组织的 WF 奖励。

    RewardManager 在每个 100 Hz 策略步计算所有 term，并将 ``func`` 输出乘以
    ``weight`` 后累加（框架还会按控制步长积分）。正权重鼓励行为，负权重表示
    代价；因此只看原始函数名而忽略符号会误判优化方向。
    """

    # 任务跟踪：速度/偏航跟踪、存活，以及零指令时抑制多余动作。
    track_lin_vel_xy = RewTerm(     # 速度跟踪
        func=mdp.track_lin_vel_xy_exp,      # y 方向跟踪的不是「期望侧向移动」，而是「禁止侧滑」。把 0 设为目标，就是告诉策略——你不许往侧面漂。
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    )
    track_ang_vel_z = RewTerm(      # 偏航跟踪
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(                # 存活
        func=mdp.is_alive,
        weight=1.0,
    )
    stand_still = RewTerm(          # 静止惩罚
        func=mdp.stand_still_l1,
        weight=-3.0,
        params={"command_name": "base_velocity"},
    )

    # 轮足几何：保持左右对称、两轮同轴、合理轮距，并让基座投影靠近轮轴中点。
    leg_symmetry = RewTerm(         # 左右对称
        func=mdp.leg_y_symmetry_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "std": math.sqrt(0.5),
        },
    )
    same_wheel_x = RewTerm(         # 轮同轴
        func=mdp.same_wheel_x_position_l1,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True)},
    )
    wheel_distance = RewTerm(       # 轮距
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
    base_at_wheel_midpoint = RewTerm(   # 基座投影
        func=mdp.base_projection_at_wheel_midpoint_exp,
        weight=0.5,
        params={
            "std": 0.05,
            "wheel_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
        },
    )

    # 基座稳定：约束高度、竖直速度、横滚/俯仰角速度和姿态倾斜。
    base_height = RewTerm(          # 基座高度
        func=mdp.base_height_l1,
        weight=-20.0,
        params={"target_height": L5A_NOMINAL_BASE_HEIGHT},
    )
    lin_vel_z = RewTerm(            # 竖直速度
        func=mdp.lin_vel_z_l2,
        weight=-0.3,
    )
    ang_vel_xy = RewTerm(           # 横滚俯仰角速度
        func=mdp.ang_vel_xy_l2,
        weight=-0.3,
    )
    orientation = RewTerm(          # 姿态倾斜
        func=mdp.flat_orientation_l2,
        weight=-12.0,
    )

    # 动作时序正则：抑制相邻动作突变及二阶不平滑，减轻实机抖动。
    action_rate = RewTerm(          # 动作变化率
        func=mdp.action_rate_l2,
        weight=-0.02,
    )
    action_smoothness = RewTerm(    # 动作平滑
        func=mdp.action_smooth_l2,
        weight=-0.01,
    )

    # 接触安全：轮子允许着地，腿部连杆和基座触地则受到惩罚。
    undesired_contacts = RewTerm(   # 异常接触
        func=mdp.undesired_contacts,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=LEG_BODY_NAMES + [BASE_BODY_NAME],
            ),
            "threshold": 10.0,
        },
    )

    # 关节可执行性：限制力矩、加速度、功率、速度和软限位附近的动作。
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,  # 关节力矩
        weight=-4.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,      # 关节加速度
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
    )
    leg_pos_limits = RewTerm(       # 软限位
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    joint_power = RewTerm(          # 关节功率
        func=mdp.joint_power_l1,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
    )
    wheel_velocity = RewTerm(       # 轮速度
        func=mdp.joint_vel_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True)},
    )
    leg_velocity = RewTerm(         # 腿速度
        func=mdp.joint_vel_l2,
        weight=-0.004,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """episode 结束条件。

    ``time_out`` 被标记为截断而非失败，算法可按超时 bootstrap；基座触地、
    严重倾倒、基座过低或异常大动作属于真正失败。奖励 Manager 会先依据这些
    状态区分存活/终止项，Runner 收集到的 done 信号随后用于回报计算。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY_NAME),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(80.0)},
    )
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35},
    )
    action_out_of_limits = DoneTerm(
        func=mdp.action_out_of_limits,
        params={"threshold": 100.0},
    )


@configclass
class CurriculumCfg:
    """当前注册的 WF blind-flat 任务不启用课程学习。

    保留空配置类是为了维持 Manager-Based 结构完整，也为以后加入指令范围或
    地形难度课程预留入口；它目前不会产生 Curriculum term。
    """

    pass


@configclass
class L5AWFFlatEnvCfg(ManagerBasedRLEnvCfg):
    """完整的 L5A WF 平地训练配置。

    这是各子配置汇合的位置。``ManagerBasedRLEnv`` 会用这些字段构造对应
    Manager；配置对象本身不执行奖励函数或随机化函数。一个策略步的大致边界是：
    ActionManager 解析 8 维动作，物理仿真连续推进 ``decimation`` 次，随后先由
    Termination/Reward Manager 计算旧 episode 的 done 和 reward；需重置的环境
    执行 reset term，再更新 command/interval event，最后 ObservationManager
    生成 reset 后的新观测。因而终止步的 reward 属于旧状态，返回 observation
    已是下一 episode 的初始状态。
    """

    scene: L5AWFSceneCfg = L5AWFSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # ① 时间离散配置：physics dt=0.005, decimation=2 → 100 Hz 策略步
        #    L5A 的时间离散是实机接口与训练契约的一部分：
        #    0.005 s 物理步长 = 200 Hz；每 2 个物理步更新一次策略 = 100 Hz。
        #    不应单独修改 decimation。它会同时改变策略控制周期、10 帧 Encoder
        #    的物理时间窗、动作延迟的秒数解释、奖励积分尺度和 interval 事件计时，
        #    使当前网络/部署 manifest 与实机控制周期不再一致。
        self.decimation = 2
        self.episode_length_s = 20.0

        # ② 设置可视化相机位置
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.55)

        # ③ PhysX 求解器配置
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.solver_type = 1                # TGS 求解器
        self.sim.physx.min_position_iteration_count = 4
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # ④ 接触传感器更新频率 = 物理步长
        self.scene.contact_forces.update_period = self.sim.dt

    def build_deployment_export_metadata(self) -> dict[str, Any]:
        """Return this task family's complete MuJoCo deployment metadata."""
        return build_l5a_wf_export_metadata()


@configclass
class L5AWFFlatEnvCfg_PLAY(L5AWFFlatEnvCfg):
    """用于评估、可视化和导出的轻量确定性 WF 变体。

    它继承训练配置以保证观测顺序和动作语义完全一致，只覆盖并行环境数、执行器
    延迟、观测噪声和随机化。训练与 Play 共用同一个 agent 配置，因此 checkpoint
    不需要转换；速度命令仍由 CommandManager 采样（可由统一 seed 复现），但 Play
    结果只代表名义模型表现，不等同于训练时完整随机化分布。
    """

    def __post_init__(self) -> None:
        # ① 继承训练配置的时间离散和物理参数
        super().__post_init__()
        # ② 缩小环境数量（训练 4096 → 播放 32）
        self.scene.num_envs = 32
        # ③ 替换为无延迟的 L5A_CFG（不包含训练用的随机动作延迟）        TODO tron2中并没有这么做，还是复用原始版本，因此这里应该注释
        # self.scene.robot = L5A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ④ 关闭 policy 和 history 的观测噪声（观测布局不变，checkpoint 可直接加载）
        self.observations.policy.enable_corruption = False
        self.observations.obs_history.enable_corruption = False
        # ⑤ 开启指令可视化
        self.commands.base_velocity.debug_vis = True

        # ⑥ 关闭所有 startup 个体差异随机化
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

        # ⑦ 收紧 reset 随机范围：每次评估从名义状态开始
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_leg_joints.params["position_range"] = (0.0, 0.0)
        self.actions.leg_pos.default_offset_range = (0.0, 0.0)

        # 追加：固定命令（直走 0.2 m/s，不旋转）
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.2),     # 固定 0.2 m/s 前向
            lin_vel_y=(0.0, 0.0),     # 无侧向
            ang_vel_z=(0.0, 0.0),     # 不旋转
            heading=(0.0, 0.0),       # 固定航向
        )
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0


# ==================== 以下是 防止加载旧 checkpoint 时，如果 checkpoint 里的 deployment_metadata 和当前配置不同，代码会报警 ===============
def _metadata_scale(term: ObsTerm) -> float:
    """Return the scalar observation scale recorded in the deployment contract."""
    scale = getattr(term, "scale", None)
    return 1.0 if scale is None else float(scale)


def _float_range(values: tuple[float, float] | list[float]) -> list[float]:
    """Convert config ranges to JSON/checkpoint-stable lists."""
    return [float(values[0]), float(values[1])]


def _seconds(value: float) -> float:
    """Round seconds to avoid metadata churn from binary floating point noise."""
    return round(float(value), 10)


def _assert_l5a_wf_deployment_metadata(metadata: dict[str, Any]) -> None:
    """Fail at config creation time if the generated deployment contract is internally inconsistent."""
    layout_dim = sum(int(item["size"]) for item in metadata["proprioception_layout"])
    if layout_dim != metadata["proprioception_dim"]:
        raise ValueError(
            f"L5A WF deployment metadata proprioception layout has {layout_dim} dims, "
            f"expected {metadata['proprioception_dim']}."
        )
    if metadata["history_duration_s"] != _seconds(metadata["control_period_s"] * metadata["history_samples"]):
        raise ValueError("L5A WF deployment metadata history duration does not match control period * samples.")
    if metadata["control_period_s"] != _seconds(metadata["physics_period_s"] * metadata["decimation"]):
        raise ValueError("L5A WF deployment metadata control period does not match physics period * decimation.")
    if len(metadata["command_order"]) != metadata["command_dim"]:
        raise ValueError("L5A WF deployment metadata command_order length does not match command_dim.")
    if len(metadata["policy_action_order"]) != metadata["action_dim"]:
        raise ValueError("L5A WF deployment metadata policy_action_order length does not match action_dim.")
    expected_indices = list(range(metadata["action_dim"]))
    for key in ("policy_actions_to_hardware_indices", "hardware_state_to_policy_indices"):
        if sorted(metadata[key]) != expected_indices:
            raise ValueError(f"L5A WF deployment metadata {key} must be a permutation of {expected_indices}.")


def build_l5a_wf_deployment_metadata() -> dict[str, Any]:
    """Build the WF deployment contract from the current L5A environment config.

    The policy network still infers tensor widths from the first real TensorDict. This metadata is
    only the checkpoint/export contract used by deployment and compatibility checks, so it should
    follow the same config objects instead of duplicating their numeric values by hand.
    """
    env_cfg = L5AWFFlatEnvCfg()
    physics_period_s = _seconds(env_cfg.sim.dt)
    control_period_s = _seconds(env_cfg.sim.dt * env_cfg.decimation)
    history_samples = int(env_cfg.observations.obs_history.history_length)
    delay_cfg = env_cfg.scene.robot.actuators["all_joints"]
    delay_steps = [int(delay_cfg.min_delay), int(delay_cfg.max_delay)]
    leg_action = env_cfg.actions.leg_pos
    wheel_action = env_cfg.actions.wheel_vel
    imu_bias_range = env_cfg.events.imu_mounting_bias.params["roll_pitch_range_deg"]

    metadata = {
        "schema_version": 1,
        "source_env_cfg": "L5AWFFlatEnvCfg",
        "physics_period_s": physics_period_s,
        "decimation": int(env_cfg.decimation),
        "control_period_s": control_period_s,
        "history_samples": history_samples,
        "history_duration_s": _seconds(control_period_s * history_samples),
        "shared_action_delay_physics_steps": delay_steps,
        "shared_action_delay_s": [_seconds(delay_steps[0] * physics_period_s), _seconds(delay_steps[1] * physics_period_s)],
        "training_joint_zero_error_rad": _float_range(leg_action.default_offset_range),
        "training_imu_mounting_bias_deg": _float_range(imu_bias_range),
        "proprioception_dim": 3 + 3 + len(LEG_JOINT_NAMES) + len(ACTUATED_JOINT_NAMES) + len(ACTUATED_JOINT_NAMES),
        "command_dim": 3,
        "action_dim": len(leg_action.joint_names) + len(wheel_action.joint_names),
        "proprioception_layout": [
            {
                "name": "base_angular_velocity",
                "size": 3,
                "scale": _metadata_scale(env_cfg.observations.policy.base_ang_vel),
                "frame": "robot_base",
            },
            {
                "name": "projected_gravity",
                "size": 3,
                "scale": _metadata_scale(env_cfg.observations.policy.projected_gravity),
                "frame": "robot_base",
            },
            {
                "name": "leg_joint_position_relative",
                "size": len(LEG_JOINT_NAMES),
                "scale": _metadata_scale(env_cfg.observations.policy.leg_joint_pos),
                "order": list(leg_action.joint_names),
            },
            {
                "name": "joint_velocity_relative",
                "size": len(ACTUATED_JOINT_NAMES),
                "scale": _metadata_scale(env_cfg.observations.policy.joint_vel),
                "order": list(ACTUATED_JOINT_NAMES),
            },
            {
                "name": "previous_action",
                "size": len(ACTUATED_JOINT_NAMES),
                "scale": _metadata_scale(env_cfg.observations.policy.last_action),
                "order": list(ACTUATED_JOINT_NAMES),
            },
        ],
        "command_order": ["linear_velocity_x", "linear_velocity_y", "angular_velocity_z"],
        "policy_action_order": list(leg_action.joint_names) + list(wheel_action.joint_names),
        "policy_action_semantics": {
            "leg_position": {
                "joints": list(leg_action.joint_names),
                "scale": float(leg_action.scale),
                "uses_default_offset": bool(leg_action.use_default_offset),
            },
            "wheel_velocity": {
                "joints": list(wheel_action.joint_names),
                "scale": float(wheel_action.scale),
                "uses_default_offset": bool(wheel_action.use_default_offset),
            },
        },
        "hardware_dof_order": list(HARDWARE_DOF_NAMES),
        "policy_actions_to_hardware_indices": list(POLICY_TO_HARDWARE_ACTION_INDICES),
        "hardware_state_to_policy_indices": list(HARDWARE_TO_POLICY_STATE_INDICES),
    }
    _assert_l5a_wf_deployment_metadata(metadata)
    return metadata


def _resolve_joint_parameter(parameter: float | dict[str, float], joint_name: str) -> float:
    """Resolve an Isaac Lab scalar-or-regex joint parameter for one named joint."""
    if isinstance(parameter, int | float):
        return float(parameter)
    matches = [float(value) for pattern, value in parameter.items() if re.fullmatch(pattern, joint_name)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one parameter match for joint {joint_name!r}, got {len(matches)}.")
    return matches[0]


def _sha256_path(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_l5a_wf_export_metadata() -> dict[str, Any]:
    """Enrich the checkpoint-compatible contract with runtime deployment parameters.

    This function is intentionally separate from ``build_l5a_wf_deployment_metadata``. The latter
    is serialized into checkpoints and must remain byte-for-byte compatible with existing runs.
    """
    metadata = build_l5a_wf_deployment_metadata()
    env_cfg = L5AWFFlatEnvCfg()
    actuator = env_cfg.scene.robot.actuators["all_joints"]
    joint_order = list(ACTUATED_JOINT_NAMES)
    command_ranges = env_cfg.commands.base_velocity.ranges
    mjcf_path = PROJECT_ROOT / "resources" / "robots" / "l5a" / "xml" / "l5aurdf20260521.xml"

    metadata.update(
        {
            "robot_model": {
                "name": "Huilun-L5A-WF",
                "mjcf_path": str(mjcf_path.relative_to(PROJECT_ROOT)),
                "mjcf_sha256": _sha256_path(mjcf_path),
                "base_body": BASE_BODY_NAME,
                "base_joint": "base_joint",
                "orientation_sensor": "orientation",
                "angular_velocity_sensor": "angular-velocity",
                "keyframe": "home",
            },
            "hardware_actuator_order": list(HARDWARE_DOF_NAMES),
            "default_joint_positions": {
                "order": joint_order,
                "values": [float(env_cfg.scene.robot.init_state.joint_pos[name]) for name in joint_order],
            },
            "joint_control": {
                "order": joint_order,
                "modes": ["position"] * len(LEG_JOINT_NAMES) + ["velocity"] * len(WHEEL_JOINT_NAMES),
                "stiffness": [_resolve_joint_parameter(actuator.stiffness, name) for name in joint_order],
                "damping": [_resolve_joint_parameter(actuator.damping, name) for name in joint_order],
                "effort_limits": [
                    _resolve_joint_parameter(actuator.effort_limit_sim, name) for name in joint_order
                ],
                "velocity_limits": [
                    _resolve_joint_parameter(actuator.velocity_limit_sim, name) for name in joint_order
                ],
            },
            "command_limits": {
                "linear_velocity_x": _float_range(command_ranges.lin_vel_x),
                "linear_velocity_y": _float_range(command_ranges.lin_vel_y),
                "angular_velocity_z": _float_range(command_ranges.ang_vel_z),
            },
        }
    )
    return metadata
