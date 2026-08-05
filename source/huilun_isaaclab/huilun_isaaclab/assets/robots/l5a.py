# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 机器人资产及全任务共享的运动学命名。

本文件是 L5A 的"硬件契约"集中入口：USD 路径、关节/刚体名称、策略与真机
DOF 顺序、名义几何尺寸和执行器参数都在这里维护。环境配置应尽量引用这些
常量，避免训练、回放和部署端各自复制一份顺序或参数后逐渐失配。
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from huilun_isaaclab.actuators import DelayedImplicitActuatorCfg

# ① 确定项目根目录和 USD 文件路径
PROJECT_ROOT = Path(__file__).resolve().parents[5]
L5A_USD_PATH = PROJECT_ROOT / "resources" / "robots" / "l5a" / "usd" / "l5a20260521.usd"

# =============================================================================
# ② 关节命名契约：策略侧 vs 真机侧
# =============================================================================
# 左右分组既用于声明四个 ActionTerm，也用于构造腿/轮语义分组。
LEFT_LEG_JOINT_NAMES = [
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
]
RIGHT_LEG_JOINT_NAMES = [
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
]
LEFT_WHEEL_JOINT_NAMES = ["left_wheel_joint"]
RIGHT_WHEEL_JOINT_NAMES = ["right_wheel_joint"]

LEG_JOINT_NAMES = LEFT_LEG_JOINT_NAMES + RIGHT_LEG_JOINT_NAMES
WHEEL_JOINT_NAMES = LEFT_WHEEL_JOINT_NAMES + RIGHT_WHEEL_JOINT_NAMES

# 真机接口沿用"每条腿的三个关节紧邻该侧车轮"的顺序：
# [左 roll, 左 pitch, 左 knee, 左轮, 右 roll, 右 pitch, 右 knee, 右轮]。
HARDWARE_DOF_NAMES = [
    *LEFT_LEG_JOINT_NAMES,
    *LEFT_WHEEL_JOINT_NAMES,
    *RIGHT_LEG_JOINT_NAMES,
    *RIGHT_WHEEL_JOINT_NAMES,
]
# WF 的全关节观测、Actor 输出和执行器统一采用硬件顺序。
WF_POLICY_DOF_NAMES = list(HARDWARE_DOF_NAMES)

# 两个映射保留在 Manifest 契约中供通用部署程序校验；当前都是恒等映射。
# 当前结果为 [0, 1, 2, 3, 4, 5, 6, 7]。
# 用法：hardware_action = policy_action[..., POLICY_TO_HARDWARE_ACTION_INDICES]。
# 列表中的每个元素都是该硬件 DOF 在策略向量中的列号。
POLICY_TO_HARDWARE_ACTION_INDICES = [WF_POLICY_DOF_NAMES.index(name) for name in HARDWARE_DOF_NAMES]
# 当前结果为 [0, 1, 2, 3, 4, 5, 6, 7]。
# 用法：policy_state = hardware_state[..., HARDWARE_TO_POLICY_STATE_INDICES]。
# 列表中的每个元素都是该策略 DOF 在硬件状态向量中的列号。
HARDWARE_TO_POLICY_STATE_INDICES = [HARDWARE_DOF_NAMES.index(name) for name in WF_POLICY_DOF_NAMES]

# 刚体名称用于接触传感器、几何奖励、质量/质心随机化等 SceneEntityCfg 解析。
LEG_BODY_NAMES = [
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
]
WHEEL_BODY_NAMES = ["left_wheel_link", "right_wheel_link"]
BASE_BODY_NAME = "base_link"

# 这些尺寸既参与奖励目标，也参与安全/几何约束；修改机械模型时应同步复核。
L5A_WHEEL_RADIUS = 0.127
L5A_MIN_TRACK_WIDTH = 0.27
L5A_MAX_TRACK_WIDTH = 0.30
L5A_NOMINAL_TRACK_WIDTH = 0.28
L5A_NOMINAL_BASE_HEIGHT = 0.645
L5A_INITIAL_BASE_HEIGHT = L5A_NOMINAL_BASE_HEIGHT + 0.005


# 基础资产保持既有 L5A 的物理参数。
#
# 控制语义由"环境里的 ActionTerm + 这里的隐式执行器"共同决定：
# - 六个腿关节接收位置目标，由非零 stiffness/damping 形成隐式 PD；
# - 两个轮关节接收速度目标，stiffness=0，主要由 damping 跟踪轮速。
# 因此不能只根据同一个 ImplicitActuatorCfg 类型判断腿轮的控制模式。
L5A_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(L5A_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 名义站立高度为 0.645 m；生成环境时额外抬高 5 mm，避免初始接触穿透。
        pos=(0.0, 0.0, L5A_INITIAL_BASE_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_hip_roll_joint": 0.0523599,
            "left_hip_pitch_joint": 0.261799,
            "left_knee_joint": -0.560251,
            "left_wheel_joint": 0.0,
            "right_hip_roll_joint": -0.0523599,
            "right_hip_pitch_joint": 0.261799,
            "right_knee_joint": -0.560251,
            "right_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=LEG_JOINT_NAMES,
            effort_limit_sim={".*hip.*": 90.0, ".*knee.*": 130.0},
            velocity_limit_sim={".*hip.*": 16.433, ".*knee.*": 14.653},
            stiffness={".*hip_roll.*": 84.0, ".*hip_pitch.*": 84.0, ".*knee.*": 84.0},
            damping={".*hip_roll.*": 2.5, ".*hip_pitch.*": 2.5, ".*knee.*": 2.5},
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            effort_limit_sim=90.0,
            velocity_limit_sim=16.433,
            stiffness=0.0,
            damping=0.8,
        ),
    },
)
"""L5A articulation configuration shared by the WF task."""


# WF 训练使用同一套 L5A 机械参数，只把执行器替换为带命令延迟的版本。
# 腿和轮放入同一个 actuator group，使每个环境采样到的延迟同时作用于全部
# 8 个 DOF；这对应 L5A 原系统的共享通信/控制延迟，而不是为左右腿轮分别采样。
L5A_WF_CFG = L5A_CFG.copy()
L5A_WF_CFG.actuators = {
    "all_joints": DelayedImplicitActuatorCfg(
        joint_names_expr=WF_POLICY_DOF_NAMES,
        effort_limit_sim={".*hip.*": 90.0, ".*knee.*": 130.0, ".*wheel.*": 90.0},
        velocity_limit_sim={".*hip.*": 16.433, ".*knee.*": 14.653, ".*wheel.*": 16.433},
        stiffness={
            ".*hip_roll.*": 84.0,
            ".*hip_pitch.*": 84.0,
            ".*knee.*": 84.0,
            ".*wheel.*": 0.0,
        },
        damping={
            ".*hip_roll.*": 2.5,
            ".*hip_pitch.*": 2.5,
            ".*knee.*": 2.5,
            ".*wheel.*": 0.8,
        },
        # 延迟单位是物理仿真步，不是策略步；WF 任务 dt=0.005 s，
        # 因此 0--6 步对应共享的 0--30 ms 命令延迟。
        min_delay=0,
        max_delay=6,
    ),
}
"""WF variant preserving L5A gains and its shared 0--30 ms command delay."""


'''
TODO
问题：enabled_self_collisions：L5A为false，但逐际动力是true

不是简单的设计问题，更准确地说：L5A 当前关掉自碰撞是偏“保守稳定”的资产处理；Tron2 WF 打开自碰撞，是因为它的碰撞资产明显更适合这么做。

关键差异在这里：

- 你的 L5A 配置是 [l5a.py](/mnt/isaacdata/myproject/huilun_isaaclab/source/huilun_isaaclab/huilun_isaaclab/assets/l5a.py:102)：`enabled_self_collisions=False`
- L5A 的 USD 转换配置里也写了 `self_collision: false`，并且 `collider_type: convex_hull`，见 [config.yaml](/mnt/isaacdata/myproject/huilun_isaaclab/resources/robots/l5a/usd/config.yaml:17)
- L5A 原始 URDF 里每个 link 的 collision 基本直接用了对应 STL mesh，例如 base、hip link 都是 mesh collision，见 [l5aurdf20260521.urdf](/mnt/isaacdata/myproject/huilun_isaaclab/resources/robots/l5a/urdf/l5aurdf20260521.urdf:52)
- Tron2 WF 配置打开了自碰撞，见 `wheelfoot_tron2a_cfg.py:25`
- 但 Tron2 的 Xacro 默认 `use_primitive_collision=true`，collision 不是直接用复杂 STL，而是大量用 `collision_box` 这类简化体，见 `robot.urdf.xacro:5` 和 `robot.urdf.xacro:66`

所以判断是：
    L5A 当前关掉自碰撞，大概率不是“机器人设计有问题”，而是“碰撞模型还没有为自碰撞训练清理过”。
    SolidWorks 导出的 STL collision 很容易在关节附近、轮腿连接处、左右腿极限姿态下产生轻微重叠。
    一旦打开自碰撞，PhysX 会把这些内部穿透当成真实接触，结果可能是：
        - reset 初始姿态就有内部接触力；
        - 腿部动作一大就出现非真实的弹开/卡住；
        - reward、接触传感器、跌倒判断受到污染；
        - PPO 训练不稳定，策略学到躲避假碰撞，而不是真实运动能力；
        - 仿真速度下降。

Tron2 WF 打开自碰撞的合理性在于：
    它的资产更像是“为 RL 训练处理过的碰撞模型”。
    简化 box/capsule/primitive 碰撞体通常可以控制间隙，避免 CAD 细节导致的假接触。
    打开自碰撞可以防止双腿互穿、膝盖撞机身、轮子穿腿，这对全向轮足训练是有价值的。

我的建议是：
    不要现在直接把 L5A 改成 `True` 当作对齐 Tron2。
    更稳的路线是先保留 `False` 跑通 WF 训练；
    下一步单独做一个 `L5A_RL_COLLISION` 版本资产，把 collision 从 STL/convex hull 改成简化 primitive，并验证初始姿态和随机姿态没有内部接触，再打开自碰撞。
    最终 WF 全向训练确实应该倾向于打开自碰撞，但前提是 L5A 的碰撞体要先清理好。

一句话结论：
    Tron2 打开自碰撞不是因为算法特殊，而是资产碰撞建模更干净；
    L5A 当前关闭是合理的迁移保守选择，不代表机械设计错，但如果长期做高质量 WF 训练，应该补一版简化碰撞资产后再开启。
'''
