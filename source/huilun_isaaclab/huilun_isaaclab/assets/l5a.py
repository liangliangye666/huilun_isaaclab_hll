# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A robot asset and kinematic naming used by all L5A tasks."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from huilun_isaaclab.actuators import DelayedImplicitActuatorCfg

PROJECT_ROOT = Path(__file__).resolve().parents[4]
L5A_USD_PATH = PROJECT_ROOT / "resources" / "robots" / "l5a" / "usd" / "l5a20260521.usd"

LEG_JOINT_NAMES = [
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
]
WHEEL_JOINT_NAMES = ["left_wheel_joint", "right_wheel_joint"]
ACTUATED_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
HARDWARE_DOF_NAMES = [
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_wheel_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_wheel_joint",
]
POLICY_TO_HARDWARE_ACTION_INDICES = [ACTUATED_JOINT_NAMES.index(name) for name in HARDWARE_DOF_NAMES]
HARDWARE_TO_POLICY_STATE_INDICES = [HARDWARE_DOF_NAMES.index(name) for name in ACTUATED_JOINT_NAMES]

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

L5A_WHEEL_RADIUS = 0.127
L5A_MIN_TRACK_WIDTH = 0.27
L5A_MAX_TRACK_WIDTH = 0.30
L5A_NOMINAL_TRACK_WIDTH = 0.28
L5A_NOMINAL_BASE_HEIGHT = 0.645


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
        pos=(0.0, 0.0, 0.695),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_hip_roll_joint": 0.0523599,
            "left_hip_pitch_joint": 0.261799,
            "left_knee_joint": -0.560251,
            "left_wheel_joint": 0.0,
            "right_hip_roll_joint": 0.0523599,
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
            stiffness={".*hip_roll.*": 40.0, ".*hip_pitch.*": 40.0, ".*knee.*": 80.0},
            damping={".*hip_roll.*": 2.0, ".*hip_pitch.*": 2.0, ".*knee.*": 2.0},
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            effort_limit_sim=90.0,
            velocity_limit_sim=16.433,
            stiffness=0.0,
            damping=1.5,
        ),
    },
)
"""L5A articulation configuration preserved from the IsaacGym balance task."""


L5A_WF_CFG = L5A_CFG.copy()
L5A_WF_CFG.actuators = {
    "all_joints": DelayedImplicitActuatorCfg(
        joint_names_expr=ACTUATED_JOINT_NAMES,
        effort_limit_sim={".*hip.*": 90.0, ".*knee.*": 130.0, ".*wheel.*": 90.0},
        velocity_limit_sim={".*hip.*": 16.433, ".*knee.*": 14.653, ".*wheel.*": 16.433},
        stiffness={
            ".*hip_roll.*": 40.0,
            ".*hip_pitch.*": 40.0,
            ".*knee.*": 80.0,
            ".*wheel.*": 0.0,
        },
        damping={
            ".*hip_roll.*": 2.0,
            ".*hip_pitch.*": 2.0,
            ".*knee.*": 2.0,
            ".*wheel.*": 1.5,
        },
        min_delay=0,
        max_delay=6,
    ),
}
"""WF variant preserving L5A gains and its shared 0--30 ms command delay."""
