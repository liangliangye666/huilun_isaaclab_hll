# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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

from . import mdp

PROJECT_ROOT = Path(__file__).resolve().parents[6]
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
WHEEL_BODY_NAMES = ["left_wheel_link", "right_wheel_link"]


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
"""L5A wheel-legged robot configuration migrated from the IsaacGym balance task."""


@configclass
class L5ABalanceSceneCfg(InteractiveSceneCfg):
    """Flat-ground L5A balance scene."""

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


@configclass
class CommandsCfg:
    """Command specifications for balance and wheel velocity tracking."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
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
    """Action specifications preserving the IsaacGym 6-position + 2-velocity split."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
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
    """Observation specifications matching the balance proprioception stack."""

    @configclass
    class PolicyCfg(ObsGroup):
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
            self.history_length = 10
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset and domain randomization events."""

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


@configclass
class RewardsCfg:
    """Reward terms migrated from the IsaacGym L5A balance task."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

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
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.3)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.00016)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
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
    """Termination terms for balance."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    base_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.35})


@configclass
class L5ABalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based L5A balance training environment."""

    scene: L5ABalanceSceneCfg = L5ABalanceSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 20.0

        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.55)

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 4
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.gpu_max_rigid_contact_count = 2**23

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class L5ABalanceEnvCfg_PLAY(L5ABalanceEnvCfg):
    """Smaller deterministic variant for play/debug."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
