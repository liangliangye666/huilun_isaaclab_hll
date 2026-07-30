# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Blind-flat WF locomotion task adapted to the Huilun L5A robot.

The environment follows the useful TRON2 WF training contract while retaining
the L5A robot model, control rate, joint limits, PD gains, action scales, wheel
radius, and track-width geometry.
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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from huilun_isaaclab.assets.l5a import (
    ACTUATED_JOINT_NAMES,
    BASE_BODY_NAME,
    L5A_CFG,
    L5A_MAX_TRACK_WIDTH,
    L5A_MIN_TRACK_WIDTH,
    L5A_NOMINAL_BASE_HEIGHT,
    L5A_NOMINAL_TRACK_WIDTH,
    L5A_WF_CFG,
    LEG_BODY_NAMES,
    LEG_JOINT_NAMES,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
)

from . import mdp


@configclass
class L5AWFSceneCfg(InteractiveSceneCfg):
    """Flat terrain, L5A articulation, contact sensing, and lighting."""

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
            # With multiply combination, 1.0 keeps the robot-side
            # restitution randomization effective instead of cancelling it.
            restitution=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = L5A_WF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


@configclass
class CommandsCfg:
    """Forward/backward, heading/yaw, and standing commands.

    L5A is a non-holonomic two-wheel platform, so the lateral command remains
    zero instead of asking the policy to learn motion through tire side-slip.
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
    """Preserve the current L5A six-position plus two-wheel-velocity interface."""

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
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Actor, history, command, privileged critic, and estimator-target groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_with_imu_bias,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_with_imu_bias,
            noise=Unoise(n_min=-0.05, n_max=0.05),
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
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HistoryCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_with_imu_bias,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_with_imu_bias,
            noise=Unoise(n_min=-0.05, n_max=0.05),
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
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EstimatorTargetCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)
        joint_torque = ObsTerm(
            func=mdp.privileged_joint_torque,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        joint_acc = ObsTerm(
            func=mdp.privileged_joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
            scale=0.0025,
        )
        wheel_lin_vel = ObsTerm(
            func=mdp.body_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True)},
        )
        body_mass = ObsTerm(
            func=mdp.current_body_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )
        wheel_contact_force = ObsTerm(
            func=mdp.body_contact_force_w,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=WHEEL_BODY_NAMES,
                    preserve_order=True,
                )
            },
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
    """WF-style dynamics randomization adapted to L5A scales."""

    imu_mounting_bias = EventTerm(
        func=mdp.randomize_imu_mounting_bias,
        mode="startup",
        params={"roll_pitch_range_deg": (-1.2, 1.2)},
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEG_BODY_NAMES + WHEEL_BODY_NAMES),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    scale_mass_inertia = EventTerm(
        func=mdp.scale_current_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "scale_range": (0.8, 1.2),
        },
    )
    physics_material = EventTerm(
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
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    motor_effort_limits = EventTerm(
        func=mdp.randomize_joint_effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True),
            "scale_range": (0.8, 1.2),
        },
    )
    base_com = EventTerm(
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
    link_com = EventTerm(
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

    reset_base = EventTerm(
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
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
            "position_range": (-0.10, 0.10),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_wheel_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
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
    """TRON2 WF reward structure with L5A geometry and joint groups."""

    # Task tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    stand_still = RewTerm(
        func=mdp.stand_still_l1,
        weight=-3.0,
        params={"command_name": "base_velocity"},
    )

    # Wheel-foot geometry
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
    base_at_wheel_midpoint = RewTerm(
        func=mdp.base_projection_at_wheel_midpoint_exp,
        weight=0.5,
        params={
            "std": 0.05,
            "wheel_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
        },
    )

    # Base regulation
    base_height = RewTerm(
        func=mdp.base_height_l1,
        weight=-20.0,
        params={"target_height": L5A_NOMINAL_BASE_HEIGHT},
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.3)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-12.0)

    # Action regulation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_smoothness = RewTerm(func=mdp.action_smooth_l2, weight=-0.01)

    # Contact regulation
    undesired_contacts = RewTerm(
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

    # Joint regulation
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-4.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
    )
    leg_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    joint_power = RewTerm(
        func=mdp.joint_power_l1,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)},
    )
    wheel_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True)},
    )
    leg_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.004,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """WF failure conditions plus the L5A minimum-height guard."""

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
    """The registered TRON2 WF blind-flat task does not enable a curriculum."""

    pass


@configclass
class L5AWFFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Full L5A WF blind-flat training configuration."""

    scene: L5AWFSceneCfg = L5AWFSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # L5A hard constraints: 200 Hz physics and 100 Hz policy.
        self.decimation = 2
        self.episode_length_s = 20.0

        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.55)

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 4
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class L5AWFFlatEnvCfg_PLAY(L5AWFFlatEnvCfg):
    """Deterministic, smaller WF configuration for evaluation and export."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.robot = L5A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.observations.policy.enable_corruption = False
        self.observations.obs_history.enable_corruption = False
        self.commands.base_velocity.debug_vis = True

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
        self.actions.leg_pos.default_offset_range = (0.0, 0.0)
