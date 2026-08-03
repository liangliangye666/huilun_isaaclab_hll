"""pytest 共享夹具。

这里构造一份最小但完整的 deployment 合同，供 Manifest、观测和控制测试复用。
测试中的数值应当和真实导出的 policy_manifest.json 保持同一语义。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def deployment_contract() -> dict:
    """返回 L5A WF-Flat 部署契约的核心字段。

    该契约是测试中"真理来源"，其数值必须与真实导出的 policy_manifest.json 一致。
    测试复用此 fixture 而不是每次从文件读取，避免磁盘 I/O 和 SHA-256 依赖。
    """
    # policy_order：策略视角，左右腿交错排列
    policy_order = [
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "left_wheel_joint",
        "right_wheel_joint",
    ]
    # hardware_order：MuJoCo 执行器视角，左侧连续、右侧连续
    hardware_order = [
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_wheel_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_wheel_joint",
    ]
    return {
        "schema_version": 1,
        "physics_period_s": 0.005,
        "decimation": 2,
        "control_period_s": 0.01,
        "history_samples": 10,
        "shared_action_delay_physics_steps": [0, 6],
        "proprioception_dim": 28,
        "command_dim": 3,
        "action_dim": 8,
        "proprioception_layout": [
            {"name": "base_angular_velocity", "size": 3, "scale": 0.25, "frame": "robot_base"},
            {"name": "projected_gravity", "size": 3, "scale": 1.0, "frame": "robot_base"},
            {
                "name": "leg_joint_position_relative",
                "size": 6,
                "scale": 1.0,
                "order": policy_order[:6],
            },
            {"name": "joint_velocity_relative", "size": 8, "scale": 0.05, "order": policy_order},
            {"name": "previous_action", "size": 8, "scale": 1.0, "order": policy_order},
        ],
        "command_order": ["linear_velocity_x", "linear_velocity_y", "angular_velocity_z"],
        "policy_action_order": policy_order,
        "policy_action_semantics": {
            "leg_position": {"joints": policy_order[:6], "scale": 0.25, "uses_default_offset": True},
            "wheel_velocity": {"joints": policy_order[6:], "scale": 1.0, "uses_default_offset": True},
        },
        "hardware_dof_order": hardware_order,
        "policy_actions_to_hardware_indices": [0, 1, 2, 6, 3, 4, 5, 7],
        "hardware_state_to_policy_indices": [0, 1, 2, 4, 5, 6, 3, 7],
        "default_joint_positions": {
            "order": policy_order,
            "values": [0.0523599, 0.261799, -0.560251, -0.0523599, 0.261799, -0.560251, 0.0, 0.0],
        },
        "joint_control": {
            "order": policy_order,
            "modes": ["position"] * 6 + ["velocity"] * 2,
            "stiffness": [40.0, 40.0, 80.0, 40.0, 40.0, 80.0, 0.0, 0.0],
            "damping": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5],
            "effort_limits": [90.0, 90.0, 130.0, 90.0, 90.0, 130.0, 90.0, 90.0],
            "velocity_limits": [16.433, 16.433, 14.653, 16.433, 16.433, 14.653, 16.433, 16.433],
        },
        "command_limits": {
            "linear_velocity_x": [-1.0, 1.0],
            "linear_velocity_y": [0.0, 0.0],
            "angular_velocity_z": [-1.0, 1.0],
        },
        "policy_output_clip": 100.0,
        "robot_model": {
            "name": "Huilun-L5A-WF",
            "mjcf_path": "resources/robots/l5a/xml/l5aurdf20260521.xml",
            "mjcf_sha256": "set-by-test",
            "keyframe": "home",
        },
    }
