"""部署 Manifest 加载和契约校验。

policy_manifest.json 是 Python sim2sim、后续 C++ sim2sim 和 sim2real 共享的
唯一契约来源。本模块在启动前校验模型签名、文件 hash、MJCF hash、关节顺序、
观测布局、动作语义和 PD 参数，避免运行到一半才暴露模型/机器人不匹配问题。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    """Raised when a deployment bundle does not match the L5A WF-Flat contract."""


# policy_order：策略视角的关节顺序（左右腿交错的逻辑分组）
EXPECTED_POLICY_ORDER = [
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "left_wheel_joint",
    "right_wheel_joint",
]
# hardware_order：MuJoCo 执行器/硬件视角的关节顺序（左侧连续，右侧连续）
EXPECTED_HARDWARE_ORDER = [
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_wheel_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_wheel_joint",
]
# 默认关节位置：髋 roll ±3°、髋 pitch 15°、膝 -32.1°、轮子 0
EXPECTED_DEFAULT_Q = [0.0523599, 0.261799, -0.560251, -0.0523599, 0.261799, -0.560251, 0.0, 0.0]
# PD 控制参数：轮子 kp=0 表示纯速度控制
EXPECTED_STIFFNESS = [40.0, 40.0, 80.0, 40.0, 40.0, 80.0, 0.0, 0.0]
EXPECTED_DAMPING = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5]
EXPECTED_EFFORT_LIMITS = [90.0, 90.0, 130.0, 90.0, 90.0, 130.0, 90.0, 90.0]
EXPECTED_VELOCITY_LIMITS = [16.433, 16.433, 14.653, 16.433, 16.433, 14.653, 16.433, 16.433]


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256，用于确认导出模型和 MJCF 没被替换或损坏。"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(name: str, shape: list[str | int]) -> dict[str, Any]:
    """构造模型签名条目，用于校验 ONNX 输入输出名称和形状。"""
    return {"name": name, "dtype": "float32", "shape": shape}


@dataclass(frozen=True)
class DeploymentBundle:
    """通过契约校验后的部署包。

    不可变数据类，包含模型目录、MJCF 路径和已解析的 manifest JSON。
    """

    model_dir: Path
    mjcf_path: Path
    manifest: dict[str, Any]

    @property
    def deployment(self) -> dict[str, Any]:
        """manifest 中的 deployment 字段，包含所有部署参数。"""
        return self.manifest["deployment"]

    def model_path(self, model_name: str) -> Path:
        """返回指定模型的 ONNX 文件绝对路径。

        Args:
            model_name: "velocity_estimator" 或 "policy"。
        """
        return self.model_dir / self.manifest["models"][model_name]["files"]["onnx"]


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    """断言 actual == expected，不匹配时抛出 ContractError 并带上标签。"""
    if actual != expected:
        raise ContractError(f"{label} mismatch: expected {expected!r}, got {actual!r}.")


def _validate_model_signatures(manifest: dict[str, Any]) -> None:
    """校验双模型输入输出签名，拒绝旧版 Encoder+Actor 组合 policy。"""
    expected = {
        "velocity_estimator": {
            "inputs": [_signature("observation_history", ["batch", 10, 28])],
            "outputs": [_signature("estimated_base_linear_velocity", ["batch", 3])],
        },
        "policy": {
            "inputs": [
                _signature("estimated_base_linear_velocity", ["batch", 3]),
                _signature("proprioception", ["batch", 28]),
                _signature("commands", ["batch", 3]),
            ],
            "outputs": [_signature("actions", ["batch", 8])],
        },
    }
    models = manifest.get("models", {})
    for model_name, signature in expected.items():
        if model_name not in models:
            raise ContractError(f"Manifest is missing models.{model_name}.")
        for direction in ("inputs", "outputs"):
            _require_equal(models[model_name].get(direction), signature[direction], f"models.{model_name}.{direction}")
        _require_equal(
            models[model_name].get("files"),
            {"onnx": f"{model_name}.onnx", "pt": f"{model_name}.pt"},
            f"models.{model_name}.files",
        )


def _validate_deployment(deployment: dict[str, Any]) -> None:
    """校验部署侧维度、顺序、控制参数和命令范围。"""
    _require_equal(deployment.get("schema_version"), 1, "deployment.schema_version")
    _require_equal(deployment.get("proprioception_dim"), 28, "deployment.proprioception_dim")
    _require_equal(deployment.get("history_samples"), 10, "deployment.history_samples")
    _require_equal(deployment.get("command_dim"), 3, "deployment.command_dim")
    _require_equal(deployment.get("action_dim"), 8, "deployment.action_dim")
    _require_equal(deployment.get("physics_period_s"), 0.005, "deployment.physics_period_s")
    _require_equal(deployment.get("decimation"), 2, "deployment.decimation")
    _require_equal(deployment.get("control_period_s"), 0.01, "deployment.control_period_s")
    _require_equal(
        deployment.get("command_order"),
        ["linear_velocity_x", "linear_velocity_y", "angular_velocity_z"],
        "deployment.command_order",
    )
    policy_order = deployment.get("policy_action_order")
    hardware_order = deployment.get("hardware_dof_order")
    _require_equal(policy_order, EXPECTED_POLICY_ORDER, "deployment.policy_action_order")
    _require_equal(hardware_order, EXPECTED_HARDWARE_ORDER, "deployment.hardware_dof_order")
    _require_equal(
        deployment.get("proprioception_layout"),
        [
            {"name": "base_angular_velocity", "size": 3, "scale": 0.25, "frame": "robot_base"},
            {"name": "projected_gravity", "size": 3, "scale": 1.0, "frame": "robot_base"},
            {"name": "leg_joint_position_relative", "size": 6, "scale": 1.0, "order": policy_order[:6]},
            {"name": "joint_velocity_relative", "size": 8, "scale": 0.05, "order": policy_order},
            {"name": "previous_action", "size": 8, "scale": 1.0, "order": policy_order},
        ],
        "deployment.proprioception_layout",
    )
    # 显式校验双向索引，确保 policy_order 与 MuJoCo/hardware_order 的转换可追踪。
    expected_policy_to_hardware = [policy_order.index(name) for name in hardware_order]
    expected_hardware_to_policy = [hardware_order.index(name) for name in policy_order]
    _require_equal(
        deployment.get("policy_actions_to_hardware_indices"),
        expected_policy_to_hardware,
        "deployment.policy_actions_to_hardware_indices",
    )
    _require_equal(
        deployment.get("hardware_state_to_policy_indices"),
        expected_hardware_to_policy,
        "deployment.hardware_state_to_policy_indices",
    )
    defaults = deployment.get("default_joint_positions", {})
    controls = deployment.get("joint_control", {})
    _require_equal(defaults.get("order"), policy_order, "deployment.default_joint_positions.order")
    _require_equal(defaults.get("values"), EXPECTED_DEFAULT_Q, "deployment.default_joint_positions.values")
    _require_equal(controls.get("order"), policy_order, "deployment.joint_control.order")
    _require_equal(controls.get("modes"), ["position"] * 6 + ["velocity"] * 2, "deployment.joint_control.modes")
    _require_equal(controls.get("stiffness"), EXPECTED_STIFFNESS, "deployment.joint_control.stiffness")
    _require_equal(controls.get("damping"), EXPECTED_DAMPING, "deployment.joint_control.damping")
    _require_equal(controls.get("effort_limits"), EXPECTED_EFFORT_LIMITS, "deployment.joint_control.effort_limits")
    _require_equal(
        controls.get("velocity_limits"), EXPECTED_VELOCITY_LIMITS, "deployment.joint_control.velocity_limits"
    )
    semantics = deployment.get("policy_action_semantics", {})
    _require_equal(
        semantics.get("leg_position", {}).get("joints"),
        policy_order[:6],
        "deployment.policy_action_semantics.leg_position.joints",
    )
    _require_equal(
        semantics.get("leg_position", {}).get("scale"),
        0.25,
        "deployment.policy_action_semantics.leg_position.scale",
    )
    _require_equal(
        semantics.get("leg_position", {}).get("uses_default_offset"),
        True,
        "deployment.policy_action_semantics.leg_position.uses_default_offset",
    )
    _require_equal(
        semantics.get("wheel_velocity", {}).get("joints"),
        policy_order[6:],
        "deployment.policy_action_semantics.wheel_velocity.joints",
    )
    _require_equal(
        semantics.get("wheel_velocity", {}).get("scale"),
        1.0,
        "deployment.policy_action_semantics.wheel_velocity.scale",
    )
    _require_equal(
        semantics.get("wheel_velocity", {}).get("uses_default_offset"),
        True,
        "deployment.policy_action_semantics.wheel_velocity.uses_default_offset",
    )
    command_limits = deployment.get("command_limits", {})
    _require_equal(command_limits.get("linear_velocity_x"), [-1.0, 1.0], "deployment.command_limits.linear_velocity_x")
    _require_equal(command_limits.get("linear_velocity_y"), [0.0, 0.0], "deployment.command_limits.linear_velocity_y")
    _require_equal(
        command_limits.get("angular_velocity_z"), [-1.0, 1.0], "deployment.command_limits.angular_velocity_z"
    )
    _require_equal(deployment.get("policy_output_clip"), 100.0, "deployment.policy_output_clip")
    _require_equal(
        deployment.get("shared_action_delay_physics_steps"), [0, 6], "deployment.shared_action_delay_physics_steps"
    )
    robot_model = deployment.get("robot_model", {})
    _require_equal(robot_model.get("name"), "Huilun-L5A-WF", "deployment.robot_model.name")
    _require_equal(robot_model.get("keyframe"), "home", "deployment.robot_model.keyframe")


'''
输入参数：
    model_dir（str | Path）：
        导出目录的路径，比如 logs/rsl_rl/l5a_wf_flat/2026-07-31_23-57-49/exported。
        这个目录下必须包含 policy_manifest.json 以及 4 个模型文件（policy.onnx、velocity_estimator.onnx、policy.pt、velocity_estimator.pt）。
    mjcf_path（str | Path）：
        L5A 机器人的 MuJoCo XML 模型文件路径，比如 resources/robots/l5a/xml/l5aurdf20260521.xml。
返回值：
    DeploymentBundle：一个不可变数据类（@dataclass(frozen=True)），包含三个字段：
    model_dir：解析后的模型目录绝对路径
    mjcf_path：解析后的 MJCF 文件绝对路径
    manifest：已解析的 policy_manifest.json 完整字典
可能抛出的异常：
    ContractError：继承自 RuntimeError，在校验的任何一步不通过时立即抛出，消息中会明确指出哪个字段出了问题。
'''
def load_bundle(model_dir: str | Path, mjcf_path: str | Path) -> DeploymentBundle:
    """加载导出目录并完成启动前契约验证。

    验证顺序：
    1. 读取 `policy_manifest.json`。
    2. 校验 format_version=2 和 split_velocity_estimator_actor。
    3. 校验 `velocity_estimator.onnx [B,10,28] -> [B,3]`。
    4. 校验 `policy.onnx [B,3]+[B,28]+[B,3] -> [B,8]`。
    5. 校验 28 维观测、关节顺序、动作语义、PD 参数和命令范围。
    6. 校验模型文件与 MJCF 的 SHA-256。

    Args:
        model_dir: 导出目录路径，包含 policy_manifest.json 和 ONNX/JIT 模型文件。
        mjcf_path: L5A 机器人 MJCF XML 文件路径。

    Returns:
        通过全部校验的 DeploymentBundle 实例。

    Raises:
        ContractError: 任何一项校验不通过时立即抛出。
    """
    # 路径解析 & 文件存在性，.resolve() 转绝对路径
    model_dir = Path(model_dir).expanduser().resolve()
    mjcf_path = Path(mjcf_path).expanduser().resolve()
    manifest_path = model_dir / "policy_manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"Manifest not found: {manifest_path}")
    if not mjcf_path.is_file():
        raise ContractError(f"MJCF not found: {mjcf_path}")
    # 读取 `policy_manifest.json`
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)

    # 只接受 v2 拆分模型；旧 format_version=1 的组合 policy 不做隐式兼容。
    _require_equal(manifest.get("format_version"), 2, "format_version")
    _require_equal(manifest.get("policy_type"), "split_velocity_estimator_actor", "policy_type")
    _require_equal(manifest.get("history_order"), "oldest_to_newest", "history_order")
    _require_equal(
        manifest.get("actor_input_order"),
        ["estimated_base_linear_velocity", "proprioception", "commands"],
        "actor_input_order",
    )
    source = manifest.get("source", {})
    _require_equal(source.get("training_task"), "Huilun-L5A-WF-Flat-v0", "source.training_task")
    _require_equal(source.get("export_task"), "Huilun-L5A-WF-Flat-Play-v0", "source.export_task")
    _validate_model_signatures(manifest)
    _validate_deployment(manifest.get("deployment", {}))

    artifacts = manifest.get("artifacts", {})
    required_artifacts = {"policy.onnx", "velocity_estimator.onnx", "policy.pt", "velocity_estimator.pt"}
    _require_equal(set(artifacts), required_artifacts, "artifacts filenames")
    for filename in sorted(required_artifacts):
        artifact_path = model_dir / filename
        if not artifact_path.is_file():
            raise ContractError(f"Required model not found: {artifact_path}")
        expected_hash = artifacts.get(filename, {}).get("sha256")
        if not expected_hash:
            raise ContractError(f"Manifest has no SHA-256 for {filename}.")
        actual_hash = sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise ContractError(f"SHA-256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}.")

    # MJCF hash 把机器人 XML 也纳入契约，防止用错右髋初始角或非平地场景。
    expected_mjcf_hash = manifest["deployment"]["robot_model"].get("mjcf_sha256")
    actual_mjcf_hash = sha256_file(mjcf_path)
    if actual_mjcf_hash != expected_mjcf_hash:
        raise ContractError(
            f"MJCF SHA-256 mismatch for {mjcf_path}: expected {expected_mjcf_hash}, got {actual_mjcf_hash}."
        )
    return DeploymentBundle(model_dir=model_dir, mjcf_path=mjcf_path, manifest=manifest)
