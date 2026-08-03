"""Manifest 契约测试。

这些测试不运行 MuJoCo 或 ONNX，只验证部署包在启动前能拒绝旧格式、错误签名、
关节顺序变化、模型 hash 错误和 MJCF hash 错误。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from l5a_sim2sim.manifest import ContractError, load_bundle, sha256_file


def _signature(name: str, shape: list[str | int]) -> dict:
    """构造模型签名条目，供 _write_bundle 生成 manifest。"""
    return {"name": name, "dtype": "float32", "shape": shape}


def _write_bundle(root: Path, deployment: dict) -> tuple[Path, Path, Path]:
    """在临时目录中写一套最小导出包，用于测试 load_bundle 的校验路径。

    生成的文件包括：4 个模型文件（policy.onnx、velocity_estimator.onnx、
    policy.pt、velocity_estimator.pt）、MJCF XML 和 policy_manifest.json。
    所有模型文件内容是伪数据（只校验 hash，不实际推理）。

    Args:
        root: pytest 临时目录根路径。
        deployment: deployment_contract fixture 提供的部署字典。

    Returns:
        (model_dir, mjcf_path, manifest_path) 三元组。
    """
    model_dir = root / "exported"
    model_dir.mkdir(parents=True)
    policy = model_dir / "policy.onnx"
    estimator = model_dir / "velocity_estimator.onnx"
    policy_jit = model_dir / "policy.pt"
    estimator_jit = model_dir / "velocity_estimator.pt"
    mjcf = root / "robot.xml"
    policy.write_bytes(b"policy")
    estimator.write_bytes(b"estimator")
    policy_jit.write_bytes(b"policy-jit")
    estimator_jit.write_bytes(b"estimator-jit")
    mjcf.write_text("<mujoco/>", encoding="utf-8")
    deployment = copy.deepcopy(deployment)
    deployment["robot_model"]["mjcf_sha256"] = sha256_file(mjcf)
    manifest = {
        "format_version": 2,
        "policy_type": "split_velocity_estimator_actor",
        "history_order": "oldest_to_newest",
        "actor_input_order": ["estimated_base_linear_velocity", "proprioception", "commands"],
        "models": {
            "velocity_estimator": {
                "files": {"onnx": "velocity_estimator.onnx", "pt": "velocity_estimator.pt"},
                "inputs": [_signature("observation_history", ["batch", 10, 28])],
                "outputs": [_signature("estimated_base_linear_velocity", ["batch", 3])],
            },
            "policy": {
                "files": {"onnx": "policy.onnx", "pt": "policy.pt"},
                "inputs": [
                    _signature("estimated_base_linear_velocity", ["batch", 3]),
                    _signature("proprioception", ["batch", 28]),
                    _signature("commands", ["batch", 3]),
                ],
                "outputs": [_signature("actions", ["batch", 8])],
            },
        },
        "source": {
            "training_task": "Huilun-L5A-WF-Flat-v0",
            "export_task": "Huilun-L5A-WF-Flat-Play-v0",
        },
        "deployment": deployment,
        "artifacts": {
            "policy.onnx": {"sha256": sha256_file(policy)},
            "velocity_estimator.onnx": {"sha256": sha256_file(estimator)},
            "policy.pt": {"sha256": sha256_file(policy_jit)},
            "velocity_estimator.pt": {"sha256": sha256_file(estimator_jit)},
        },
    }
    manifest_path = model_dir / "policy_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return model_dir, mjcf, manifest_path


def _mutate_manifest(path: Path, callback) -> None:
    """按测试提供的 callback 修改 Manifest，模拟导出包损坏或契约漂移。

    Args:
        path: policy_manifest.json 的路径。
        callback: 接受 manifest 字典并原地修改的可调用对象。
    """
    manifest = json.loads(path.read_text(encoding="utf-8"))
    callback(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_valid_manifest_loads(tmp_path: Path, deployment_contract: dict) -> None:
    """完整契约应当能被加载，并暴露 deployment 字段。"""
    model_dir, mjcf, _ = _write_bundle(tmp_path, deployment_contract)
    bundle = load_bundle(model_dir, mjcf)
    assert bundle.deployment["action_dim"] == 8


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda manifest: manifest.update(format_version=1), "format_version mismatch"),
        (
            lambda manifest: manifest["models"]["policy"]["inputs"][1].update(shape=["batch", 27]),
            "models.policy.inputs mismatch",
        ),
        (
            lambda manifest: manifest["deployment"].update(
                policy_action_order=list(reversed(manifest["deployment"]["policy_action_order"]))
            ),
            "deployment.policy_action_order mismatch",
        ),
    ],
)
def test_contract_mismatches_fail_before_runtime(tmp_path: Path, deployment_contract: dict, mutation, message) -> None:
    """格式、模型签名和关节顺序错误必须在运行前失败。"""
    model_dir, mjcf, manifest_path = _write_bundle(tmp_path, deployment_contract)
    _mutate_manifest(manifest_path, mutation)
    with pytest.raises(ContractError, match=message):
        load_bundle(model_dir, mjcf)


def test_model_and_mjcf_hash_mismatches_fail(tmp_path: Path, deployment_contract: dict) -> None:
    """模型或 XML 被替换后，SHA-256 校验必须阻止启动。"""
    model_dir, mjcf, _ = _write_bundle(tmp_path, deployment_contract)
    (model_dir / "policy.onnx").write_bytes(b"changed")
    with pytest.raises(ContractError, match="SHA-256 mismatch for policy.onnx"):
        load_bundle(model_dir, mjcf)

    model_dir, mjcf, _ = _write_bundle(tmp_path / "second", deployment_contract)
    mjcf.write_text("<mujoco model='changed'/>", encoding="utf-8")
    with pytest.raises(ContractError, match="MJCF SHA-256 mismatch"):
        load_bundle(model_dir, mjcf)
