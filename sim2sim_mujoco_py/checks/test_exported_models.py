"""真实导出模型的集成测试。

这些测试使用目标 exported 目录，保护双模型导出语义：ONNX 拆分模型应与旧组合
网络数值一致，JIT 与 ONNX 的 Encoder/Actor 拆分语义也应一致。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import REPOSITORY_ROOT
from l5a_sim2sim.manifest import load_bundle
from l5a_sim2sim.policy import SplitOnnxPolicy

# 指向仓库内训练产出的导出目录
MODEL_DIR = REPOSITORY_ROOT / "logs" / "rsl_rl" / "l5a_wf_flat" / "2026-07-31_23-57-49" / "exported"
MJCF_PATH = REPOSITORY_ROOT / "resources" / "robots" / "l5a" / "xml" / "l5aurdf20260521.xml"
# 旧版合并模型备份路径，仅当文件存在时才会参与对比测试
OLD_COMBINED_POLICY = Path("/tmp/l5a_wf_combined_policy_v1.onnx")


@pytest.mark.integration
def test_split_models_match_previous_combined_policy() -> None:
    """随机输入下，拆分后的 Encoder -> Actor 应复现旧组合 policy 的输出。"""
    ort = pytest.importorskip("onnxruntime")
    if not OLD_COMBINED_POLICY.is_file():
        pytest.skip(f"Previous combined model backup is unavailable: {OLD_COMBINED_POLICY}")
    bundle = load_bundle(MODEL_DIR, MJCF_PATH)
    split = SplitOnnxPolicy(bundle)
    combined = ort.InferenceSession(str(OLD_COMBINED_POLICY), providers=["CPUExecutionProvider"])
    random = np.random.default_rng(20260801)
    for _ in range(8):
        proprioception = random.normal(size=(4, 28)).astype(np.float32)
        history = random.normal(size=(4, 10, 28)).astype(np.float32)
        commands = random.uniform(-1.0, 1.0, size=(4, 3)).astype(np.float32)
        estimated_velocity = split.estimator.run(["estimated_base_linear_velocity"], {"observation_history": history})[
            0
        ]
        split_actions = split.actor.run(
            ["actions"],
            {
                "estimated_base_linear_velocity": estimated_velocity,
                "proprioception": proprioception,
                "commands": commands,
            },
        )[0]
        combined_actions = combined.run(
            ["actions"],
            {
                "proprioception": proprioception,
                "observation_history": history,
                "commands": commands,
            },
        )[0]
        np.testing.assert_allclose(split_actions, combined_actions, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.integration
def test_jit_models_have_the_same_split_semantics_as_onnx() -> None:
    """确认 policy.pt/velocity_estimator.pt 与同名 ONNX 一样都是拆分模型。"""
    torch = pytest.importorskip("torch")
    bundle = load_bundle(MODEL_DIR, MJCF_PATH)
    split = SplitOnnxPolicy(bundle)
    estimator_jit = torch.jit.load(str(MODEL_DIR / "velocity_estimator.pt")).eval()
    actor_jit = torch.jit.load(str(MODEL_DIR / "policy.pt")).eval()
    random = np.random.default_rng(41000)
    history = random.normal(size=(3, 10, 28)).astype(np.float32)
    proprioception = random.normal(size=(3, 28)).astype(np.float32)
    commands = random.uniform(-1.0, 1.0, size=(3, 3)).astype(np.float32)

    velocity_onnx = split.estimator.run(["estimated_base_linear_velocity"], {"observation_history": history})[0]
    action_onnx = split.actor.run(
        ["actions"],
        {
            "estimated_base_linear_velocity": velocity_onnx,
            "proprioception": proprioception,
            "commands": commands,
        },
    )[0]
    with torch.inference_mode():
        velocity_jit = estimator_jit(torch.from_numpy(history)).numpy()
        action_jit = actor_jit(
            torch.from_numpy(velocity_jit),
            torch.from_numpy(proprioception),
            torch.from_numpy(commands),
        ).numpy()
    np.testing.assert_allclose(velocity_jit, velocity_onnx, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(action_jit, action_onnx, rtol=1.0e-5, atol=1.0e-5)
