"""无界面端到端测试。

用真实模型和真实 MJCF 跑短时 headless 仿真，保护主循环的基本闭环：无 NaN/Inf、
机器人未跌倒，并且 100 Hz 策略与 200 Hz 物理步保持 1:2 比例。
"""

from __future__ import annotations

import pytest
from conftest import REPOSITORY_ROOT
from l5a_sim2sim.keyboard import KeyboardCommand
from l5a_sim2sim.manifest import load_bundle
from l5a_sim2sim.simulator import Sim2SimRunner

# 指向仓库内训练产出的导出目录
MODEL_DIR = REPOSITORY_ROOT / "logs" / "rsl_rl" / "l5a_wf_flat" / "2026-07-31_23-57-49" / "exported"
# L5A 机器人 MJCF 模型文件路径
MJCF_PATH = REPOSITORY_ROOT / "resources" / "robots" / "l5a" / "xml" / "l5aurdf20260521.xml"


@pytest.mark.integration
@pytest.mark.parametrize("vx", [0.0, 0.2])
def test_headless_policy_and_physics_step_ratio(vx: float) -> None:
    """零速度和平动命令都应维持 decimation=2 的步数关系。"""
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    bundle = load_bundle(MODEL_DIR, MJCF_PATH)
    command = KeyboardCommand(vx, 0.0, 0.1, 0.1, bundle.deployment["command_limits"])
    runner = Sim2SimRunner(bundle, command)
    summary = runner.run(duration_s=1.0, realtime_factor=100.0, headless=True, keyboard_enabled=False)
    assert summary.physics_steps == 2 * summary.policy_steps
    assert summary.physics_steps == 200
    assert not summary.fell
