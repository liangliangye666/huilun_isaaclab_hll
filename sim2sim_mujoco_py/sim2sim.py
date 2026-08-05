"""MuJoCo sim2sim 直接运行入口。
::

     ┌──────────────┐
     │  MuJoCo 仿真  │  ← 读取关节角度/角速度、基座姿态、角速度
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  观测构造     │  ← 拼接投影重力、关节误差、上一步动作 → 28 维 proprioception
     │  observation  │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  历史窗口     │  ← 维护最近 10 帧观测的滑动窗口（oldest → newest）
     │  history      │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Encoder      │  ← history [1,10,28] → estimated velocity [1,3]
     │  (ONNX)       │     估计机器人的基座线速度（只用历史，不看当前帧）
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Actor        │  ← estimated velocity + 当前 proprioception + 命令
     │  (ONNX)       │     → 原始 action [1,8]（8 个关节的目标）
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  PD 控制器    │  ← action → 位置/速度目标 → kp*(q_err) + kd*(dq_err)
     │  control      │     计算每个关节的力矩
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  MuJoCo 仿真  │  ← 力矩写入执行器，推进一个物理步（~5ms）
     └──────────────┘

一个策略周期 = 一次 Encoder + 一次 Actor + Manifest 指定的 decimation 个物理步。
策略动作在这些物理步中保持不变，但每个物理步都重新读取关节状态并重新计算 PD 力矩。

本文件只保存用户经常调整的运行参数。实际数据流由 ``src/simulator.py``负责组织。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from simulator import RuntimeConfig, run_simulation

# =============================================================================
# 常用配置区：通常只需要修改这里
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# ---- 模型与机器人 ----
# exported 目录必须包含 policy_manifest.json、policy.onnx 和 velocity_estimator.onnx。
MODEL_DIR = REPOSITORY_ROOT / "logs/rsl_rl/l5a_wf_flat/2026-08-05_01-13-53/exported"

# 与这组策略匹配的 MuJoCo XML。更换机器人时同时修改 MODEL_DIR 和 MJCF_PATH。
# XML 里定义了机器人的关节、质量、执行器、传感器等物理属性。
MJCF_PATH = REPOSITORY_ROOT / "resources/robots/l5a/xml/l5aurdf20260521.xml"

# ---- 命令（告诉机器人怎么走） ----
# 固定命令的排列必须与 policy_manifest.json 中的 command_order 相同。
# 当前顺序是 [linear_velocity_x, linear_velocity_y, angular_velocity_z]，
# 即 [前进速度(m/s), 侧移速度(m/s), 转向速度(rad/s)]。
FIXED_COMMAND = (0.2, 0.0, 0.0)

# ---- 键盘控制 ----
# False：始终使用 FIXED_COMMAND；True：允许在 viewer 中用键盘调整命令。
# W/S 调整前进速度，A/D 调整偏航，C 清零所有命令，R 复位机器人，空格暂停。
KEYBOARD_ENABLED = False
KEYBOARD_LINEAR_STEP = 0.1   # 每次按键前进速度的变化量（m/s）
KEYBOARD_YAW_STEP = 0.1      # 每次按键偏航速度的变化量（rad/s）

# ---- 仿真运行 ----
# False 打开 viewer 可视化窗口；True 无界面运行（适合批量测试或远程服务器）。
# DURATION_S=0 表示 viewer 一直运行到手动关闭窗口。
# REALTIME_FACTOR=1.0 表示实时速度；2.0 是两倍速；0.5 是半速（慢动作）。
HEADLESS = False
DURATION_S = 0.0
REALTIME_FACTOR = 1.0

# ---- 动作延迟（模拟真实通信延迟） ----
# 动作延迟按 MuJoCo 物理步计数（物理步通常 ~5ms）。
# 设为 2 表示约 10ms 延迟：当前策略动作要等 2 个物理步后才真正执行。
# 必须位于 Manifest 记录的 shared_action_delay_physics_steps 训练范围内。
ACTION_DELAY_STEPS = 0

# ---- 调试轨迹记录 ----
# 设置为 Path("trace.npz") 可保存策略步轨迹；None 表示不记录。
# 记录内容：时间、基座高度、关节位置/速度、命令、观测、估计速度、动作、力矩。
TRACE_PATH: Path | None = None

# ---- 摔倒检测 ----
# 基座高度低于此值就标记 fell=True，但不会自动复位或终止仿真。
# 这个值只影响 RunSummary 中的 fell 标志，不影响仿真行为。
FALL_HEIGHT_M = 0.35


def main() -> None:
    """把顶部配置整理为运行时对象，并启动一次完整的 sim2sim。"""
    runtime = RuntimeConfig(
        model_dir=MODEL_DIR,
        mjcf_path=MJCF_PATH,
        fixed_command=tuple(float(value) for value in FIXED_COMMAND),
        keyboard_enabled=KEYBOARD_ENABLED,
        keyboard_linear_step=KEYBOARD_LINEAR_STEP,
        keyboard_yaw_step=KEYBOARD_YAW_STEP,
        headless=HEADLESS,
        duration_s=DURATION_S,
        realtime_factor=REALTIME_FACTOR,
        action_delay_steps=ACTION_DELAY_STEPS,
        trace_path=TRACE_PATH,
        fall_height_m=FALL_HEIGHT_M,
    )
    summary = run_simulation(runtime)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
