"""命令行入口和运行对象装配。

本文件只负责把用户输入的参数转换成运行时对象：
Manifest/MJCF -> DeploymentBundle，固定或键盘速度命令 -> KeyboardCommand，
最后交给 Sim2SimRunner 执行单进程 MuJoCo sim2sim 主循环。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import DEFAULT_MJCF, DEFAULT_MODEL_DIR, RuntimeDefaults
from .keyboard import KeyboardCommand
from .manifest import load_bundle
from .simulator import Sim2SimRunner


def build_parser(defaults: RuntimeDefaults) -> argparse.ArgumentParser:
    """定义公开 CLI 参数，并把不常改的默认值交给 YAML 配置维护。

    Args:
        defaults: 从 YAML 加载的运行默认值，用于 --duration 和 --realtime-factor 的默认值。
    """
    parser = argparse.ArgumentParser(description="Huilun L5A WF-Flat Python MuJoCo sim2sim")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--vx", type=float, default=0.0, help="Fixed forward velocity command in m/s.")
    parser.add_argument("--wz", type=float, default=0.0, help="Fixed yaw velocity command in rad/s.")
    parser.add_argument("--keyboard", action="store_true", help="Enable W/S/A/D/C/R/Space viewer controls.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=defaults.duration_s,
        help="Simulated seconds to run; use 0 for unlimited viewer operation.",
    )
    parser.add_argument("--realtime-factor", type=float, default=defaults.realtime_factor)
    parser.add_argument("--action-delay-steps", type=int, choices=range(0, 7), default=0)
    parser.add_argument("--trace", type=Path, default=None, help="Optional NPZ policy-step trace path.")
    return parser


'''
main()
 │
 ├─1. RuntimeDefaults.load()  ← 读取 YAML 默认参数
 ├─2. build_parser().parse_args()  ← 解析命令行
 ├─3. load_bundle()  ← 加载并校验模型契约（manifest.py）
 ├─4. KeyboardCommand()  ← 创建命令管理器
 ├─5. Sim2SimRunner()  ← 创建仿真运行器
 └─6. runner.run()  ← 进入主仿真循环
'''
def main(argv: list[str] | None = None) -> None:
    """解析命令行、校验互斥选项，然后启动仿真主循环。"""
    # 读取 YAML 默认参数
    defaults = RuntimeDefaults.load()
    # 解析命令行
    args = build_parser(defaults).parse_args(argv)
    if args.keyboard and args.headless:
        raise SystemExit("--keyboard requires the MuJoCo viewer; remove --headless.")
    if args.duration < 0.0:
        raise SystemExit("--duration must be non-negative.")
    if args.realtime_factor <= 0.0:
        raise SystemExit("--realtime-factor must be positive.")
    duration = None if args.duration == 0.0 else args.duration
    if args.headless and duration is None:
        raise SystemExit("Headless mode requires --duration greater than zero.")

    # 启动前先加载部署契约；如果模型、Manifest、MJCF 或关节顺序不匹配，
    # load_bundle 会在 MuJoCo/ONNX 真正运行前失败。
    # 加载并校验模型契约（manifest.py）
    bundle = load_bundle(args.model_dir, args.mjcf)
    # 创建命令管理器
    command = KeyboardCommand(
        args.vx,
        args.wz,
        defaults.vx_step,
        defaults.wz_step,
        bundle.deployment["command_limits"],
    )
    # 创建仿真运行器
    runner = Sim2SimRunner(
        bundle,
        command,
        action_delay_steps=args.action_delay_steps,
        fall_height_m=defaults.fall_height_m,
        trace_path=args.trace,
    )
    # 进入主仿真循环
    summary = runner.run(duration, args.realtime_factor, args.headless, args.keyboard)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
