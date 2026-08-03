"""MuJoCo viewer 键盘命令状态。

MuJoCo viewer 的 key_callback 可能由 viewer/GLFW 线程触发，而仿真主循环会在
另一个时刻读取命令。因此这里用 Lock 保护 `[vx, 0, wz]`、暂停和 reset 标志。
"""

from __future__ import annotations

import threading

import numpy as np


class KeyboardCommand:
    """保存固定命令或键盘更新后的速度命令。

    命令格式为 `[linear_velocity_x, 0, angular_velocity_z]`，其中 vy 始终为 0
    （L5A WF-Flat 任务不支持横向速度命令）。所有读写操作都用 Lock 保护，
    因为 MuJoCo viewer 的 key_callback 可能在 GLFW 线程中执行。
    """

    def __init__(self, vx: float, wz: float, vx_step: float, wz_step: float, command_limits: dict) -> None:
        """初始化命令管理器。

        Args:
            vx: 初始前进速度命令，m/s。
            wz: 初始偏航速度命令，rad/s。
            vx_step: 每次按 W/S 时前进速度的增减量。
            wz_step: 每次按 A/D 时偏航速度的增减量。
            command_limits: 命令范围字典，含 linear_velocity_x 和 angular_velocity_z 的 [min, max]。
        """
        self._lock = threading.Lock()       # 线程锁
        self._command = np.array([vx, 0.0, wz], dtype=np.float32)
        self.vx_step = float(vx_step)
        self.wz_step = float(wz_step)
        self.vx_limits = tuple(float(value) for value in command_limits["linear_velocity_x"])
        self.wz_limits = tuple(float(value) for value in command_limits["angular_velocity_z"])
        self.paused = False
        self.reset_requested = False
        self._clamp()

    def _clamp(self) -> None:
        """把速度命令裁剪到训练范围，并强制 vy 为 0。"""
        self._command[0] = np.clip(self._command[0], *self.vx_limits)
        self._command[1] = 0.0  # L5A WF-Flat 不支持横向速度命令
        self._command[2] = np.clip(self._command[2], *self.wz_limits)

    def command(self) -> np.ndarray:
        """返回 `[linear_velocity_x, 0, angular_velocity_z]` 的副本。"""
        with self._lock:
            return self._command.copy()

    def consume_reset(self) -> bool:
        """读取并清除一次性 reset 请求，避免一次按键触发多次 reset。"""
        with self._lock:
            requested = self.reset_requested
            self.reset_requested = False
            return requested

    def key_callback(self, keycode: int) -> None:
        """MuJoCo viewer 注册的按键回调。"""
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        with self._lock:
            if key == "W":
                self._command[0] += self.vx_step    # 加速前进
            elif key == "S":
                self._command[0] -= self.vx_step    # 减速/后退
            elif key == "A":
                self._command[2] += self.wz_step    # 左转
            elif key == "D":
                self._command[2] -= self.wz_step    # 右转
            elif key == "C":
                self._command[:] = 0.0              # 清零停止
            elif key == "R":
                self.reset_requested = True         # 重置
            elif keycode == 32:
                self.paused = not self.paused       # 空格暂停
            self._clamp()
