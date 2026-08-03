"""键盘命令测试。

不需要打开 MuJoCo viewer，直接调用 key_callback 验证 W/S/A/D/C/R/Space
对速度命令、暂停和 reset 标志的影响。
"""

from __future__ import annotations

import numpy as np
from l5a_sim2sim.keyboard import KeyboardCommand


def test_keyboard_commands_clamp_clear_pause_and_reset() -> None:
    """速度命令要被限幅，reset 是一次性消费，C 清零且 vy 始终为 0。"""
    keyboard = KeyboardCommand(
        vx=0.95,
        wz=-0.95,
        vx_step=0.1,
        wz_step=0.1,
        command_limits={"linear_velocity_x": [-1.0, 1.0], "angular_velocity_z": [-1.0, 1.0]},
    )
    keyboard.key_callback(ord("W"))
    keyboard.key_callback(ord("D"))
    np.testing.assert_allclose(keyboard.command(), [1.0, 0.0, -1.0])

    keyboard.key_callback(ord("A"))
    keyboard.key_callback(ord("S"))
    np.testing.assert_allclose(keyboard.command(), [0.9, 0.0, -0.9])

    keyboard.key_callback(32)
    assert keyboard.paused
    keyboard.key_callback(ord("R"))
    assert keyboard.consume_reset()
    assert not keyboard.consume_reset()
    keyboard.key_callback(ord("C"))
    np.testing.assert_array_equal(keyboard.command(), np.zeros(3))
