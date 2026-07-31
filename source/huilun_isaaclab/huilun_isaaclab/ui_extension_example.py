# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse UI 扩展示例，与 L5A 强化学习训练无关。

Isaac Sim 根据 ``extension.toml`` 加载本模块，并调用 ``IExt`` 生命周期。
接手训练代码时可以跳过本文件；它既不创建环境，也不会进入 policy step。
"""

import omni.ext


# 普通公开函数可被其他 Omniverse 扩展通过 Python 包路径调用。
def some_public_function(x: int):
    print("[huilun_isaaclab] some_public_function was called with x: ", x)
    return x**x


# 扩展启用时，管理器实例化 IExt 子类并调用 on_startup；禁用时调用 on_shutdown。
class ExampleExtension(omni.ext.IExt):
    """最小 UI 生命周期示例，不参与训练。"""

    def on_startup(self, ext_id):
        """创建一个带计数按钮的演示窗口。``ext_id`` 是当前扩展标识。"""
        print("[huilun_isaaclab] startup")

        self._count = 0

        self._window = omni.ui.Window("My Window", width=300, height=300)
        with self._window.frame:
            with omni.ui.VStack():
                label = omni.ui.Label("")

                def on_click():
                    self._count += 1
                    label.text = f"count: {self._count}"

                def on_reset():
                    self._count = 0
                    label.text = "empty"

                on_reset()

                with omni.ui.HStack():
                    omni.ui.Button("Add", clicked_fn=on_click)
                    omni.ui.Button("Reset", clicked_fn=on_reset)

    def on_shutdown(self):
        """扩展关闭回调；窗口资源由 Omniverse UI 生命周期统一回收。"""
        print("[huilun_isaaclab] shutdown")
