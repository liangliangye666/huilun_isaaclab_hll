# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``huilun_isaaclab`` 扩展的 Python 安装入口。

IsaacLab 以 editable extension 的方式安装本目录。新增 ``assets``、``learning``
或任务子包时无需手工登记，但必须保留各目录的 ``__init__.py``，这样
``find_packages()`` 才会把它们加入安装结果。
"""

import os
import sys

from setuptools import find_packages, setup

if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib

# 包元数据统一来自 extension.toml，避免 setup.py 与扩展管理器各维护一份版本号。
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(EXTENSION_PATH, "config", "extension.toml"), "rb") as f:
    EXTENSION_TOML_DATA = tomllib.load(f)

# 这里只列扩展自身的最小依赖；IsaacLab、Isaac Sim 和 RSL-RL 由训练环境提供。
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
]

# ``find_packages`` 对本项目很重要：它会包含嵌套的 l5a/mdp 和 learning/rsl_rl。
setup(
    name="huilun_isaaclab",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
