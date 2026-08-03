"""MJCF 机器人模型测试。

保护 L5A XML 的部署假设：只保留平地 plane、存在 home keyframe、右髋初始角正确，
并且 MuJoCo 编译后的 nq/nv/nu、物理步长和关键传感器名称符合 sim2sim 契约。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from conftest import REPOSITORY_ROOT

# L5A 机器人 MJCF 模型文件路径
MJCF_PATH = REPOSITORY_ROOT / "resources" / "robots" / "l5a" / "xml" / "l5aurdf20260521.xml"


def test_mjcf_source_has_flat_world_and_named_home_keyframe() -> None:
    """直接检查 XML 源文件，避免场景或 keyframe 被无意改回旧版本。"""
    root = ET.parse(MJCF_PATH).getroot()
    world_geoms = root.findall("./worldbody/geom")
    assert [geom.get("name") for geom in world_geoms] == ["plane"]
    key = root.find("./keyframe/key")
    assert key is not None
    assert key.get("name") == "home"
    qpos = [float(value) for value in key.get("qpos", "").split()]
    assert len(qpos) == 15
    assert qpos[11] == pytest.approx(-0.0523599)


def test_mjcf_compiles_with_expected_dimensions_names_and_timing() -> None:
    """实际编译 MJCF，验证 MuJoCo 看到的维度、传感器和 keyframe 位置。"""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    assert (model.nq, model.nv, model.nu) == (15, 14, 8)
    assert model.opt.timestep == pytest.approx(0.005)
    world_geoms = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] == 0
    }
    assert world_geoms == {"plane"}
    for name in ("orientation", "angular-velocity", "linear-acceleration"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) >= 0
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    right_hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_roll_joint")
    right_hip_qpos_address = model.jnt_qposadr[right_hip_id]
    assert np.asarray(model.key_qpos[key_id])[right_hip_qpos_address] == pytest.approx(-0.0523599)
