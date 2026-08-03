"""运行时默认路径和轻量 YAML 配置。

这里不保存控制契约；关节顺序、PD、观测布局等必须来自导出目录中的
policy_manifest.json。这个文件只保存模型/MJCF 默认路径和 viewer 运行参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPOSITORY_ROOT / "logs" / "rsl_rl" / "l5a_wf_flat" / "2026-07-31_23-57-49" / "exported"
DEFAULT_MJCF = REPOSITORY_ROOT / "resources" / "robots" / "l5a" / "xml" / "l5aurdf20260521.xml"


@dataclass(frozen=True)
class RuntimeDefaults:
    """不影响训练契约的运行偏好，例如键盘步进和默认运行时长。

    字段说明：
        vx_step: 每次按 W/S 时前进速度的增减量，m/s。
        wz_step: 每次按 A/D 时偏航速度的增减量，rad/s。
        duration_s: 默认仿真时长，秒。0 表示无限制（viewer 模式）。
        realtime_factor: 实时因子，>1 加速，<1 减速。
        fall_height_m: 摔倒判定高度阈值，基座 z 低于此值视为摔倒。
    """

    vx_step: float
    wz_step: float
    duration_s: float
    realtime_factor: float
    fall_height_m: float

    @classmethod
    def load(cls, path: Path | None = None) -> RuntimeDefaults:
        """从包内 YAML 读取默认值；测试或调试时也可传入外部 YAML。

        Args:
            path: 可选外部 YAML 路径，不传则使用包内 data/l5a_wf_flat.yaml。
        """
        config_path = path or files("l5a_sim2sim").joinpath("data/l5a_wf_flat.yaml")
        with config_path.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        return cls(
            vx_step=float(values["command_step"]["linear_velocity_x"]),
            wz_step=float(values["command_step"]["angular_velocity_z"]),
            duration_s=float(values["duration_s"]),
            realtime_factor=float(values["realtime_factor"]),
            fall_height_m=float(values["fall_height_m"]),
        )
