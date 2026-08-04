"""读取部署 Manifest，并执行 Encoder + Actor 双 ONNX 推理。

``policy_manifest.json`` 是训练导出侧和部署运行侧之间的数据契约。本模块只校验
Python sim2sim 真正依赖的字段与 ONNX 签名，不绑定具体机器人、任务名或固定维度。

.. rubric:: 为什么要分成 Encoder 和 Actor 两个模型？

这是 RSL-RL 训练框架的一种常见设计模式：

- **Encoder（速度估计器）**：接收历史 10 帧 proprioception [1,10,28]，输出估计的
  基座线速度 [1,3]。它的作用是弥补仿真中缺失的真实速度传感器——MuJoCo 里的速度
  传感器可能有噪声或延迟，Encoder 学会从关节角度变化和 IMU 数据中"推算"速度。

- **Actor（策略网络）**：接收估计速度 + 当前 proprioception [1,28] + 命令 [1,3]，
  输出原始动作 [1,8]。它只需要知道"当前状态"来做决策，不需要关心历史。

分离的好处：Encoder 可以独立训练和验证（比如用真实速度作为监督信号），
Actor 专注于策略优化。推理时两者串联即可。

.. rubric:: 为什么要在 __init__ 里做这么多签名校验？

ONNX 模型加载成功不等于推理不会报错。如果 Manifest 记录的 shape/dtype 和实际
ONNX 文件不一致（比如训练时改了输出维度但没更新 Manifest），错误会在第一次
``session.run()`` 时才暴露，排查起来很麻烦。提前校验可以"fail fast"——
启动时就明确告诉你哪个字段对不上，而不是跑了一半才崩溃。

推理链路为：历史观测 -> velocity estimator -> 估计基座线速度 -> actor -> action。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


class DeploymentError(RuntimeError):
    """部署文件缺失或运行契约不完整。"""


def _required(mapping: dict[str, Any], key: str, location: str) -> Any:
    """读取运行必需字段，并给出比普通 KeyError 更明确的错误。"""
    if key not in mapping:
        raise DeploymentError(f"policy_manifest.json 缺少必需字段：{location}.{key}")
    return mapping[key]


@dataclass(frozen=True)
class DeploymentBundle:
    """已经读取的模型目录、MJCF 路径和 Manifest 内容。

    使用不可变数据类，避免运行期间意外替换模型目录或 XML 路径。Manifest 字典
    仍按原始 JSON 保存，具体模块只读取自己需要的字段。
    """

    model_dir: Path
    mjcf_path: Path
    manifest: dict[str, Any]

    @property
    def deployment(self) -> dict[str, Any]:
        """返回 Manifest 中记录仿真、观测和控制参数的 ``deployment`` 字段。"""
        return _required(self.manifest, "deployment", "root")

    def model_config(self, model_name: str) -> dict[str, Any]:
        """返回指定 ONNX 模型的文件名及输入输出签名。"""
        models = _required(self.manifest, "models", "root")
        return _required(models, model_name, "models")

    def model_path(self, model_name: str) -> Path:
        """根据 Manifest 解析 ONNX 文件路径，并确认文件存在。"""
        model_config = self.model_config(model_name)                        # 从 manifest["models"] 读取配置
        files = _required(model_config, "files", f"models.{model_name}")    # 必须有 "files" 字段
        filename = _required(files, "onnx", f"models.{model_name}.files")   # 必须有 "onnx" 文件名
        path = self.model_dir / str(filename)                               # 拼接完整路径
        if not path.is_file():                                              # 检查文件是否存在
            raise DeploymentError(f"找不到 ONNX 模型：{path}")
        return path


'''
整个 sim2sim 系统的 "启动验证器"
它在一切仿真开始之前，检查部署所需的三个关键文件（Manifest JSON + 两个 ONNX 模型）是否齐全、格式是否正确，并把它们打包成一个不可变的 DeploymentBundle 数据容器。
'''
def load_deployment(model_dir: str | Path, mjcf_path: str | Path) -> DeploymentBundle:
    """读取部署目录和 MuJoCo XML，并做启动所需的最小完整性检查。

    Args:
        model_dir: 训练导出的模型目录路径：包含 ``policy_manifest.json`` 和两个 ONNX 文件（velocity_estimator.onnx 和 policy.onnx）。
        mjcf_path: 与当前策略和 Manifest 匹配的 MuJoCo XML 路径。

    Returns:
        保存了三个字段——model_dir（绝对路径）、mjcf_path（绝对路径）、manifest（解析后的 JSON 字典）。
        它是后续所有模块（MujocoAdapter、SplitOnnxPolicy、ObservationBuilder 等）读取配置的唯一入口。

    Raises:
        DeploymentError: 文件缺失、JSON 无法解析或必需模型字段不存在。

    这里不会校验固定任务名、固定关节数或文件 hash；这些内容可能随新机器人变化。
    ONNX 名称、shape 和 dtype 会在创建 :class:`SplitOnnxPolicy` 时继续核对。
    """
    # Path() 把字符串包装成路径对象；expanduser() 把 ~ 展开成用户主目录（如 /home/username）；resolve() 把相对路径解析为绝对路径。
    model_dir = Path(model_dir).expanduser().resolve()
    mjcf_path = Path(mjcf_path).expanduser().resolve()
    manifest_path = model_dir / "policy_manifest.json"  # 拼接 Manifest 路径

    if not manifest_path.is_file():
        raise DeploymentError(f"找不到 policy_manifest.json：{manifest_path}")
    if not mjcf_path.is_file():
        raise DeploymentError(f"找不到 MuJoCo XML：{mjcf_path}")

    # 读取 `policy_manifest.json`
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except json.JSONDecodeError as error:
        raise DeploymentError(f"无法解析 {manifest_path}：{error}") from error
    if not isinstance(manifest, dict):
        raise DeploymentError("policy_manifest.json 顶层必须是 JSON object。")

    # 创建不可变数据容器
    bundle = DeploymentBundle(model_dir=model_dir, mjcf_path=mjcf_path, manifest=manifest)
    # ONNX 文件存在性检查
    # 启动前确认两个运行时 ONNX 文件确实存在。JIT 文件和 hash 不参与 Python sim2sim。
    bundle.model_path("velocity_estimator")
    bundle.model_path("policy")
    # Manifest 顶层字段校验
    _required(bundle.manifest, "deployment", "root")
    return bundle


def _runtime_shape(shape: list[Any]) -> tuple[int, ...]:
    """把 Manifest 的 [batch, ...] 转成单 batch 推理使用的具体 shape。"""
    if len(shape) < 2:
        raise DeploymentError(f"ONNX tensor shape 至少需要 batch 和 feature 维：{shape}")
    dimensions: list[int] = [1]
    for value in shape[1:]:
        if not isinstance(value, int) or value <= 0:
            raise DeploymentError(f"除 batch 外的 ONNX tensor 维度必须是正整数：{shape}")
        dimensions.append(value)
    return tuple(dimensions)


def _normalized_shape(shape: list[Any]) -> list[Any]:
    """忽略动态 batch 维的符号名称，保留其余维度进行签名比较。"""
    if not shape:
        return shape
    return [None, *shape[1:]]


def _onnx_type(dtype: str) -> str:
    """把 Manifest dtype 转成 ONNX Runtime 使用的类型字符串。"""
    if dtype != "float32":
        raise DeploymentError(f"当前运行器只支持 float32 ONNX tensor，收到 {dtype!r}。")
    return "tensor(float)"


class SplitOnnxPolicy:
    """加载并执行 Velocity Estimator 与 Actor 两个 ONNX Runtime session。

    所有 tensor 的名称和 shape 都从 Manifest 读取。例如当前 L5A 模型的数据流是
    ``history [1,10,28] -> velocity [1,3]``，随后 Actor 接收估计速度、当前
    proprioception 和 commands，输出 ``action [1,8]``。更换机器人后维度可以变化，
    但 Actor 的三个输入语义保持不变。
    """

    def __init__(self, bundle: DeploymentBundle) -> None:
        """读取动态 tensor 签名，并创建两个 CPU ONNX Runtime 会话。

        初始化流程：
        1. 从 Manifest 读取 Encoder 和 Actor 的输入/输出 tensor 签名
        2. 验证 Encoder 是 1 输入 1 输出，Actor 是 3 输入 1 输出
        3. 验证 Actor 的三个输入名称必须是固定语义名
        4. 验证 Encoder 输出 shape 与 Actor 的 estimated_base_linear_velocity 输入一致
        5. 创建 ONNX Runtime 会话（单线程，避免 100Hz 循环中的调度抖动）
        6. 执行 Manifest 与 ONNX 文件的签名交叉验证（fail fast）
        """
        self.bundle = bundle
        estimator_config = bundle.model_config("velocity_estimator")
        actor_config = bundle.model_config("policy")

        # ---- 读取并验证 tensor 签名 ----
        self.estimator_inputs = _required(estimator_config, "inputs", "models.velocity_estimator")
        self.estimator_outputs = _required(estimator_config, "outputs", "models.velocity_estimator")
        self.actor_inputs = _required(actor_config, "inputs", "models.policy")
        self.actor_outputs = _required(actor_config, "outputs", "models.policy")
        if len(self.estimator_inputs) != 1 or len(self.estimator_outputs) != 1:
            raise DeploymentError("velocity_estimator.onnx 必须具有一个输入和一个输出。")
        if len(self.actor_inputs) != 3 or len(self.actor_outputs) != 1:
            raise DeploymentError("policy.onnx 必须具有三个输入和一个输出。")

        self.estimator_input = self.estimator_inputs[0]
        self.estimator_output = self.estimator_outputs[0]
        self.actor_input_by_name = {item["name"]: item for item in self.actor_inputs}
        # Actor 的三个输入名称是硬编码的语义约定——改了名字就说明 Manifest 版本不兼容
        required_actor_inputs = {
            "estimated_base_linear_velocity",
            "proprioception",
            "commands",
        }
        if set(self.actor_input_by_name) != required_actor_inputs:
            raise DeploymentError("policy.onnx 输入必须是 estimated_base_linear_velocity、proprioception 和 commands。")
        self.actor_output = self.actor_outputs[0]

        # ---- 从 Manifest 解析各 tensor 的运行时 shape ----
        # _runtime_shape 把 [batch, N, M] 转成 (1, N, M)，batch 固定为 1
        self.history_shape = _runtime_shape(self.estimator_input["shape"])
        self.estimated_velocity_shape = _runtime_shape(self.estimator_output["shape"])
        self.proprioception_shape = _runtime_shape(self.actor_input_by_name["proprioception"]["shape"])
        self.command_shape = _runtime_shape(self.actor_input_by_name["commands"]["shape"])
        self.action_shape = _runtime_shape(self.actor_output["shape"])
        actor_velocity_shape = _runtime_shape(self.actor_input_by_name["estimated_base_linear_velocity"]["shape"])
        # 关键校验：Encoder 的输出必须能直接喂给 Actor，否则两模型不匹配
        if actor_velocity_shape != self.estimated_velocity_shape:
            raise DeploymentError("Encoder 输出 shape 与 Actor 的 estimated_base_linear_velocity 输入 shape 不一致。")

        # ---- 创建 ONNX Runtime 会话 ----
        # 策略网络较小，限制为单线程可减少 100 Hz 循环中的线程调度抖动。
        # graph_optimization_level 设为 ALL 以启用 ONNX Runtime 的所有图优化。
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self.estimator = ort.InferenceSession(
            str(bundle.model_path("velocity_estimator")), sess_options=options, providers=providers
        )
        self.actor = ort.InferenceSession(str(bundle.model_path("policy")), sess_options=options, providers=providers)

        # ---- 交叉验证：Manifest vs 实际 ONNX 文件 ----
        # 这步是最重要的"fail fast"检查。如果 Manifest 说 shape 是 [1,10,28]
        # 但 ONNX 文件实际是 [1,10,32]，这里会立即报错而不是等到推理时才崩溃。
        self._validate_session(
            self.estimator,
            self.estimator_inputs,
            self.estimator_outputs,
            bundle.model_path("velocity_estimator").name,
        )
        self._validate_session(
            self.actor,
            self.actor_inputs,
            self.actor_outputs,
            bundle.model_path("policy").name,
        )
        # policy_output_clip 用于裁剪 Actor 输出，防止极端值导致 PD 控制器发散
        self.action_clip = float(bundle.deployment.get("policy_output_clip", np.inf))

    @staticmethod
    def _validate_session(
        session: ort.InferenceSession,
        expected_inputs: list[dict[str, Any]],
        expected_outputs: list[dict[str, Any]],
        filename: str,
    ) -> None:
        """核对 Manifest 与 ONNX 文件自己的输入输出名称、shape 和 dtype。

        batch 维可能在导出时使用不同符号名，因此比较时统一归一化为 ``None``；
        feature 维必须精确匹配，防止模型可以加载但在第一次推理时才报 shape 错误。
        """

        def expected_signature(items: list[dict[str, Any]]) -> list[tuple[str, list[Any], str]]:
            return [(item["name"], _normalized_shape(item["shape"]), _onnx_type(item["dtype"])) for item in items]

        actual_inputs = [(item.name, _normalized_shape(item.shape), item.type) for item in session.get_inputs()]
        actual_outputs = [(item.name, _normalized_shape(item.shape), item.type) for item in session.get_outputs()]
        expected_input_signature = expected_signature(expected_inputs)
        expected_output_signature = expected_signature(expected_outputs)
        if actual_inputs != expected_input_signature:
            raise DeploymentError(
                f"{filename} 输入签名与 Manifest 不一致：{actual_inputs} != {expected_input_signature}"
            )
        if actual_outputs != expected_output_signature:
            raise DeploymentError(
                f"{filename} 输出签名与 Manifest 不一致：{actual_outputs} != {expected_output_signature}"
            )

    @property
    def history_samples(self) -> int:
        """返回 Encoder 历史窗口包含的观测帧数。"""
        return self.history_shape[1]

    @property
    def proprioception_dim(self) -> int:
        """返回 Actor 当前 proprioception 的维度。"""
        return self.proprioception_shape[1]

    @property
    def command_dim(self) -> int:
        """返回 Actor commands 输入的维度。"""
        return self.command_shape[1]

    @property
    def action_dim(self) -> int:
        """返回 Actor 输出动作的维度。"""
        return self.action_shape[1]

    def infer(
        self,
        observation_history: np.ndarray,
        proprioception: np.ndarray,
        commands: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行一次策略周期的双模型推理。

        输入均包含 batch 维且必须与 Manifest 完全一致。返回值会移除 batch 维，
        得到一维估计速度和一维原始 action，供主循环记录及 PD 控制使用。
        """
        history = np.ascontiguousarray(observation_history, dtype=np.float32)
        proprioception = np.ascontiguousarray(proprioception, dtype=np.float32)
        commands = np.ascontiguousarray(commands, dtype=np.float32)
        for name, value, expected_shape in (
            ("observation_history", history, self.history_shape),
            ("proprioception", proprioception, self.proprioception_shape),
            ("commands", commands, self.command_shape),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} shape 应为 {expected_shape}，实际为 {value.shape}。")

        # 第一步：Encoder 只使用 oldest-to-newest 历史窗口估计基座线速度。
        estimated_velocity = self.estimator.run(
            [self.estimator_output["name"]],
            {self.estimator_input["name"]: history},
        )[0]
        # 第二步：Actor 使用估计速度、当前本体观测和速度命令产生原始动作。
        actor_feed = {
            "estimated_base_linear_velocity": estimated_velocity,
            "proprioception": proprioception,
            "commands": commands,
        }
        actions = self.actor.run([self.actor_output["name"]], actor_feed)[0]
        if estimated_velocity.shape != self.estimated_velocity_shape:
            raise RuntimeError(
                f"Encoder 输出 shape 应为 {self.estimated_velocity_shape}，实际为 {estimated_velocity.shape}。"
            )
        if actions.shape != self.action_shape:
            raise RuntimeError(f"Actor 输出 shape 应为 {self.action_shape}，实际为 {actions.shape}。")
        if not np.all(np.isfinite(estimated_velocity)) or not np.all(np.isfinite(actions)):
            raise FloatingPointError("ONNX 推理结果包含 NaN 或 Inf。")

        # 输出裁剪来自训练导出参数；copy=False 避免不必要的数组复制。
        actions = np.clip(actions, -self.action_clip, self.action_clip).astype(np.float32, copy=False)
        return estimated_velocity[0].copy(), actions[0].copy()
