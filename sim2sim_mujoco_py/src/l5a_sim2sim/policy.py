"""双 ONNX 模型推理封装。

当前导出契约是 split_velocity_estimator_actor：
`velocity_estimator.onnx` 只做 Encoder，输入 `[B,10,28]` 历史；
`policy.onnx` 只做 Actor，输入估计速度 `[B,3]`、当前 proprioception `[B,28]`
和命令 `[B,3]`，输出 8 维 action。
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .manifest import ContractError, DeploymentBundle


class SplitOnnxPolicy:
    """加载并执行 Velocity Estimator + Actor 两个 ONNX Runtime session。"""

    # 创建 ONNX Runtime 会话
    def __init__(self, bundle: DeploymentBundle) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        # 使用 CPUExecutionProvider，策略网络很小，100 Hz 单线程推理足够稳定。
        self.estimator = ort.InferenceSession(      # 速度估计器
            str(bundle.model_path("velocity_estimator")), sess_options=options, providers=providers
        )
        self.actor = ort.InferenceSession(str(bundle.model_path("policy")), sess_options=options, providers=providers)  # 策略网络
        self.action_clip = float(bundle.deployment["policy_output_clip"])
        self._validate_session(
            self.estimator,
            [("observation_history", [None, 10, 28], "tensor(float)")],
            [("estimated_base_linear_velocity", [None, 3], "tensor(float)")],
            "velocity_estimator.onnx",
        )
        self._validate_session(
            self.actor,
            [
                ("estimated_base_linear_velocity", [None, 3], "tensor(float)"),
                ("proprioception", [None, 28], "tensor(float)"),
                ("commands", [None, 3], "tensor(float)"),
            ],
            [("actions", [None, 8], "tensor(float)")],
            "policy.onnx",
        )

    @staticmethod
    def _validate_session(session, expected_inputs, expected_outputs, filename: str) -> None:
        """对比 ONNX session 的实际输入输出签名与契约预期是否一致。

        Args:
            session: 已加载的 ONNX Runtime InferenceSession。
            expected_inputs: 预期输入列表，每项 `(name, [batch, ...], "tensor(float)")`。
            expected_outputs: 预期输出列表，格式同上。
            filename: 用于报错信息中的文件名标识。
        """
        def normalize_shape(shape):
            """把 ONNX 返回的 dim_value 为 0 的 batch 维归一化为 None。"""
            return [None if index == 0 else value for index, value in enumerate(shape)]

        actual_inputs = [(item.name, normalize_shape(item.shape), item.type) for item in session.get_inputs()]
        actual_outputs = [(item.name, normalize_shape(item.shape), item.type) for item in session.get_outputs()]
        if actual_inputs != expected_inputs:
            raise ContractError(
                f"{filename} input signature mismatch: expected {expected_inputs}, got {actual_inputs}."
            )
        if actual_outputs != expected_outputs:
            raise ContractError(
                f"{filename} output signature mismatch: expected {expected_outputs}, got {actual_outputs}."
            )

    def infer(
        self, observation_history: np.ndarray, proprioception: np.ndarray, commands: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行一次策略周期推理。

        输入 shape 固定为 history `(1,10,28)`、proprioception `(1,28)`、
        commands `(1,3)`；返回去掉 batch 维后的估计速度 `(3,)` 和 action `(8,)`。
        """
        history = np.ascontiguousarray(observation_history, dtype=np.float32)
        proprioception = np.ascontiguousarray(proprioception, dtype=np.float32)
        commands = np.ascontiguousarray(commands, dtype=np.float32)
        if history.shape != (1, 10, 28):
            raise ValueError(f"Expected history shape (1, 10, 28), got {history.shape}.")
        if proprioception.shape != (1, 28):
            raise ValueError(f"Expected proprioception shape (1, 28), got {proprioception.shape}.")
        if commands.shape != (1, 3):
            raise ValueError(f"Expected commands shape (1, 3), got {commands.shape}.")
        # 数据流：10 帧历史先进入 Encoder，估计出的 base linear velocity 再作为 Actor 输入。
        # 第一步：速度估计器用历史观测估计基座线速度
        estimated_velocity = self.estimator.run(["estimated_base_linear_velocity"], {"observation_history": history})[0]
        # 第二步：策略网络根据估计速度 + 当前观测 + 命令输出动作
        actions = self.actor.run(
            ["actions"],
            {
                "estimated_base_linear_velocity": estimated_velocity,
                "proprioception": proprioception,
                "commands": commands,
            },
        )[0]
        if estimated_velocity.shape != (1, 3) or actions.shape != (1, 8):
            raise RuntimeError(
                f"Unexpected inference output shapes: velocity={estimated_velocity.shape}, actions={actions.shape}."
            )
        if not np.all(np.isfinite(estimated_velocity)) or not np.all(np.isfinite(actions)):
            raise FloatingPointError("ONNX inference returned NaN or Inf.")
        actions = np.clip(actions, -self.action_clip, self.action_clip).astype(np.float32, copy=False)
        return estimated_velocity[0].copy(), actions[0].copy()
