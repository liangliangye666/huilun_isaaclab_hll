# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""将 L5A 的 Encoder 与 Actor 导出为可部署的 JIT/ONNX 产物。

训练时 policy 接收 TensorDict，部署端则通常只有固定顺序的张量。本模块把边界收敛为
三个显式输入：当前本体感知、10 帧观测历史和速度指令，并同时导出：

* ``policy.*``：Encoder + Actor 的完整控制策略；
* ``velocity_estimator.*``：仅 Encoder，便于离线评估速度估计误差；
* ``policy_manifest.json``：输入输出形状、拼接顺序、部署元数据与文件哈希。

所有文件先写入临时 staging 目录，产物就位后最后原子替换 manifest。部署程序应把
manifest 当作一次发布的提交标记，并用其中的 SHA-256 校验对应模型，避免加载半套产物。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile

import torch
import torch.nn as nn

from .estimator_actor_critic import VelocityEstimatorActorCritic


def _sha256_file(path: str) -> str:
    """流式计算文件 SHA-256，避免把较大的 ONNX/JIT 文件整体读入内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


'''
把 TensorDict 接口收敛为三个张量输入（本体感知、历史、指令），方便 JIT/ONNX 跟踪。
'''
class _CombinedVelocityEstimatorPolicy(nn.Module):
    """面向部署的无 TensorDict 包装：历史 -> Encoder -> Actor -> 动作。"""

    def __init__(self, policy: VelocityEstimatorActorCritic) -> None:
        super().__init__()
        # deepcopy 将导出副本与仍在训练的模块解耦；调用方随后会统一切换到 CPU/eval。
        self.estimator = copy.deepcopy(policy.velocity_estimator)
        self.actor = copy.deepcopy(policy.actor)
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.state_dependent_std = policy.state_dependent_std

    def forward(
        self,
        proprioception: torch.Tensor,
        observation_history: torch.Tensor,
        commands: torch.Tensor,
    ) -> torch.Tensor:
        """由三个明确输入生成动作，拼接顺序必须与训练时 ``get_actor_obs`` 一致。"""
        # ① 展平历史：[N, H, D] → [N, H*D]，然后 Encoder 估计速度 [N, 3]
        estimated_velocity = self.estimator(observation_history.flatten(start_dim=-2))
        # ② 按训练时的顺序拼接：[估计速度(3) | 当前本体感知(28) | 指令(3)] → [N, 34]
        actor_input = torch.cat((estimated_velocity, proprioception, commands), dim=-1)
        # ③ 归一化后送入 Actor MLP，输出动作 [N, num_actions]
        actor_output = self.actor(self.normalizer(actor_input))
        # ④ state-dependent std 模式：输出含均值 + 标准差，部署只取均值
        if self.state_dependent_std:
            return actor_output[..., 0, :]
        return actor_output

    @torch.jit.export
    def estimate_velocity(self, observation_history: torch.Tensor) -> torch.Tensor:
        """在组合 JIT 模型上额外暴露速度估计接口，便于部署侧诊断。"""
        return self.estimator(observation_history.flatten(start_dim=-2))


'''
纯 Encoder，不含 Actor 和归一化，方便离线评估速度估计误差。
'''
class _VelocityEstimatorOnly(nn.Module):
    """仅包含历史 Encoder 的诊断模型，不执行归一化或 Actor。"""

    def __init__(self, policy: VelocityEstimatorActorCritic) -> None:
        super().__init__()
        self.estimator = copy.deepcopy(policy.velocity_estimator)

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        """将 ``[N,H,D]`` 历史展平后输出 ``[N,3]`` 机体系线速度。"""
        return self.estimator(observation_history.flatten(start_dim=-2))


'''
作用：将训练好的模型导出为真机可用的 JIT/ONNX 格式，同时生成 manifest.json 记录输入输出契约。
导出产物：
    导出目录/
    ├── policy.pt                    # JIT: Encoder + Actor 组合（可选）
    ├── policy.onnx                  # ONNX: Encoder + Actor 组合（可选）
    ├── velocity_estimator.pt        # JIT: 纯 Encoder，用于离线诊断（可选）
    ├── velocity_estimator.onnx      # ONNX: 纯 Encoder（可选）
    └── policy_manifest.json         # 输入输出 shape、拼接顺序、文件哈希
'''
def export_velocity_estimator_policy(
    policy: VelocityEstimatorActorCritic,
    path: str,
    export_jit: bool = True,
    export_onnx: bool = True,
) -> None:
    """导出完整策略、独立 Encoder 以及描述两者契约的 manifest。

    JIT 与 ONNX 可按需启用。ONNX 只将 batch 维声明为动态，历史长度和单帧宽度保持固定，
    这样部署端若误传帧数或观测布局会立即报错。manifest 记录每个实际生成产物的哈希，
    并且最后才发布，因此消费者观察到新 manifest 时，其引用的模型均已就位。
    """
    # ① 创建输出目录
    os.makedirs(path, exist_ok=True)
    # ② 构建两个导出用模型（deepcopy 避免影响训练中的模块）
    combined = _CombinedVelocityEstimatorPolicy(policy).cpu().eval()
    estimator = _VelocityEstimatorOnly(policy).cpu().eval()
    # ③ 构建 manifest：记录输入/输出 shape、拼接顺序、部署元数据
    manifest = {
        "format_version": 1,
        "policy_type": "history_velocity_estimator_actor",
        "inputs": {
            "proprioception": ["batch", policy.proprio_dim],
            "observation_history": ["batch", policy.history_length, policy.proprio_dim],
            "commands": ["batch", policy.command_dim],
        },
        "outputs": {"actions": ["batch", policy.num_actions]},
        "history_order": "oldest_to_newest",
        "actor_input_order": ["estimated_base_linear_velocity", "proprioception", "commands"],
        "estimator_output": {
            "name": "estimated_base_linear_velocity",
            "shape": ["batch", policy.estimator_output_dim],
            "frame": "robot_base",
            "units": "m/s",
        },
        "deployment": policy.deployment_metadata,
    }
    artifact_names: list[str] = []
    # ④ 在 staging 临时目录完成所有导出操作（失败不影响正式产物）
    with tempfile.TemporaryDirectory(dir=path, prefix=".l5a-export-") as staging_dir:
        if export_jit:
            # ⑤ 导出完整策略 JIT + 独立 Encoder JIT
            torch.jit.script(combined).save(os.path.join(staging_dir, "policy.pt"))
            torch.jit.script(estimator).save(os.path.join(staging_dir, "velocity_estimator.pt"))
            artifact_names.extend(("policy.pt", "velocity_estimator.pt"))

        if export_onnx:
            # ⑥ 构造样例张量：batch=1，其余维度与训练一致
            proprioception = torch.zeros(1, policy.proprio_dim)
            observation_history = torch.zeros(1, policy.history_length, policy.proprio_dim)
            commands = torch.zeros(1, policy.command_dim)
            # ⑦ 导出 ONNX：只把 batch 维设为动态
            torch.onnx.export(
                combined,
                (proprioception, observation_history, commands),
                os.path.join(staging_dir, "policy.onnx"),
                export_params=True,
                opset_version=18,
                input_names=["proprioception", "observation_history", "commands"],
                output_names=["actions"],
                dynamic_axes={
                    "proprioception": {0: "batch"},
                    "observation_history": {0: "batch"},
                    "commands": {0: "batch"},
                    "actions": {0: "batch"},
                },
                dynamo=False,
            )
            torch.onnx.export(
                estimator,
                observation_history,
                os.path.join(staging_dir, "velocity_estimator.onnx"),
                export_params=True,
                opset_version=18,
                input_names=["observation_history"],
                output_names=["estimated_base_lin_vel"],
                dynamic_axes={
                    "observation_history": {0: "batch"},
                    "estimated_base_lin_vel": {0: "batch"},
                },
                dynamo=False,
            )
            artifact_names.extend(("policy.onnx", "velocity_estimator.onnx"))

        # ⑧ 计算所有产物的 SHA-256 哈希，写入 manifest
        manifest["artifacts"] = {
            name: {"sha256": _sha256_file(os.path.join(staging_dir, name))} for name in artifact_names
        }
        staged_manifest = os.path.join(staging_dir, "policy_manifest.json")
        with open(staged_manifest, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)

        # ⑨ 原子发布：先移动模型文件，最后移动 manifest（os.replace 是原子操作）
        for name in artifact_names:
            os.replace(os.path.join(staging_dir, name), os.path.join(path, name))
        os.replace(staged_manifest, os.path.join(path, "policy_manifest.json"))
