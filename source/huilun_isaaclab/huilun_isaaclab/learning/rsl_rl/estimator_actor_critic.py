# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""带历史速度估计器的 Actor-Critic。

本模块是环境观测与 RSL-RL 网络之间的适配层。环境不再把所有观测压成一个向量，
而是通过 :class:`~tensordict.TensorDict` 提供具有明确用途的分组：

* ``policy``：当前时刻的本体感知观测，形状为 ``[N, D]``；
* ``obs_history``：最近 H 帧本体感知观测，形状为 ``[N, H, D]``；
* ``commands``：速度指令，形状为 ``[N, C]``；
* ``base_lin_vel_target``：仿真器给出的无噪声监督标签，形状为 ``[N, 3]``；
* ``critic``：只交给 Critic 的特权观测集合，由 RSL-RL 按配置拼接。

默认 L5A WF 配置中 ``H=10``、``D=28``，因此 Encoder 的数据流为
``10×28 -> 280 -> 256 -> 128 -> 3``。最后的三维输出表示基座坐标系中的线速度，
并与当前本体感知和指令拼接后交给 Actor。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from tensordict import TensorDict


class VelocityEstimatorActorCritic(ActorCritic):
    """带独立监督 Encoder 的非对称 Actor-Critic。

    Actor 只能看到可在真机复现的信息：历史观测估计的线速度、当前本体感知和指令。
    Critic 则仍按 ``obs_groups["critic"]`` 使用训练期特权信息。速度真值只作为 Encoder
    的监督标签，绝不会拼入 Actor 输入，从而避免在仿真训练时产生不可部署的信息泄漏。
    """

    is_velocity_estimator_policy: bool = True

    """
    作用：根据环境首批观测初始化所有网络层。

    输入参数（只列关键的自定义参数）：
        obs: TensorDict — 环境 reset 后产出的第一批观测，从真实张量推导维度，不硬编码
        obs_groups: dict — 告诉 RSL-RL 父类「policy 组包含哪些 key、critic 组包含哪些 key」
        num_actions: int — 动作空间维度（L5A 是 8）
        proprio_group / history_group / command_group / estimator_target_group — 各组在 TensorDict 中的 key 名
        estimator_output_dim: int — Encoder 输出维度（默认 3）
        estimator_hidden_dims: list — Encoder 隐藏层宽度（默认 [256, 128]）
        estimator_output_detach: bool — 是否切断 Encoder→Actor 梯度（必须 True）
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        proprio_group: str = "policy",
        history_group: str = "obs_history",
        command_group: str = "commands",
        estimator_target_group: str = "base_lin_vel_target",
        estimator_output_dim: int = 3,
        estimator_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        estimator_activation: str = "elu",
        estimator_output_detach: bool = True,
        estimator_orthogonal_init: bool = False,
        deployment_metadata: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """根据首批环境观测确定各输入维度并构造网络。

        ``obs`` 是环境 reset 后产生的批量 TensorDict。维度从真实张量推导，而不是在
        网络中重复硬编码，因此调整观测宽度时会在契约校验处尽早失败。默认配置构造的
        Encoder 为 ``280 -> 256 -> 128 -> 3``，Actor 输入宽度为
        ``3 + 28 + 3 = 34``。
        """
        # ① 参数合法性校验
        if estimator_output_dim <= 0:
            raise ValueError("estimator_output_dim must be positive.")
        if not estimator_hidden_dims or any(width <= 0 for width in estimator_hidden_dims):
            raise ValueError("estimator_hidden_dims must contain only positive widths.")
        # ② 观测契约校验：检查 TensorDict 中各组的 shape 和内容是否与配置一致
        self._validate_observation_contract(
            obs,
            obs_groups,
            proprio_group,
            history_group,
            command_group,
            estimator_target_group,
            estimator_output_dim,
        )

        # ③ 缓存各组名称和维度，后续 forward 时不需要重复查字典
        self.proprio_group = proprio_group
        self.history_group = history_group
        self.command_group = command_group
        self.estimator_target_group = estimator_target_group
        self.estimator_output_dim = estimator_output_dim
        self.estimator_output_detach = estimator_output_detach
        self.deployment_metadata = dict(deployment_metadata or {})
        self.num_actions = num_actions
        self.history_length = obs[history_group].shape[-2]
        self.proprio_dim = obs[proprio_group].shape[-1]
        self.command_dim = obs[command_group].shape[-1]
        self._validate_deployment_metadata_contract()

        # ④ 构造"合成观测组"，欺骗父类 ActorCritic 让它按正确宽度建 Actor MLP
        #    真正的 Actor 输入 = [encoder估计速度, 当前本体感知, 指令] = 3 + 28 + 3 = 34 维
        #    这里先创建一个占位 group，父类据此推导 MLP 输入层宽度
        synthetic_group = "__estimated_base_lin_vel"
        augmented_obs = obs.clone()
        augmented_obs[synthetic_group] = obs[proprio_group].new_zeros(
            *obs[proprio_group].shape[:-1],
            estimator_output_dim,
        )
        augmented_obs_groups = {name: list(groups) for name, groups in obs_groups.items()}
        augmented_obs_groups["policy"] = [synthetic_group, proprio_group, command_group]
        super().__init__(augmented_obs, augmented_obs_groups, num_actions, **kwargs)

        # ⑤ 建立速度估计器 MLP：10帧×28维 = 280 → 256 → 128 → 3
        self.velocity_estimator = MLP(
            self.history_length * self.proprio_dim,
            estimator_output_dim,
            list(estimator_hidden_dims),
            estimator_activation,
        )
        # ⑥ 可选：正交初始化 Encoder 权重
        if estimator_orthogonal_init:
            linear_layers = [module for module in self.velocity_estimator if isinstance(module, nn.Linear)]
            for layer in linear_layers[:-1]:
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(linear_layers[-1].weight, gain=0.01)
            nn.init.zeros_(linear_layers[-1].bias)
        print(f"Velocity estimator MLP: {self.velocity_estimator}")

    """
    作用：在网络构建前验证环境与网络之间的数据契约，避免维度不匹配在训练中途才暴露。

    类比：快递员送货前先检查包裹标签是否正确——收件人、地址、重量都核对一遍，不对就当场拒收，而不是等送到了才发现送错了。
    """

    @staticmethod
    def _validate_observation_contract(
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        proprio_group: str,
        history_group: str,
        command_group: str,
        estimator_target_group: str,
        estimator_output_dim: int,
    ) -> None:
        """在建网前验证环境和网络之间的 TensorDict 契约。

        这些检查不仅验证分组是否存在，还确保历史单帧宽度等于当前本体感知宽度。
        如果环境侧新增、删除或错误拼接观测项，训练会在初始化阶段报告清晰的维度错误，
        而不是等到矩阵乘法或导出阶段才失败。
        """
        # ① 检查所有必需分组是否存在
        required = (proprio_group, history_group, command_group, estimator_target_group)
        missing = [name for name in required if name not in obs.keys()]
        if missing:
            raise KeyError(f"Velocity estimator observation groups are missing: {missing}")
        # ② 检查 policy 组 shape：[N, D]，D > 0
        if len(obs[proprio_group].shape) != 2:
            raise ValueError(f"'{proprio_group}' must have shape [N, D], got {tuple(obs[proprio_group].shape)}")
        if obs[proprio_group].shape[-1] <= 0:
            raise ValueError(f"'{proprio_group}' must contain at least one feature.")
        # ③ 检查 history 组 shape：[N, H, D]，且单帧宽度 == policy 宽度
        if len(obs[history_group].shape) != 3:
            raise ValueError(f"'{history_group}' must have shape [N, H, D], got {tuple(obs[history_group].shape)}")
        if obs[history_group].shape[-2] <= 0:
            raise ValueError(f"'{history_group}' must contain at least one history frame.")
        if obs[history_group].shape[-1] != obs[proprio_group].shape[-1]:
            raise ValueError("History frame width must equal the single-frame proprioception width.")
        # ④ 检查 commands 组 shape：[N, C]，C > 0
        if len(obs[command_group].shape) != 2:
            raise ValueError(f"'{command_group}' must have shape [N, C], got {tuple(obs[command_group].shape)}")
        if obs[command_group].shape[-1] <= 0:
            raise ValueError(f"'{command_group}' must contain at least one command.")
        # ⑤ 检查监督标签 shape：[N, E]，且 E == Encoder 输出维度
        if len(obs[estimator_target_group].shape) != 2:
            raise ValueError(
                f"'{estimator_target_group}' must have shape [N, E], got {tuple(obs[estimator_target_group].shape)}"
            )
        if obs[estimator_target_group].shape[-1] != estimator_output_dim:
            raise ValueError(
                f"Estimator target width {obs[estimator_target_group].shape[-1]} "
                f"does not match estimator output width {estimator_output_dim}."
            )
        # ⑥ 检查 obs_groups 中 policy 和 critic 分组是否存在
        for set_name in ("policy", "critic"):
            if set_name not in obs_groups:
                raise KeyError(f"obs_groups must explicitly define the '{set_name}' observation set.")
        # ⑦ 检查 policy 分组只声明 [本体感知, 指令]（禁止泄漏历史或真值给 Actor）
        expected_policy_groups = [proprio_group, command_group]
        if obs_groups["policy"] != expected_policy_groups:
            raise ValueError(f"obs_groups['policy'] must be {expected_policy_groups}; got {obs_groups['policy']}.")

    def _validate_deployment_metadata_contract(self) -> None:
        """用真实 TensorDict 维度校验 checkpoint/export 使用的部署契约。"""
        metadata = self.deployment_metadata
        if not metadata:
            raise ValueError("Velocity estimator deployment_metadata must not be empty.")

        required_keys = (
            "schema_version",
            "control_period_s",
            "physics_period_s",
            "decimation",
            "history_samples",
            "history_duration_s",
            "proprioception_dim",
            "command_dim",
            "action_dim",
            "proprioception_layout",
            "command_order",
            "policy_action_order",
            "policy_actions_to_hardware_indices",
            "hardware_state_to_policy_indices",
        )
        missing = [key for key in required_keys if key not in metadata]
        if missing:
            raise KeyError(f"Velocity estimator deployment_metadata is missing required keys: {missing}")

        if int(metadata["history_samples"]) != self.history_length:
            raise ValueError(
                f"deployment_metadata.history_samples={metadata['history_samples']} does not match "
                f"obs['{self.history_group}'].shape[-2]={self.history_length}."
            )
        if int(metadata["proprioception_dim"]) != self.proprio_dim:
            raise ValueError(
                f"deployment_metadata.proprioception_dim={metadata['proprioception_dim']} does not match "
                f"obs['{self.proprio_group}'].shape[-1]={self.proprio_dim}."
            )
        if int(metadata["command_dim"]) != self.command_dim:
            raise ValueError(
                f"deployment_metadata.command_dim={metadata['command_dim']} does not match "
                f"obs['{self.command_group}'].shape[-1]={self.command_dim}."
            )
        if int(metadata["action_dim"]) != self.num_actions:
            raise ValueError(
                f"deployment_metadata.action_dim={metadata['action_dim']} does not match "
                f"num_actions={self.num_actions}."
            )

        layout = metadata["proprioception_layout"]
        if not isinstance(layout, list):
            raise TypeError("deployment_metadata.proprioception_layout must be a list.")
        layout_dim = sum(int(item["size"]) for item in layout)
        if layout_dim != self.proprio_dim:
            raise ValueError(
                f"deployment_metadata.proprioception_layout sums to {layout_dim}, "
                f"but proprioception_dim is {self.proprio_dim}."
            )
        if len(metadata["command_order"]) != self.command_dim:
            raise ValueError("deployment_metadata.command_order length does not match command_dim.")
        if len(metadata["policy_action_order"]) != self.num_actions:
            raise ValueError("deployment_metadata.policy_action_order length does not match num_actions.")

        expected_indices = list(range(self.num_actions))
        for key in ("policy_actions_to_hardware_indices", "hardware_state_to_policy_indices"):
            if sorted(int(index) for index in metadata[key]) != expected_indices:
                raise ValueError(f"deployment_metadata.{key} must be a permutation of {expected_indices}.")

        expected_history_duration = float(metadata["control_period_s"]) * self.history_length
        if not math.isclose(
            float(metadata["history_duration_s"]),
            expected_history_duration,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "deployment_metadata.history_duration_s does not match control_period_s * obs_history length."
            )
        expected_control_period = float(metadata["physics_period_s"]) * int(metadata["decimation"])
        if not math.isclose(float(metadata["control_period_s"]), expected_control_period, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("deployment_metadata.control_period_s does not match physics_period_s * decimation.")

    """
    作用：从 [N, H, D] 历史观测估计 [N, 3] 机体系线速度。这是 Encoder 的唯一前向接口。

    输入参数：
        obs: TensorDict — 包含 history_group key 的观测字典
        detach: bool — True 时切断梯度（Actor 用），False 时保留（Encoder 训练用）
    输出：
        torch.Tensor [N, 3]，基座坐标系下估计的线速度 (vx, vy, vz)

        历史 [N, 10, 28]
        → flatten(start_dim=-2) → [N, 280]
        → velocity_estimator MLP → [N, 3]
        → detach? 切梯度 : 保留梯度
    """

    def estimate_base_lin_vel(self, obs: TensorDict, detach: bool = False) -> torch.Tensor:
        """从 ``[N, H, D]`` 历史观测估计 ``[N, 3]`` 机体系基座线速度。

        ``detach=True`` 只切断"Actor/PPO 损失 -> Encoder"的梯度路径；Encoder 自身仍会在
        :class:`VelocityEstimatorPPO` 的监督更新中以 ``detach=False`` 正常反向传播。
        """
        # ① 取出历史：shape [N, H, D]，例如 [4096, 10, 28]
        history = obs[self.history_group]
        # ② 展平最后两维：[N, H, D] → [N, H*D]，例如 [4096, 280]
        flattened_history = history.flatten(start_dim=-2)
        # ③ MLP 前向：280 → 256 → 128 → 3（基座系 vx, vy, vz）
        estimate = self.velocity_estimator(flattened_history)
        # ④ detach=True 时切断梯度（Actor 前向用），False 时保留（Encoder 训练用）
        return estimate.detach() if detach else estimate

    def get_estimator_target(self, obs: TensorDict) -> torch.Tensor:
        """返回显式、无观测噪声的 ``[N, 3]`` 速度监督标签。

        标签拥有独立 TensorDict key，不依赖 Critic 向量中的位置索引，因此改变特权观测
        排列不会悄悄改变 Encoder 的学习目标。
        """
        return obs[self.estimator_target_group]

    """
    作用：按部署顺序拼接 Actor 输入：
        [估计速度(3) | 本体感知(28) | 指令(3)] → [N, 34]。
        这个拼接顺序会被写入 manifest，真机部署必须保持一致。

    输入：obs: TensorDict — 完整观测字典

    输出：torch.Tensor [N, 34] — Actor MLP 的输入向量

        ① Encoder 前向（detach=True）：[N,10,28] → [N,3]
        ② 拼接：cat([估计速度, 本体感知, 指令], dim=-1) → [N,34]
        ③ 送入 Actor MLP（在父类 RSL-RL 的 update 中自动完成）
    """

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """按部署顺序拼接 Actor 输入：``[估计速度, 本体感知, 指令]``。

        默认 ``estimator_output_detach=True``，使 PPO 只更新 Actor/Critic，而 Encoder 只由
        速度 MSE 更新。该顺序同时写入部署 manifest，真机侧必须保持一致。
        """
        # ① Encoder 根据 10 帧历史估计基座线速度 [N, 3]
        estimate = self.estimate_base_lin_vel(obs, detach=self.estimator_output_detach)
        # ② 拼接：[估计速度(3) | 当前本体感知(28) | 指令(3)] → [N, 34]
        return torch.cat((estimate, obs[self.proprio_group], obs[self.command_group]), dim=-1)
