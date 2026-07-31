# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""为基座线速度 Encoder 增加独立监督更新的 PPO 扩展。

一次更新包含两条互不混合的梯度路径：

1. 上游 RSL-RL PPO 使用 rollout 更新 Actor/Critic；
2. 本模块重新遍历同一份 rollout，以 ``MSE(估计线速度, 仿真真值)`` 更新 Encoder。

Actor 前向时使用的是已经 ``detach`` 的速度估计，因此 PPO 不会更新 Encoder；同时
Encoder 参数会从 PPO optimizer 中剔除。两套 optimizer 的参数集合严格分离，可以分别
保存 Adam 动量、设置学习率与梯度裁剪阈值。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from .estimator_actor_critic import VelocityEstimatorActorCritic


class _EstimatorRolloutStorage(RolloutStorage):
    """让同一份 rollout 同时服务于 PPO 更新和 Encoder 监督更新。

    RSL-RL 的 ``PPO.update()`` 结束时会调用 ``storage.clear()``。但 Encoder 随后还需要
    TensorDict 中的 ``obs_history`` 与 ``base_lin_vel_target``。``defer_clear`` 为真时先
    拦截该清理动作，待两阶段更新都结束后再由本模块统一清空。
    """

    defer_clear: bool = False

    def clear(self) -> None:
        """仅在未处于两阶段更新窗口时真正清空 rollout。"""
        if not self.defer_clear:
            super().clear()


class VelocityEstimatorPPO(PPO):
    """先执行标准 RSL-RL PPO，再用监督 MSE 训练历史速度 Encoder。

    该类不修改 PPO 的 clipped surrogate、value loss、GAE 等主算法，只负责参数所有权
    分离、rollout 生命周期延长，以及附加的 Encoder mini-batch 更新。
    """

    policy: VelocityEstimatorActorCritic

    '''
    作用：建立两套完全独立的 optimizer（一个管 Actor+Critic，一个管 Encoder）。
    '''
    def __init__(
        self,
        policy: VelocityEstimatorActorCritic,
        estimator_learning_rate: float = 1.0e-3,
        estimator_loss_coef: float = 1.0,
        estimator_max_grad_norm: float = 0.1,
        estimator_num_learning_epochs: int | None = None,
        **kwargs,
    ) -> None:
        """构造相互独立的 PPO optimizer 与 Encoder optimizer。

        ``estimator_num_learning_epochs=None`` 表示 Encoder 与 PPO 使用相同的学习 epoch
        数；否则可单独控制 Encoder 对每份 rollout 的重复学习次数。
        """
        super().__init__(policy, **kwargs)
        # ① 类型检查：policy 必须是 VelocityEstimatorActorCritic
        if not isinstance(policy, VelocityEstimatorActorCritic):
            raise TypeError("VelocityEstimatorPPO requires VelocityEstimatorActorCritic.")
        # ② 关键检查：detach 必须为 True，否则两套 optimizer 会互相干扰
        if not policy.estimator_output_detach:
            raise ValueError(
                "VelocityEstimatorPPO requires estimator_output_detach=True because "
                "PPO and the estimator use independent optimizers."
            )
        # ③ 超参数合法性校验
        if estimator_learning_rate <= 0.0:
            raise ValueError("estimator_learning_rate must be positive.")
        if estimator_loss_coef < 0.0:
            raise ValueError("estimator_loss_coef must be non-negative.")
        if estimator_max_grad_norm <= 0.0:
            raise ValueError("estimator_max_grad_norm must be positive.")
        if estimator_num_learning_epochs is not None and estimator_num_learning_epochs <= 0:
            raise ValueError("estimator_num_learning_epochs must be positive when provided.")

        # ④ 参数所有权分离：找出 Encoder 参数 id，剩余的归 PPO optimizer
        estimator_parameter_ids = {id(parameter) for parameter in policy.velocity_estimator.parameters()}
        ppo_parameters = [
            parameter for parameter in policy.parameters() if id(parameter) not in estimator_parameter_ids
        ]
        # ⑤ 重建 PPO optimizer（只包含 Actor + Critic 参数）
        self.optimizer = optim.Adam(ppo_parameters, lr=self.learning_rate)
        # ⑥ 建立独立的 Encoder optimizer
        self.estimator_optimizer = optim.Adam(
            policy.velocity_estimator.parameters(),
            lr=estimator_learning_rate,
        )
        self.estimator_loss_coef = estimator_loss_coef
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.estimator_num_learning_epochs = estimator_num_learning_epochs

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        """建立支持延迟清理的 rollout storage。

        ``obs`` 是完整 TensorDict，所以每个 transition 除 PPO 所需分组外，也会保留历史
        观测和显式监督标签，供第二阶段 Encoder 更新使用。
        """
        self.storage = _EstimatorRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    '''
    作用：依次执行 PPO 更新（Actor+Critic）和监督 MSE 更新（Encoder），两次更新共享同一份 rollout 数据。
    '''
    def update(self) -> dict[str, float]:
        """依次更新 Actor/Critic 与 Encoder，并在最后准确释放 rollout。

        返回值沿用 RSL-RL 的 loss 字典，并额外加入 ``base_lin_vel_estimator``，便于在
        TensorBoard/W&B 中直接观察速度估计误差是否收敛。
        """
        # ① 清理 Encoder 上次迭代的残留梯度（Encoder 参数不在 PPO optimizer 中）
        self.estimator_optimizer.zero_grad(set_to_none=True)
        if self.storage is None:
            raise RuntimeError("Rollout storage must be initialized before update().")
        # ② 延迟清理：让同一份 rollout 同时服务于 PPO 更新 + Encoder 监督更新
        self.storage.defer_clear = True
        try:
            # ③ 第一阶段：标准 PPO 更新 Actor + Critic（内部会调用 self.storage.clear()，被拦截）
            loss_dict = super().update()
            # ④ 确定 Encoder 的学习 epoch 数
            num_epochs = (
                self.num_learning_epochs
                if self.estimator_num_learning_epochs is None
                else self.estimator_num_learning_epochs
            )
            # ⑤ 第二阶段：遍历同一份 rollout 的 minibatch，用 MSE 训练 Encoder
            generator = self.storage.mini_batch_generator(self.num_mini_batches, num_epochs)
            mean_estimator_loss = 0.0
            num_updates = 0

            for batch in generator:
                # ⑥ 从 TensorDict 中分别读取历史和监督标签（不依赖 Critic 向量的列位置）
                obs_batch = batch[0]
                estimate = self.policy.estimate_base_lin_vel(obs_batch)       # Encoder 前向（保留梯度）
                target = self.policy.get_estimator_target(obs_batch).detach()  # 真值线速度（不需要梯度）
                # ⑦ MSE 损失：均方误差 (est_vx - true_vx)² + (est_vy - true_vy)² + (est_vz - true_vz)²
                estimator_loss = torch.mean(torch.square(estimate - target))

                self.estimator_optimizer.zero_grad(set_to_none=True)
                (self.estimator_loss_coef * estimator_loss).backward()
                if self.is_multi_gpu:
                    self._reduce_estimator_gradients()
                # ⑧ Encoder 独立的梯度裁剪
                nn.utils.clip_grad_norm_(
                    self.policy.velocity_estimator.parameters(),
                    self.estimator_max_grad_norm,
                )
                self.estimator_optimizer.step()

                mean_estimator_loss += estimator_loss.item()
                num_updates += 1

            # ⑨ 将 Encoder 损失写入日志字典
            loss_dict["base_lin_vel_estimator"] = mean_estimator_loss / max(num_updates, 1)
            return loss_dict
        finally:
            # ⑩ 无论成功/失败，恢复 clear 行为并释放本轮数据
            self.storage.defer_clear = False
            self.storage.clear()

    def _reduce_estimator_gradients(self) -> None:
        """在多 GPU worker 间求和后取平均，保持 Encoder 梯度尺度不随卡数变化。"""
        for parameter in self.policy.velocity_estimator.parameters():
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
            parameter.grad /= self.gpu_world_size
