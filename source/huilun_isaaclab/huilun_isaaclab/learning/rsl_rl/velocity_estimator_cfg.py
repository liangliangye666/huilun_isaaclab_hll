# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 历史速度估计策略的 RSL-RL 配置结构。

这里仅声明“如何连接环境观测与学习模块”，实际维度仍由首批 TensorDict 验证。配置类会
通过 IsaacLab 的 ``configclass`` 进入任务 runner 配置，最终分别传给 Actor-Critic 和 PPO。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class VelocityEstimatorActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-Critic 与监督历史 Encoder 的网络及观测分组配置。

    默认 L5A WF 数据流为 ``[N,10,28] -> [N,280] -> 256 -> 128 -> 3``。
    这些 group 名必须与环境 ObservationCfg 和 runner 的 ``obs_groups`` 保持一致。
    """

    class_name: str = "VelocityEstimatorActorCritic"
    # 当前单帧本体感知；供 Actor 使用，同时也是 history 中每一帧的特征定义。    告诉网络，TensorDict 里哪个 key 是当前帧本体感知
    proprio_group: str = "policy"
    # 按 oldest_to_newest 排列的本体感知历史，默认 10 帧。                  哪个 key 是 10 帧历史
    history_group: str = "obs_history"
    # 期望机体速度指令，不属于 Encoder 输入，而是在 Encoder 输出后直接交给 Actor。  哪个 key 是指令
    command_group: str = "commands"
    # 仿真真值监督标签；仅用于 Encoder MSE，不允许进入部署 Actor 的观测集合。   哪个 key 是速度真值标签
    estimator_target_group: str = "base_lin_vel_target"
    # 三维输出依次对应机体系线速度分量，单位为 m/s。                          Encoder 输出 3 维（vx, vy, vz）
    estimator_output_dim: int = 3
    # 与展平历史输入共同构成默认 280 -> 256 -> 128 -> 3 MLP。               Encoder 隐藏层宽度
    estimator_hidden_dims: list[int] = [256, 128]
    estimator_activation: str = "elu"
    # 必须保持为 True：切断 PPO 经 Actor 回传到 Encoder 的梯度，维持两套优化目标独立。  必须为 True，切断 PPO→Encoder 的梯度
    estimator_output_detach: bool = True
    estimator_orthogonal_init: bool = False
    # 非网络参数的部署契约，如观测顺序、缩放、关节映射；随 checkpoint 与 manifest 保存。
    deployment_metadata: dict = {}


@configclass
class VelocityEstimatorPPOCfg(RslRlPpoAlgorithmCfg):    # 声明 Encoder 训练超参数
    """标准 PPO 之外的独立速度估计监督目标配置。"""

    class_name: str = "VelocityEstimatorPPO"
    # 仅供 Encoder Adam 使用，不影响 PPO 自适应学习率。                 Encoder 专用学习率（独立于 PPO 的 adaptive LR）
    estimator_learning_rate: float = 1.0e-3
    # 乘在 MSE 上的反向传播系数；日志仍记录未乘系数的原始 MSE。             MSE 损失的缩放系数
    estimator_loss_coef: float = 1.0
    # Encoder 自己的梯度裁剪阈值，与 PPO 的 max_grad_norm 相互独立。    Encoder 独立梯度裁剪阈值
    estimator_max_grad_norm: float = 0.1
    # None 表示沿用 PPO num_learning_epochs，否则覆盖为 Encoder 专用 epoch 数。     None 表示和 PPO 用同样多的 epoch
    estimator_num_learning_epochs: int | None = None
