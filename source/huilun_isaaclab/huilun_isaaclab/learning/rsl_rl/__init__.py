# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 历史速度估计训练链路的公共入口。

外部任务配置只需从本包导入配置类；训练脚本使用自定义 Runner，Runner 再显式构造
Actor-Critic 与 PPO。部署脚本通过 ``export_velocity_estimator_policy`` 输出独立 Actor、
独立 Encoder 和 manifest。这里集中重导出公共接口，内部辅助包装不会泄漏给调用方。
"""

from .estimator_actor_critic import VelocityEstimatorActorCritic
from .estimator_exporter import export_velocity_estimator_policy
from .estimator_ppo import VelocityEstimatorPPO
from .estimator_runner import VelocityEstimatorOnPolicyRunner
from .velocity_estimator_cfg import VelocityEstimatorActorCriticCfg, VelocityEstimatorPPOCfg

__all__ = [
    # 网络与其 IsaacLab/RSL-RL 配置结构。
    "VelocityEstimatorActorCritic",
    "VelocityEstimatorActorCriticCfg",
    # PPO 两阶段更新与对应配置结构。
    "VelocityEstimatorPPO",
    "VelocityEstimatorPPOCfg",
    # 负责显式构造、checkpoint 保存/恢复的训练入口。
    "VelocityEstimatorOnPolicyRunner",
    # 负责 JIT、ONNX 与 manifest 一致发布的部署入口。
    "export_velocity_estimator_policy",
]


'''
整体代码框架
    rsl_rl/
    ├── __init__.py                       # 统一导出，外部只用 import 这一个文件
    ├── velocity_estimator_cfg.py         # ① 配置文件：声明网络结构和训练超参数
    ├── estimator_actor_critic.py         # ② 网络模型：Actor/Critic/Encoder 三层组装
    ├── estimator_ppo.py                  # ③ 训练算法：PPO + 监督 MSE 双阶段更新
    ├── estimator_runner.py               # ④ 训练入口：构造 + 保存 + 恢复 checkpoint
    └── estimator_exporter.py             # ⑤ 部署导出：JIT/ONNX + manifest
调用关系：
    velocity_estimator_cfg.py  (纯配置，无逻辑)
            ↓ 被读入
    estimator_runner.py  (训练入口，显式构造网络和算法)
        ├──→ estimator_actor_critic.py  (网络定义)
        └──→ estimator_ppo.py          (算法定义)
                ↓ 训练后
    estimator_exporter.py  (导出为真机可用的 JIT/ONNX)
核心数据流
    以 L5A WF 默认配置为例，一条数据从头到尾的流向：
        环境 reset，产出 TensorDict:
        ┌─────────────────────────────────────────────────────┐
        │ "policy" (本体感知):         [N, 28]               │  ← 当前时刻的关节/IMU观测
        │ "obs_history" (10帧历史):    [N, 10, 28]           │  ← 最近10帧本体感知，oldest→newest
        │ "commands" (速度指令):       [N, 3]                │  ← 期望的 vx, vy, ωz
        │ "base_lin_vel_target" (真值): [N, 3]               │  ← 仿真器真值，仅训练用
        │ "critic" (特权观测):         [N, ...]              │  ← 仅 Critic 可见
        └─────────────────────────────────────────────────────┘

                            ↓

        ┌────── Encoder 前向（训练/推理共用） ──────┐
        │ [N, 10, 28] → flatten → [N, 280]          │
        │     → Linear(280→256) → ELU               │
        │     → Linear(256→128) → ELU               │
        │     → Linear(128→3)                       │
        │ 输出: estimated_vel [N, 3]                 │
        └────────────────────────────────────────────┘
                │                              │
                │ detach=True (Actor用)        │ detach=False (Encoder训练用)
                ↓                              ↓
        ┌─── Actor 输入拼接 ───┐         ┌─── Encoder 监督损失 ───┐
        │ [est_vel(3) |         │         │ MSE(est_vel, true_vel) │
        │  proprio(28) |        │         │ → 只更新 Encoder       │
        │  commands(3)]         │         └────────────────────────┘
        │ = [N, 34]             │
        │     ↓                 │
        │  Actor MLP → 动作[N,8]│
        │     ↓                 │
        │  PPO Loss（Actor+Critic）
        │  只更新 Actor/Critic  │
        └───────────────────────┘
    关键设计：
        Encoder 输出在传入 Actor 时被 detach（切断梯度）。
        PPO 的损失回传不到 Encoder，Encoder 的 MSE 损失也回传不到 Actor。
        两套参数、两个 optimizer、互不干扰。
'''
