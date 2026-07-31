# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A 速度估计策略在 RSL-RL 3.x 中的构造与 checkpoint 生命周期。

标准 :class:`rsl_rl.runners.OnPolicyRunner` 负责环境交互、rollout 收集和日志；本模块只
替换两个边界：显式构造项目内的 ``VelocityEstimatorActorCritic/VelocityEstimatorPPO``，
以及在 checkpoint 中补齐 Encoder optimizer 和部署元数据。这样训练恢复与部署导出都
共享同一份观测、动作和硬件映射契约。
"""

from __future__ import annotations

import math
import warnings
from numbers import Number

import torch
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from .estimator_actor_critic import VelocityEstimatorActorCritic
from .estimator_ppo import VelocityEstimatorPPO


def _metadata_values_equal(current: object, checkpoint: object) -> bool:
    """Compare metadata scalars with tolerance for generated floating point seconds."""
    if isinstance(current, bool) or isinstance(checkpoint, bool):
        return current == checkpoint
    if isinstance(current, Number) and isinstance(checkpoint, Number):
        return math.isclose(float(current), float(checkpoint), rel_tol=1e-9, abs_tol=1e-12)
    return current == checkpoint


def _short_metadata_value(value: object) -> str:
    """Keep metadata diff messages readable when a value is a long list or dict."""
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def _collect_metadata_diffs(current: object, checkpoint: object, path: str = "deployment_metadata") -> list[str]:
    """Return path-level differences between current config metadata and checkpoint metadata."""
    if isinstance(current, dict) and isinstance(checkpoint, dict):
        diffs: list[str] = []
        for key in sorted(set(current) | set(checkpoint)):
            child_path = f"{path}.{key}"
            if key not in current:
                diffs.append(f"{child_path}: checkpoint has extra value {_short_metadata_value(checkpoint[key])}")
            elif key not in checkpoint:
                diffs.append(f"{child_path}: missing in checkpoint; current={_short_metadata_value(current[key])}")
            else:
                diffs.extend(_collect_metadata_diffs(current[key], checkpoint[key], child_path))
        return diffs

    if isinstance(current, list) and isinstance(checkpoint, list):
        diffs = []
        if len(current) != len(checkpoint):
            diffs.append(f"{path}: length current={len(current)} checkpoint={len(checkpoint)}")
        for index, (current_item, checkpoint_item) in enumerate(zip(current, checkpoint, strict=False)):
            diffs.extend(_collect_metadata_diffs(current_item, checkpoint_item, f"{path}[{index}]"))
        return diffs

    if _metadata_values_equal(current, checkpoint):
        return []
    return [f"{path}: current={_short_metadata_value(current)} checkpoint={_short_metadata_value(checkpoint)}"]


def _append_float_mismatch(
    mismatches: list[str],
    path: str,
    metadata_value: object,
    runtime_value: object,
) -> None:
    """Append a runtime mismatch if two numeric values differ beyond metadata tolerance."""
    if not math.isclose(float(metadata_value), float(runtime_value), rel_tol=1e-9, abs_tol=1e-12):
        mismatches.append(f"{path}: metadata={metadata_value!r} runtime={runtime_value!r}")


"""
作用：接管训练生命周期（构造 → 保存 → 恢复），把标准 RSL-RL 流程中的网络/算法替换为项目自定义版本。
"""


class VelocityEstimatorOnPolicyRunner(OnPolicyRunner):
    """显式构造并完整保存项目内速度估计训练组件的 On-Policy Runner。"""

    """
    作用：显式构造 VelocityEstimatorActorCritic 和 VelocityEstimatorPPO（不依赖字符串反射），并初始化 rollout storage。
    """

    def _construct_algorithm(self, obs: TensorDict) -> VelocityEstimatorPPO:
        """由环境首批 TensorDict 和 runner 配置构建 policy、algorithm 与 storage。

        RND/对称性配置仍交给 RSL-RL 官方解析；项目只接管自定义类的实例化，避免依赖
        字符串反射是否能发现本地类。``obs`` 同时用于校验 observation contract 和确定
        Actor、Critic、Encoder 的真实输入宽度。
        """
        # ① RSL-RL 标准配置解析：RND、对称性
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # ② 兼容旧版 empirical_normalization 配置（已废弃，自动迁移到新字段）
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "empirical_normalization is deprecated; use policy observation normalization fields.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # ③ 复制配置后 pop class_name，避免传给父类构造函数
        policy_cfg = dict(self.policy_cfg)
        policy_class_name = policy_cfg.pop("class_name")
        if policy_class_name != "VelocityEstimatorActorCritic":
            raise ValueError(f"Unsupported estimator policy class: {policy_class_name}")
        # ④ 显式构造 policy = VelocityEstimatorActorCritic(obs, obs_groups, num_actions, ...)
        policy = VelocityEstimatorActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)
        self._validate_policy_metadata_against_runtime(policy)

        # ⑤ 显式构造 algorithm = VelocityEstimatorPPO(policy, ...)
        algorithm_cfg = dict(self.alg_cfg)
        algorithm_class_name = algorithm_cfg.pop("class_name")
        if algorithm_class_name != "VelocityEstimatorPPO":
            raise ValueError(f"Unsupported estimator algorithm class: {algorithm_class_name}")
        algorithm = VelocityEstimatorPPO(
            policy,
            device=self.device,
            **algorithm_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        # ⑥ 初始化 rollout storage（使用延迟清理版本）
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm

    def _validate_policy_metadata_against_runtime(self, policy: VelocityEstimatorActorCritic) -> None:
        """Compare deployment-critical metadata with the live IsaacLab environment config.

        This check intentionally covers only the runtime contract that changes tensor semantics or
        control timing: control period, physics period, decimation, history length, action joint order,
        action scale, and whether action targets use default offsets.

        Training randomization records such as joint zero-error range, IMU mounting-bias range, and
        actuator-delay randomization stay in ``deployment_metadata`` for checkpoint/export traceability,
        and are still compared against the checkpoint in :meth:`load`. They are not compared against the
        live environment here because Play/evaluation configs deliberately disable or narrow these
        randomizations while keeping the same policy input/output contract.
        """
        metadata = policy.deployment_metadata
        env = getattr(self.env, "unwrapped", self.env)
        mismatches: list[str] = []

        if hasattr(env, "step_dt"):
            _append_float_mismatch(mismatches, "control_period_s", metadata["control_period_s"], env.step_dt)
        if hasattr(env, "physics_dt"):
            _append_float_mismatch(mismatches, "physics_period_s", metadata["physics_period_s"], env.physics_dt)

        cfg = getattr(env, "cfg", None)
        if cfg is not None:
            if getattr(cfg, "decimation", None) != metadata["decimation"]:
                mismatches.append(f"decimation: metadata={metadata['decimation']!r} runtime={cfg.decimation!r}")

            history_cfg = getattr(getattr(cfg, "observations", None), policy.history_group, None)
            if history_cfg is not None and getattr(history_cfg, "history_length", None) != metadata["history_samples"]:
                mismatches.append(
                    f"history_samples: metadata={metadata['history_samples']!r} runtime={history_cfg.history_length!r}"
                )

            actions_cfg = getattr(cfg, "actions", None)
            if actions_cfg is not None:
                leg_action = getattr(actions_cfg, "leg_pos", None)
                wheel_action = getattr(actions_cfg, "wheel_vel", None)
                if leg_action is not None:
                    if list(leg_action.joint_names) != metadata["policy_action_semantics"]["leg_position"]["joints"]:
                        mismatches.append("policy_action_semantics.leg_position.joints differs from runtime cfg.")
                    _append_float_mismatch(
                        mismatches,
                        "policy_action_semantics.leg_position.scale",
                        metadata["policy_action_semantics"]["leg_position"]["scale"],
                        leg_action.scale,
                    )
                    if (
                        bool(leg_action.use_default_offset)
                        != metadata["policy_action_semantics"]["leg_position"]["uses_default_offset"]
                    ):
                        mismatches.append("policy_action_semantics.leg_position.uses_default_offset differs.")
                if wheel_action is not None:
                    if (
                        list(wheel_action.joint_names)
                        != metadata["policy_action_semantics"]["wheel_velocity"]["joints"]
                    ):
                        mismatches.append("policy_action_semantics.wheel_velocity.joints differs from runtime cfg.")
                    _append_float_mismatch(
                        mismatches,
                        "policy_action_semantics.wheel_velocity.scale",
                        metadata["policy_action_semantics"]["wheel_velocity"]["scale"],
                        wheel_action.scale,
                    )

        if mismatches:
            message = "\n".join(f"- {item}" for item in mismatches)
            raise RuntimeError(f"Deployment metadata does not match the live L5A WF environment:\n{message}")

    """
    作用：保存完整 checkpoint，除了模型权重和 PPO optimizer 状态外，还保存 Encoder optimizer 和部署元数据。
    """

    def save(self, path: str, infos: dict | None = None) -> None:
        """保存可无损续训、同时可安全导出的完整 checkpoint。

        除模型和 PPO optimizer 外，还保存 Encoder optimizer 的 Adam 状态。部署元数据描述
        观测布局、缩放、关节顺序等非权重契约，导出器会把它写进 manifest；仅有权重文件
        并不足以保证真机输入排列正确。
        """
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "deployment_metadata": self.alg.policy.deployment_metadata,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        if self.logger_type in ("neptune", "wandb") and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)
        """
        参考保存结构：
            checkpoint.pt:
                ├── model_state_dict           # Actor + Critic + Encoder 所有权重
                ├── optimizer_state_dict       # PPO optimizer (Actor + Critic) 的 Adam 动量
                ├── estimator_optimizer_state_dict  # Encoder optimizer 的 Adam 动量（新增）
                ├── deployment_metadata        # 部署契约（观测顺序、缩放、关节映射）
                ├── iter                       # 当前训练迭代数
                └── infos                      # 额外信息
        """

    """
    作用：
        恢复 checkpoint，优先保留 checkpoint 自身的部署契约。
        如果旧 checkpoint 缺少 Encoder optimizer，模型仍可加载但 Encoder 的 Adam 动量从零开始。
    """

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        """恢复 checkpoint，并优先保留 checkpoint 自身的部署契约。

        ``map_location`` 用于在不同训练设备或 CPU 导出环境中加载。checkpoint 的
        deployment metadata 必须与当前任务配置完全一致，否则直接报错并列出差异；
        这样不会在观测、动作或控制周期已经变更时继续使用旧权重。
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # deployment_metadata 不参与网络计算，却是部署端正确解释张量不可缺少的接口契约。
        checkpoint_metadata = loaded_dict.get("deployment_metadata")
        if checkpoint_metadata is None:
            raise RuntimeError("Checkpoint has no deployment_metadata; refusing to load it for L5A WF policy.")
        metadata_diffs = _collect_metadata_diffs(self.alg.policy.deployment_metadata, checkpoint_metadata)
        if metadata_diffs:
            shown_diffs = metadata_diffs[:50]
            diff_message = "\n".join(f"- {item}" for item in shown_diffs)
            if len(metadata_diffs) > len(shown_diffs):
                diff_message += f"\n- ... {len(metadata_diffs) - len(shown_diffs)} more differences omitted"
            raise RuntimeError(
                "Checkpoint deployment_metadata differs from the current L5A WF configuration; "
                f"refusing to load.\n{diff_message}"
            )

        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        # 推理/导出可以关闭 optimizer 恢复；续训则必须同时恢复两套 optimizer，才能保持
        # PPO 与监督 Encoder 各自的 Adam 动量连续性。
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            estimator_state = loaded_dict.get("estimator_optimizer_state_dict")
            if estimator_state is None:
                warnings.warn(
                    "Checkpoint has no estimator optimizer state; estimator Adam moments start fresh.",
                    RuntimeWarning,
                )
            else:
                self.alg.estimator_optimizer.load_state_dict(estimator_state)
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
