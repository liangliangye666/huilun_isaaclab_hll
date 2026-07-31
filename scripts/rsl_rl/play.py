# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--record_debug",
    action="store_true",
    default=False,
    help="Record play-time tensors for debugging policy, robot state, rewards, contacts, and actions.",
)
parser.add_argument(
    "--record_debug_length",
    type=int,
    default=2000,
    help="Maximum number of policy steps to record when --record_debug is enabled. Use 0 for no debug limit.",
)
parser.add_argument(
    "--record_debug_envs",
    type=int,
    default=4,
    help="Number of leading vectorized environments to record for debug traces.",
)
parser.add_argument(
    "--record_debug_interval",
    type=int,
    default=1,
    help="Record one debug sample every N policy steps.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
ORIGINAL_ARGV = sys.argv.copy()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import json
import os
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import huilun_isaaclab.tasks  # noqa: F401
import numpy as np
import torch
from huilun_isaaclab.learning.rsl_rl import (
    VelocityEstimatorOnPolicyRunner,
    export_velocity_estimator_policy,
)
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

RSL_RL_VERSION = "3.1.2"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    raise RuntimeError(f"rsl-rl-lib>={RSL_RL_VERSION} is required, but {installed_version} is installed.")



@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    if args_cli.record_debug:
        if args_cli.record_debug_envs <= 0:
            raise ValueError("--record_debug_envs must be positive.")
        if args_cli.record_debug_interval <= 0:
            raise ValueError("--record_debug_interval must be positive.")
        if args_cli.record_debug_length < 0:
            raise ValueError("--record_debug_length must be non-negative.")

    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for loading training checkpoints
    train_log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    train_log_root_path = os.path.abspath(train_log_root_path)
    print(f"[INFO] Loading experiment from directory: {train_log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(train_log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.record_debug:
        play_log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", f"{agent_cfg.experiment_name}_play"))
        checkpoint_tag = Path(resume_path).stem
        log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir_name += f"_{agent_cfg.run_name}" if agent_cfg.run_name else f"_{checkpoint_tag}"
        log_dir = os.path.join(play_log_root_path, log_dir_name)
        env_cfg.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        with open(os.path.join(log_dir, "play_context.json"), "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "task": args_cli.task,
                    "checkpoint": os.path.abspath(resume_path),
                    "command_line": ORIGINAL_ARGV,
                    "record_debug_length": args_cli.record_debug_length,
                    "record_debug_envs": args_cli.record_debug_envs,
                    "record_debug_interval": args_cli.record_debug_interval,
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[INFO] Recording play debug logs to directory: {log_dir}")
    else:
        log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "VelocityEstimatorOnPolicyRunner":
        runner = VelocityEstimatorOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(log_dir, "exported")
    if getattr(policy_nn, "is_velocity_estimator_policy", False):
        export_velocity_estimator_policy(policy_nn, path=export_model_dir)
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    debug_recorder = None
    if args_cli.record_debug:
        debug_recorder = PlayDebugRecorder(env, policy_nn, log_dir, resume_path, args_cli, ORIGINAL_ARGV)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                pre_obs = obs
                # agent stepping
                actions = policy(pre_obs)
                # env stepping
                obs, rewards, dones, extras = env.step(actions)
                if debug_recorder is not None:
                    debug_recorder.record(timestep, pre_obs, actions, obs, rewards, dones, extras)
                # reset recurrent states for episodes that have terminated
                policy_nn.reset(dones)

            timestep += 1
            # Exit the play loop after recording one video
            if args_cli.video and timestep >= args_cli.video_length:
                break
            if args_cli.record_debug and args_cli.record_debug_length > 0:
                if timestep >= args_cli.record_debug_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if debug_recorder is not None:
            debug_recorder.close()

    # close the simulator
    env.close()


# =============== 日志记录功能 --record_debug ============================
'''
feat: 增加play.py中的日志记录功能 --record_debug，目的是为了后续分析抖动、动作异常、速度估计偏差、接触异常、奖励异常等问题

增加的参数：
--record_debug
--record_debug_length 2000      # 默认记录 2000 个 policy step；设为 0 表示不限长
--record_debug_envs 4           # 默认记录前 4 个 env
--record_debug_interval 1       # 默认每步记录一次

保存路径会放到 train 同级的 log 目录下，例如：
logs/rsl_rl/l5a_wf_flat_play/2026-07-31_12-34-56_model_14999/
'''
def _debug_key(name: str) -> str:
    """Convert manager or tensor names to stable NPZ keys."""
    return name.replace("/", "__").replace(" ", "_").replace(".", "_")


def _json_safe(value):
    """Convert common runtime values to JSON-safe containers."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class PlayDebugRecorder:
    """Read-only recorder for play-time diagnostics.

    The recorder samples existing manager, robot, sensor, and observation buffers. It must not call
    reward/MDP functions because some project rewards maintain internal state.
    """

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        policy_nn,
        log_dir: str,
        resume_path: str,
        args_cli: argparse.Namespace,
        original_argv: list[str],
    ) -> None:
        self.env = env
        self.unwrapped = env.unwrapped
        self.policy_nn = policy_nn
        self.log_dir = log_dir
        self.debug_dir = os.path.join(log_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)

        self.num_envs = int(self.unwrapped.num_envs)
        env_count = max(1, min(int(args_cli.record_debug_envs), self.num_envs))
        self.env_ids = torch.arange(env_count, device=self.unwrapped.device)
        self.interval = max(1, int(args_cli.record_debug_interval))
        self.step_dt = float(getattr(self.unwrapped, "step_dt", 0.0))
        self.physics_dt = float(getattr(self.unwrapped, "physics_dt", 0.0))
        self.buffers: dict[str, list[np.ndarray]] = {}
        self.recorded_steps = 0

        self.extras_path = os.path.join(self.debug_dir, "extras.jsonl")
        self.extras_file = open(self.extras_path, "w", encoding="utf-8")
        self.trace_path = os.path.join(self.debug_dir, "debug_trace.npz")
        self.manifest_path = os.path.join(self.debug_dir, "debug_manifest.json")

        try:
            self.robot = self.unwrapped.scene["robot"]
        except Exception:
            self.robot = None
        self.contact_sensor = getattr(self.unwrapped.scene, "sensors", {}).get("contact_forces", None)
        self.joint_ids, self.joint_names = self._resolve_robot_joints()
        self.wheel_body_ids, self.wheel_body_names = self._resolve_wheel_bodies()
        self._write_manifest(resume_path=resume_path, args_cli=args_cli, original_argv=original_argv)

    def _resolve_robot_joints(self) -> tuple[list[int] | slice, list[str]]:
        if self.robot is None:
            return slice(None), []
        metadata = getattr(self.policy_nn, "deployment_metadata", {})
        policy_joint_names = metadata.get("policy_action_order")
        if policy_joint_names:
            try:
                joint_ids, joint_names = self.robot.find_joints(policy_joint_names, preserve_order=True)
                return joint_ids, joint_names
            except Exception:
                pass
        return slice(None), list(self.robot.joint_names)

    def _resolve_wheel_bodies(self) -> tuple[list[int] | slice, list[str]]:
        if self.robot is None:
            return slice(None), []
        try:
            body_ids, body_names = self.robot.find_bodies(".*wheel.*", preserve_order=True)
            return body_ids, body_names
        except Exception:
            return slice(None), list(self.robot.body_names)

    def _select(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.ndim > 0 and tensor.shape[0] == self.num_envs:
                tensor = tensor[self.env_ids]
            return tensor.cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] == self.num_envs:
                return value[self.env_ids.detach().cpu().numpy()]
            return value
        if isinstance(value, (int, float, bool)):
            return np.asarray(value)
        return None

    def _append(self, key: str, value) -> None:
        array = self._select(value)
        if array is None:
            return
        self.buffers.setdefault(_debug_key(key), []).append(np.asarray(array))

    def _append_obs(self, prefix: str, obs) -> None:
        if obs is None or not hasattr(obs, "keys"):
            return
        for key in obs.keys():
            value = obs[key]
            if isinstance(value, torch.Tensor):
                self._append(f"{prefix}/{key}", value)

    def _append_action_terms(self) -> None:
        action_manager = getattr(self.unwrapped, "action_manager", None)
        if action_manager is None:
            return
        self._append("action_manager/action", action_manager.action)
        self._append("action_manager/prev_action", action_manager.prev_action)
        for term_name in action_manager.active_terms:
            term = action_manager.get_term(term_name)
            self._append(f"action_terms/{term_name}/raw_actions", getattr(term, "raw_actions", None))
            self._append(f"action_terms/{term_name}/processed_actions", getattr(term, "processed_actions", None))

    def _append_command_terms(self) -> None:
        command_manager = getattr(self.unwrapped, "command_manager", None)
        if command_manager is None:
            return
        for term_name in command_manager.active_terms:
            self._append(f"commands/{term_name}", command_manager.get_command(term_name))

    def _append_reward_terms(self) -> None:
        reward_manager = getattr(self.unwrapped, "reward_manager", None)
        if reward_manager is None:
            return
        self._append("reward_manager/step_reward_terms", getattr(reward_manager, "_step_reward", None))
        self._append("reward_manager/reward_buf", getattr(reward_manager, "_reward_buf", None))
        episode_sums = getattr(reward_manager, "_episode_sums", {})
        for term_name, value in episode_sums.items():
            self._append(f"reward_manager/episode_sums/{term_name}", value)

    def _append_termination_terms(self) -> None:
        termination_manager = getattr(self.unwrapped, "termination_manager", None)
        if termination_manager is None:
            return
        self._append("termination_manager/dones", termination_manager.dones)
        self._append("termination_manager/terminated", termination_manager.terminated)
        self._append("termination_manager/time_outs", termination_manager.time_outs)
        for term_name in termination_manager.active_terms:
            self._append(f"terminations/{term_name}", termination_manager.get_term(term_name))

    def _append_robot_state(self) -> None:
        if self.robot is None:
            return
        data = self.robot.data
        for attr_name in (
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_lin_vel_b",
            "root_ang_vel_w",
            "root_ang_vel_b",
            "projected_gravity_b",
        ):
            self._append(f"robot/{attr_name}", getattr(data, attr_name, None))
        for attr_name in (
            "joint_pos",
            "joint_vel",
            "joint_acc",
            "default_joint_pos",
            "joint_pos_target",
            "joint_vel_target",
            "applied_torque",
            "computed_torque",
        ):
            value = getattr(data, attr_name, None)
            if value is not None:
                self._append(f"robot/{attr_name}_policy_order", value[:, self.joint_ids])
        for attr_name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            value = getattr(data, attr_name, None)
            if value is not None:
                self._append(f"robot/{attr_name}_wheels", value[:, self.wheel_body_ids])

    def _append_contact_state(self) -> None:
        if self.contact_sensor is None:
            return
        contact_data = self.contact_sensor.data
        self._append("contact_forces/net_forces_w", getattr(contact_data, "net_forces_w", None))
        self._append("contact_forces/net_forces_w_history", getattr(contact_data, "net_forces_w_history", None))

    def _append_policy_diagnostics(self, obs) -> None:
        with torch.inference_mode():
            if hasattr(self.policy_nn, "get_actor_obs"):
                self._append("policy/actor_input", self.policy_nn.get_actor_obs(obs))
            if hasattr(self.policy_nn, "estimate_base_lin_vel"):
                self._append("policy/estimated_base_lin_vel", self.policy_nn.estimate_base_lin_vel(obs, detach=True))
            if hasattr(self.policy_nn, "get_estimator_target"):
                self._append("policy/estimator_target_base_lin_vel", self.policy_nn.get_estimator_target(obs))

    def record(self, step: int, pre_obs, policy_actions, post_obs, rewards, dones, extras: dict) -> None:
        if step % self.interval != 0:
            return
        self._append("step", step)
        self._append("time_s", step * self.step_dt)
        self._append("episode_length", getattr(self.unwrapped, "episode_length_buf", None))
        self._append("policy/actions_output", policy_actions)
        self._append("step/rewards", rewards)
        self._append("step/dones", dones)
        if isinstance(extras, dict) and "time_outs" in extras:
            self._append("step/time_outs", extras["time_outs"])
        self._append_obs("pre_obs", pre_obs)
        self._append_obs("post_obs", post_obs)
        self._append_policy_diagnostics(pre_obs)
        self._append_action_terms()
        self._append_command_terms()
        self._append_reward_terms()
        self._append_termination_terms()
        self._append_robot_state()
        self._append_contact_state()
        self._write_extras(step, extras)
        self.recorded_steps += 1

    def _write_extras(self, step: int, extras: dict) -> None:
        payload = {"step": int(step), "time_s": float(step * self.step_dt), "extras": _json_safe(extras)}
        self.extras_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_manifest(self, resume_path: str, args_cli: argparse.Namespace, original_argv: list[str]) -> None:
        reward_manager = getattr(self.unwrapped, "reward_manager", None)
        action_manager = getattr(self.unwrapped, "action_manager", None)
        termination_manager = getattr(self.unwrapped, "termination_manager", None)
        observation_manager = getattr(self.unwrapped, "observation_manager", None)
        command_manager = getattr(self.unwrapped, "command_manager", None)
        robot_actuators = {}
        if self.robot is not None:
            for name, actuator in self.robot.actuators.items():
                robot_actuators[name] = {
                    "class": actuator.__class__.__name__,
                    "joint_indices": _json_safe(getattr(actuator, "joint_indices", None)),
                    "min_delay": _json_safe(getattr(getattr(actuator, "cfg", None), "min_delay", None)),
                    "max_delay": _json_safe(getattr(getattr(actuator, "cfg", None), "max_delay", None)),
                }
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "command_line": original_argv,
            "task": args_cli.task,
            "checkpoint": os.path.abspath(resume_path),
            "log_dir": self.log_dir,
            "trace_path": self.trace_path,
            "extras_path": self.extras_path,
            "num_envs": self.num_envs,
            "recorded_env_ids": self.env_ids.detach().cpu().tolist(),
            "record_debug_interval": self.interval,
            "step_dt": self.step_dt,
            "physics_dt": self.physics_dt,
            "decimation": _json_safe(getattr(getattr(self.unwrapped, "cfg", None), "decimation", None)),
            "max_episode_length": _json_safe(getattr(self.unwrapped, "max_episode_length", None)),
            "policy_class": self.policy_nn.__class__.__name__,
            "deployment_metadata": _json_safe(getattr(self.policy_nn, "deployment_metadata", {})),
            "observation_terms": _json_safe(getattr(observation_manager, "active_terms", {})),
            "action_terms": {
                name: int(dim)
                for name, dim in zip(
                    getattr(action_manager, "active_terms", []),
                    getattr(action_manager, "action_term_dim", []),
                    strict=False,
                )
            },
            "command_terms": list(getattr(command_manager, "active_terms", [])),
            "reward_terms": {
                name: float(cfg.weight)
                for name, cfg in zip(
                    getattr(reward_manager, "active_terms", []),
                    getattr(reward_manager, "_term_cfgs", []),
                    strict=False,
                )
            },
            "termination_terms": list(getattr(termination_manager, "active_terms", [])),
            "robot_joint_names": list(getattr(self.robot, "joint_names", [])) if self.robot is not None else [],
            "robot_body_names": list(getattr(self.robot, "body_names", [])) if self.robot is not None else [],
            "recorded_joint_names": self.joint_names,
            "recorded_wheel_body_names": self.wheel_body_names,
            "contact_sensor_body_names": list(getattr(self.contact_sensor, "body_names", []))
            if self.contact_sensor is not None
            else [],
            "robot_actuators": robot_actuators,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)

    def close(self) -> None:
        if not self.extras_file.closed:
            self.extras_file.close()
        arrays = {}
        for key, values in self.buffers.items():
            try:
                arrays[key] = np.stack(values, axis=0)
            except ValueError:
                arrays[key] = np.asarray(values, dtype=object)
        np.savez_compressed(self.trace_path, **arrays)
        with open(self.manifest_path, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest["num_recorded_steps"] = self.recorded_steps
        manifest["npz_keys"] = sorted(arrays.keys())
        manifest["npz_shapes"] = {key: list(value.shape) for key, value in arrays.items() if hasattr(value, "shape")}
        with open(self.manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved play debug trace to: {self.trace_path}")




if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()