"""MuJoCo 适配器和单进程 sim2sim 主循环。

本文件把其他模块串起来：从 MuJoCo 读状态，构建观测和历史，执行双 ONNX 推理，
用混合 PD 控制器计算力矩，再写回 MuJoCo。整个过程在一个 Python 进程内完成。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .control import ActionDelayBuffer, MixedPDController
from .history import ObservationHistory
from .keyboard import KeyboardCommand
from .manifest import ContractError, DeploymentBundle
from .observation import ObservationBuilder
from .policy import SplitOnnxPolicy


def _named_id(model: mujoco.MjModel, object_type, name: str) -> int:
    """按名字解析 MuJoCo 对象 id；找不到时转成清晰的契约错误。"""
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ContractError(f"MJCF has no {object_type.name} named {name!r}.")
    return int(object_id)


# MuJoCo 仿真器封装
class MujocoAdapter:
    """封装 MuJoCo model/data，并暴露按 policy_order 排列的机器人状态。"""

    def __init__(self, bundle: DeploymentBundle) -> None:
        self.bundle = bundle
        # MjModel（模型）：机器人的物理属性（质量、关节、几何体），不变。
        # MjData（数据）：机器人的实时状态（位置、速度、力），每步更新。
        self.model = mujoco.MjModel.from_xml_path(str(bundle.mjcf_path))
        self.data = mujoco.MjData(self.model)
        deployment = bundle.deployment
        self.policy_order = list(deployment["policy_action_order"])
        self.hardware_order = list(deployment["hardware_dof_order"])
        self._validate_model(deployment)
        # 所有关节、执行器和传感器都按名字解析，避免依赖 qpos[-8:] 这类脆弱位置假设。
        self.policy_joint_ids = np.asarray(
            [_named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.policy_order], dtype=np.int32
        )
        self.policy_qpos_addresses = self.model.jnt_qposadr[self.policy_joint_ids].copy()
        self.policy_dof_addresses = self.model.jnt_dofadr[self.policy_joint_ids].copy()
        # 控制器先按 policy_order 计算力矩，再根据这些 actuator id 写入 hardware_order。
        self.hardware_actuator_ids = np.asarray(
            [_named_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.hardware_order], dtype=np.int32
        )
        for actuator_id, joint_name in zip(self.hardware_actuator_ids, self.hardware_order, strict=True):
            joint_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if int(self.model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise ContractError(f"Actuator {joint_name!r} is not connected to its same-named joint.")
        self.orientation_sensor_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
        self.gyro_sensor_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
        self.base_joint_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_joint")
        self.base_qpos_address = int(self.model.jnt_qposadr[self.base_joint_id])
        key_name = deployment["robot_model"]["keyframe"]
        self.keyframe_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        self.reset()

    def _validate_model(self, deployment: dict) -> None:
        """检查 XML 的维度、物理步长和平地场景是否符合部署契约。"""
        expected_dimensions = (15, 14, 8)
        actual_dimensions = (self.model.nq, self.model.nv, self.model.nu)
        if actual_dimensions != expected_dimensions:
            raise ContractError(
                f"MJCF dimensions mismatch: expected nq/nv/nu={expected_dimensions}, got {actual_dimensions}."
            )
        if not np.isclose(self.model.opt.timestep, deployment["physics_period_s"], rtol=0.0, atol=1.0e-12):
            raise ContractError(
                f"MJCF timestep mismatch: expected {deployment['physics_period_s']}, got {self.model.opt.timestep}."
            )
        world_geoms = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in range(self.model.ngeom)
            if self.model.geom_bodyid[geom_id] == 0
        }
        if world_geoms != {"plane"}:
            raise ContractError(f"Expected a flat world with only the plane geom, got {sorted(world_geoms)}.")

    @staticmethod
    def _sensor_data(model: mujoco.MjModel, data: mujoco.MjData, sensor_id: int) -> np.ndarray:
        """从 MuJoCo 的扁平 sensordata 数组中按地址和维度切片，返回副本。

        MuJoCo 把所有传感器的数值拼接在一个一维数组 `data.sensordata` 中，
        每个传感器的起始地址和维度由 `model.sensor_adr` 和 `model.sensor_dim` 给出。

        Args:
            model: MuJoCo 模型对象。
            data: MuJoCo 数据对象。
            sensor_id: 传感器在模型中的 id。

        Returns:
            该传感器当前读数的副本，shape 由 `sensor_dim` 决定。
        """
        address = int(model.sensor_adr[sensor_id])
        dimension = int(model.sensor_dim[sensor_id])
        return np.asarray(data.sensordata[address : address + dimension]).copy()

    def reset(self) -> None:
        """从 Manifest 指定的 keyframe 复位，并确认关节默认角和契约一致。"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        mujoco.mj_forward(self.model, self.data)
        expected_q = np.asarray(self.bundle.deployment["default_joint_positions"]["values"])
        actual_q, _ = self.joint_state_policy()
        if not np.allclose(actual_q, expected_q, rtol=0.0, atol=1.0e-7):
            raise ContractError(
                f"MJCF keyframe joint positions do not match manifest defaults: {actual_q} vs {expected_q}."
            )

    def joint_state_policy(self) -> tuple[np.ndarray, np.ndarray]:
        """读取 policy_order 下的 8 维关节位置和速度。"""
        q = np.asarray(self.data.qpos[self.policy_qpos_addresses]).copy()
        dq = np.asarray(self.data.qvel[self.policy_dof_addresses]).copy()
        return q, dq

    def orientation_wxyz(self) -> np.ndarray:
        """读取基座姿态四元数，wxyz 顺序，shape `(4,)`。"""
        value = self._sensor_data(self.model, self.data, self.orientation_sensor_id)
        if value.shape != (4,):
            raise ContractError(f"orientation sensor has dimension {value.shape}, expected (4,).")
        return value

    def angular_velocity(self) -> np.ndarray:
        """读取基座角速度，机器人坐标系下，shape `(3,)`。"""
        value = self._sensor_data(self.model, self.data, self.gyro_sensor_id)
        if value.shape != (3,):
            raise ContractError(f"angular-velocity sensor has dimension {value.shape}, expected (3,).")
        return value

    @property
    def base_height(self) -> float:
        """基座在世界坐标系中的 z 坐标（高度），用于摔倒检测。"""
        return float(self.data.qpos[self.base_qpos_address + 2])

    def apply_hardware_torque(self, torque: np.ndarray) -> None:
        """把 hardware_order 下的力矩写入 MuJoCo data.ctrl。"""
        torque = np.asarray(torque, dtype=np.float64)
        if torque.shape != (8,):
            raise ValueError(f"Expected hardware torque shape (8,), got {torque.shape}.")
        self.data.ctrl[self.hardware_actuator_ids] = torque

    def step(self) -> None:
        """推进一个物理步（0.005s），并检查状态合法性防止数值爆炸扩散。"""
        mujoco.mj_step(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise FloatingPointError("MuJoCo state contains NaN or Inf.")


@dataclass(frozen=True)
class RunSummary:
    """一次运行结束后打印的简短统计，便于 headless 验证。"""

    policy_steps: int
    physics_steps: int
    resets: int
    min_base_height: float
    final_base_height: float
    fell: bool

# 主仿真循环
class Sim2SimRunner:
    """单进程 sim2sim 编排器。

    负责把 MujocoAdapter、SplitOnnxPolicy、MixedPDController、ObservationBuilder、
    ObservationHistory、ActionDelayBuffer 和 KeyboardCommand 串成一个闭环主循环。
    """

    def __init__(
        self,
        bundle: DeploymentBundle,
        command: KeyboardCommand,
        action_delay_steps: int = 0,
        fall_height_m: float = 0.35,
        trace_path: str | Path | None = None,
    ) -> None:
        """初始化所有子组件并执行首次 reset。

        Args:
            bundle: 经过契约校验的部署包，包含 manifest、模型路径和 MJCF。
            command: 键盘命令管理器，提供每策略步的速度命令。
            action_delay_steps: 动作延迟物理步数（0~6），模拟真实硬件通信延迟。
            fall_height_m: 摔倒判定高度阈值，基座 z 低于此值视为摔倒。
            trace_path: 可选 NPZ 文件路径，开启后每个策略步记录完整状态。
        """
        self.bundle = bundle
        self.adapter = MujocoAdapter(bundle)
        self.policy = SplitOnnxPolicy(bundle)
        self.controller = MixedPDController(bundle.deployment)
        self.observation_builder = ObservationBuilder(bundle.deployment)
        self.history = ObservationHistory(10, 28)
        self.delay = ActionDelayBuffer(action_delay_steps)
        self.command = command
        self.fall_height_m = float(fall_height_m)
        self.trace_path = Path(trace_path).expanduser().resolve() if trace_path else None
        self.trace: dict[str, list[np.ndarray | float]] = {
            key: []
            for key in (
                "time",
                "base_height",
                "qpos",
                "qvel",
                "command",
                "proprioception",
                "estimated_base_linear_velocity",
                "action",
                "torque_policy",
            )
        }
        self.previous_action = np.zeros(8, dtype=np.float32)
        self.current_observation = np.zeros(28, dtype=np.float32)
        self.reset()

    def _observe(self) -> np.ndarray:
        """从 MuJoCo 当前状态构建一帧 28 维 proprioception。"""
        q, dq = self.adapter.joint_state_policy()
        return self.observation_builder.build(
            self.adapter.angular_velocity(),
            self.adapter.orientation_wxyz(),
            q,
            dq,
            self.previous_action,
        )

    def reset(self) -> None:
        """复位仿真、延迟队列、上一帧动作和 10 帧历史。"""
        self.adapter.reset()
        self.delay.reset()
        self.previous_action.fill(0.0)
        self.history.reset()
        self.current_observation = self._observe()
        self.history.append(self.current_observation)

    def _record(self, estimated_velocity: np.ndarray, action: np.ndarray, torque_policy: np.ndarray) -> None:
        """按策略步记录可选 trace，用于离线排查观测、动作和力矩。

        Args:
            estimated_velocity: velocity_estimator 输出的基座线速度 `(3,)`。
            action: Actor 输出的原始 8 维动作。
            torque_policy: decimation 最后一个物理步的 PD 力矩（policy_order）。
        """
        if self.trace_path is None:
            return
        values = {
            "time": float(self.adapter.data.time),
            "base_height": self.adapter.base_height,
            "qpos": self.adapter.data.qpos.copy(),
            "qvel": self.adapter.data.qvel.copy(),
            "command": self.command.command(),
            "proprioception": self.current_observation.copy(),
            "estimated_base_linear_velocity": estimated_velocity.copy(),
            "action": action.copy(),
            "torque_policy": torque_policy.copy(),
        }
        for key, value in values.items():
            self.trace[key].append(value)

    def _save_trace(self) -> None:
        """把运行期间累积的 trace 数据写入压缩 NPZ 文件，供离线回放和调试。"""
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(values) for key, values in self.trace.items()}
        np.savez_compressed(self.trace_path, **arrays)

    def run(
        self,
        duration_s: float | None,
        realtime_factor: float,
        headless: bool,
        keyboard_enabled: bool,
    ) -> RunSummary:
        """执行主循环。

        每 10 ms 做一次 `history -> Encoder -> Actor -> action`；每个 action
        在 decimation=2 的两个 5 ms MuJoCo 物理步内保持并计算 PD 力矩。
        """
        if realtime_factor <= 0.0:
            raise ValueError("Realtime factor must be positive.")
        if headless and duration_s is None:
            raise ValueError("Headless mode requires a finite duration.")
        viewer = None
        if not headless:
            from mujoco import viewer as mujoco_viewer

            callback = self.command.key_callback if keyboard_enabled else None
            viewer = mujoco_viewer.launch_passive(self.adapter.model, self.adapter.data, key_callback=callback)

        policy_steps = 0
        physics_steps = 0
        resets = 0
        min_height = self.adapter.base_height
        fell = False
        max_physics_steps = None
        if duration_s is not None:
            max_physics_steps = int(round(duration_s / self.bundle.deployment["physics_period_s"]))
        try:
            while max_physics_steps is None or physics_steps < max_physics_steps:
                if viewer is not None and not viewer.is_running():
                    break
                if self.command.consume_reset():
                    self.reset()
                    resets += 1
                if self.command.paused:
                    if viewer is not None:
                        viewer.sync()
                    time.sleep(0.01)
                    continue

                tick_start = time.perf_counter()

                # === 策略推理（每0.01秒一次）===
                command = self.command.command()
                estimated_velocity, action = self.policy.infer(
                    self.history.batched(),             # [1,10,28]
                    self.current_observation[None, :],  # [1,28]
                    command[None, :]                    # [1,3]
                )
                self.previous_action = action.copy()
                last_torque_policy = np.zeros(8, dtype=np.float64)

                # decimation=2：一次策略输出覆盖两个 200 Hz 物理步，策略等效 100 Hz。
                # === 物理仿真（每0.005秒一次，跑2步）===
                for _ in range(self.bundle.deployment["decimation"]):
                    delayed_action = self.delay.apply(action)
                    q, dq = self.adapter.joint_state_policy()
                    last_torque_policy = self.controller.compute_policy_torque(delayed_action, q, dq)
                    self.adapter.apply_hardware_torque(self.controller.to_hardware_order(last_torque_policy))
                    self.adapter.step()
                    physics_steps += 1
                    min_height = min(min_height, self.adapter.base_height)
                    fell = fell or self.adapter.base_height < self.fall_height_m
                    if max_physics_steps is not None and physics_steps >= max_physics_steps:
                        break

                # === 构建下一帧观测 ===
                self.current_observation = self._observe()
                self.history.append(self.current_observation)
                self._record(estimated_velocity, action, last_torque_policy)
                policy_steps += 1
                if viewer is not None:
                    viewer.sync()
                target_wall_time = self.bundle.deployment["control_period_s"] / realtime_factor

                # === 时间同步 ===
                sleep_time = target_wall_time - (time.perf_counter() - tick_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
        finally:
            if viewer is not None:
                viewer.close()
                # launch_passive owns an asynchronous GLFW thread; allow it to finish teardown.
                time.sleep(0.25)
            self._save_trace()
        return RunSummary(
            policy_steps=policy_steps,
            physics_steps=physics_steps,
            resets=resets,
            min_base_height=min_height,
            final_base_height=self.adapter.base_height,
            fell=fell,
        )
