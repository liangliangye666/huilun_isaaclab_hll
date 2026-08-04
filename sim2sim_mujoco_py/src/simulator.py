"""MuJoCo 适配器、键盘命令状态和单进程 sim2sim 主循环。

本模块把其余核心组件串成闭环：从 MuJoCo 读取状态，构造当前 proprioception
和历史窗口，依次执行 Encoder 与 Actor，用位置/速度 PD 控制器计算力矩，再写回
MuJoCo。仿真、推理、控制和 viewer 同属一个 Python 进程。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from control import ActionDelayBuffer, GenericPDController
from model import DeploymentBundle, DeploymentError, SplitOnnxPolicy, load_deployment
from observation import ObservationBuilder, ObservationHistory


@dataclass(frozen=True)
class RuntimeConfig:
    """用户在 ``sim2sim.py`` 顶部配置的运行参数快照。

    ``fixed_command`` 的顺序由 Manifest ``command_order`` 定义；动作延迟按物理步
    计数；``duration_s=0`` 只允许用于有 viewer 的无限时运行。

    各字段含义：
        model_dir: 包含 policy_manifest.json 和两个 ONNX 文件的导出目录
        mjcf_path: 与策略匹配的 MuJoCo XML 机器人模型路径
        fixed_command: 固定速度命令，顺序与 Manifest command_order 一致
        keyboard_enabled: 是否启用键盘实时调整命令（依赖 viewer）
        keyboard_linear_step: 每次按键调整前进速度的步长（m/s）
        keyboard_yaw_step: 每次按键调整偏航速度的步长（rad/s）
        headless: True 无界面运行，False 打开 MuJoCo viewer
        duration_s: 仿真时长（秒）；0 表示 viewer 模式下无限运行
        realtime_factor: 实时倍率（1.0=实时，2.0=两倍速，0.5=半速）
        action_delay_steps: 动作延迟的物理步数（必须位于训练范围内）
        trace_path: 可选的调试轨迹保存路径（.npz 格式）
        fall_height_m: 摔倒判断的基座高度阈值（米）
    """

    model_dir: Path
    mjcf_path: Path
    fixed_command: tuple[float, ...]
    keyboard_enabled: bool
    keyboard_linear_step: float
    keyboard_yaw_step: float
    headless: bool
    duration_s: float
    realtime_factor: float
    action_delay_steps: int
    trace_path: Path | None
    fall_height_m: float


@dataclass(frozen=True)
class RunSummary:
    """运行结束后的统计结果，主要用于 headless 验证和日志输出。

    各字段含义：
        policy_steps: 执行的策略周期总数（一次 = Encoder + Actor + decimation 个物理步）
        physics_steps: 执行的 MuJoCo 物理步总数
        resets: 用户手动复位的次数
        min_base_height: 整个运行过程中基座的最低高度（米）
        final_base_height: 运行结束时基座的高度（米）
        fell: 是否曾在任何时刻低于 fall_height_m 阈值
    """

    policy_steps: int
    physics_steps: int
    resets: int
    min_base_height: float
    final_base_height: float
    fell: bool


# 键盘控制器
class KeyboardCommand:
    """保存固定命令，并处理 MuJoCo viewer 的键盘回调。

    viewer 的 ``key_callback`` 可能由 GLFW 线程触发，而主循环会同时读取命令、
    pause 和 reset 状态，因此所有共享状态都必须在同一把 Lock 下访问。

    输入参数：
        initial_command（tuple[float, ...]）：
            用户配置的固定速度命令，如 (0.2, 0.0, 0.0) 表示前进 0.2m/s。
            它的长度和顺序必须与 Manifest 中的 command_order 完全一致。
            这个参数是用户在 sim2sim.py 中设置的 FIXED_COMMAND。

        command_order（list[str]）：
            命令各维度的名称列表，来自 Manifest。
            例如 ["linear_velocity_x", "linear_velocity_y", "angular_velocity_z"]。
            它定义了命令向量中每个数字的含义。

        command_limits（dict[str, list[float]]）：
            每个命令维度的取值范围，来自 Manifest 的训练配置。
            例如 {"linear_velocity_x": [-1.0, 1.0], "angular_velocity_z": [-1.0, 1.0]}。
            键盘调整命令后会自动裁剪到这个范围内，防止给策略网络输入超出训练分布的值。

        linear_step（float）：
            每次按 W/S 键调整前进速度的步长（m/s）。
            例如 0.1 表示按一次 W 速度增加 0.1m/s。

        yaw_step（float）：
            每次按 A/D 键调整偏航速度的步长（rad/s）。
            例如 0.1 表示按一次 A 逆时针转 0.1rad/s。
    """

    def __init__(
        self,
        initial_command: tuple[float, ...],
        command_order: list[str],
        command_limits: dict[str, list[float]],
        linear_step: float,
        yaw_step: float,
    ) -> None:
        """按 Manifest 命令顺序保存初值，并定位可由键盘修改的命令分量。"""
        self._lock = threading.Lock()   # 创建线程锁，保护后续所有共享状态
        self._command = np.asarray(initial_command, dtype=np.float32)   # 把元组转成 NumPy 数组 [0.2, 0.0, 0.0]
        # 维度校验：用户在 sim2sim.py 中配置的 FIXED_COMMAND 长度和 Manifest 定义的命令维度
        if self._command.shape != (len(command_order),):
            raise ValueError(f"FIXED_COMMAND 应有 {len(command_order)} 维，实际为 {self._command.shape}。")
        self.command_order = list(command_order)    # 保存命令名称列表（副本，防止外部修改）
        self.command_limits = command_limits        # 保存取值范围（引用，共享 Manifest 数据）
        # 保存键盘调整步长
        self.linear_step = float(linear_step)
        self.yaw_step = float(yaw_step)
        # 定位可键盘控制的命令维度
        self.linear_index = self._optional_index("linear_velocity_x")
        self.yaw_index = self._optional_index("angular_velocity_z")
        '''
        状态标志初始化：
            _paused：
                仿真是否暂停。空格键切换。
                主循环检查这个标志来决定是否跳过物理推进。
            _reset_requested：
                是否请求复位。R 键设置。
                主循环检查后执行复位并清零（通过 consume_reset() 方法），保证一次按键只触发一次复位。
        '''
        self._paused = False    # 初始状态：不暂停
        self._reset_requested = False   # 初始状态：不请求复位
        self._clamp()   # 初始裁剪：逐维裁剪 self._command 到 command_limits 范围内

    def _optional_index(self, name: str) -> int | None:
        """返回可选命令分量索引，使没有 vx 或 wz 的任务仍可使用固定命令。"""
        return self.command_order.index(name) if name in self.command_order else None

    def _clamp(self) -> None:
        """按 Manifest 的训练命令范围逐维裁剪当前命令。"""
        for index, name in enumerate(self.command_order):
            if name not in self.command_limits:
                continue
            minimum, maximum = self.command_limits[name]
            self._command[index] = np.clip(self._command[index], float(minimum), float(maximum))

    def value(self) -> np.ndarray:
        """在线程锁内返回命令副本，防止主循环读到修改一半的数据。"""
        with self._lock:
            return self._command.copy()

    def is_paused(self) -> bool:
        """返回当前暂停状态。"""
        with self._lock:
            return self._paused

    def consume_reset(self) -> bool:
        """读取并清除一次性 reset 请求，避免一次按键触发多次复位。"""
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested

    '''
    MuJoCo viewer 的键盘事件处理器。
        当用户在可视化窗口中按下键盘时，GLFW 渲染线程会调用这个回调函数，把按键编码传进来，然后根据按键类型修改共享的命令状态（前进速度、偏航速度）或控制状态（暂停、复位）。
    '''
    def key_callback(self, keycode: int) -> None:
        """W/S 调整前进速度，A/D 调整偏航，C 清零，R 复位，空格暂停。"""
        '''
        输入参数：
            keycode（int）：GLFW 传入的按键编码。
            对于字母键（A-Z），这个值是 ASCII 码（如 W=87, S=83, A=65, D=68）；对于特殊键（如空格），是 GLFW 定义的常量（空格=32）。
            这个值由 MuJoCo viewer 内部自动传入，用户不需要手动调用。
        输出：
            self._command：命令数组（前进速度、侧移速度、转向速度）
            self._paused：暂停标志
            self._reset_requested：复位请求标志
        '''
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        '''
        chr(keycode)：
            Python 内置函数，把 ASCII 码转成对应字符。
            例如 chr(87) → 'W'，chr(65) → 'A'。
        .upper()：
            转成大写。
            这样无论用户按的是大写 W 还是小写 w，都统一处理。
            GLFW 传入的 keycode 取决于键盘状态（CapsLock、Shift），统一大写消除了这种不确定性。
        if 0 <= keycode < 256 else ""：
            安全边界检查。
            GLFW 的某些特殊键（如方向键、功能键 F1-F12）的编码可能超出 0-255 的 ASCII 范围。
            chr() 对超出范围的编码会抛出 ValueError，或者产生无意义的 Unicode 字符。
            这个检查确保：
                合法的 ASCII 键 → 转成大写字符
                特殊键（如空格=32 仍在 0-255 内）→ 也能正常转字符
                真正超出范围的键 → 赋值为空字符串 ""，后续所有 key == "W" 之类的比较都不匹配，自然跳过
        '''

        with self._lock:
            if key == "W" and self.linear_index is not None:
                self._command[self.linear_index] += self.linear_step    # W 键——增加前进速度
            elif key == "S" and self.linear_index is not None:
                self._command[self.linear_index] -= self.linear_step    # S 键——减少前进速度
            elif key == "A" and self.yaw_index is not None:
                self._command[self.yaw_index] += self.yaw_step          # A/D 键——调整偏航速度
            elif key == "D" and self.yaw_index is not None:
                self._command[self.yaw_index] -= self.yaw_step
            elif key == "C":
                self._command.fill(0.0)                                 # C 键——清零所有命令
            elif key == "R":
                self._reset_requested = True                            # R 键——请求复位
            elif keycode == 32:
                self._paused = not self._paused                         # 空格键——暂停/继续
            self._clamp()


'''
对 MuJoCo 原生的 mj_name2id 做了两件事：错误语义转换和类型保证。
    它把 MuJoCo 的"返回负数表示找不到"的 C 风格约定，转成 Python 风格的"抛异常"行为，并附带清晰的错误信息。
通俗类比：
    MuJoCo 的 mj_name2id 找不到名字时只会返回 -1（摇头），你根本不知道它在摇什么头。
    _named_id 这个翻译官会在它摇头时说："你的 XML 里找不到名叫 left_hip 的关节！"
'''
def _named_id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
    """按名称解析 MuJoCo 对象 id，并把找不到名称转换为明确的部署错误。"""
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise DeploymentError(f"MuJoCo XML 中找不到名称为 {name!r} 的 {object_type.name}。")
    return int(object_id)


# MuJoCo 仿真器封装
class MujocoAdapter:
    """封装 MuJoCo model/data，并按 Manifest 名称读写机器人状态。

    所有关节、执行器、传感器、基座和 keyframe 都按名称解析，不依赖
    ``qpos[-N:]`` 或固定 id。对外关节状态统一使用 ``policy_action_order``，写入
    ``data.ctrl`` 前则使用 ``hardware_actuator_order``。
    """
    '''
    输入参数：
        bundle（DeploymentBundle）：部署数据包，包含三个核心信息：
            bundle.mjcf_path：
                MuJoCo XML 机器人模型的文件路径
            bundle.deployment：
                Manifest 中的 deployment 字段字典，包含 policy_action_order、hardware_dof_order、hardware_actuator_order、robot_model、physics_period_s 等关键配置
            bundle.manifest：
                完整的 Manifest 字典（通过 bundle 间接访问）
    '''

    def __init__(self, bundle: DeploymentBundle) -> None:
        self.bundle = bundle
        self.deployment = bundle.deployment
        # ======== [1] 加载物理模型 ===========
        # MjModel（模型）：机器人的物理属性（质量、关节、几何体），不变。
        # MjData（数据）：机器人的实时状态（位置、速度、力），每步更新。
        self.model = mujoco.MjModel.from_xml_path(str(bundle.mjcf_path))    # 从 XML 文件加载机器人物理定义（质量、关节、执行器、传感器）
        self.data = mujoco.MjData(self.model)   # 创建运行时数据容器（位置、速度、传感器读数、外力）

        # ======== [2] 物理时间步校验 ===========
        physics_period = float(self.deployment["physics_period_s"])
        if not np.isclose(self.model.opt.timestep, physics_period, rtol=0.0, atol=1.0e-12):
            raise DeploymentError(
                f"MuJoCo timestep={self.model.opt.timestep}，Manifest physics_period_s={physics_period}。"
            )
        '''
        这段校验比较的是两个不同来源的物理步长：
            self.model.opt.timestep        ←  来自 XML 第 3 行: <option timestep="0.005">
                                            加载 XML 后 MuJoCo 把它存到了 model.opt.timestep

            self.deployment["physics_period_s"]  ←  来自 Manifest JSON（训练导出时写入）
                                                    训练脚本中 env_cfg.sim.dt 的值
        '''

        # ======== [3] 保存关节顺序信息 ===========
        self.policy_order = list(self.deployment["policy_action_order"])    # 策略的关节排列顺序
        self.hardware_order = list(self.deployment["hardware_dof_order"])   # 硬件的关节排列顺序
        robot_model = self.deployment["robot_model"]
        actuator_names = list(self.deployment["hardware_actuator_order"])
        if len(actuator_names) != len(self.hardware_order):
            raise DeploymentError("hardware_actuator_order 与 hardware_dof_order 长度不一致。")

        # ======== [4] 解析策略关节的 ID 和地址 ===========
        # qpos/qvel 是全模型扁平数组，必须通过关节 id 对应的地址读取受控关节。
        self.policy_joint_ids = np.asarray(     # 每个策略关节在 model 中的 ID
            [_named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.policy_order],
            dtype=np.int32,
        )
        for joint_id, joint_name in zip(self.policy_joint_ids, self.policy_order, strict=True):
            joint_type = self.model.jnt_type[joint_id]
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise DeploymentError(f"受控关节 {joint_name!r} 必须是单自由度 hinge 或 slide。")
        '''
        关节类型校验：MuJoCo 支持多种关节类型——free（6 自由度自由体）、ball（3 自由度球关节）、hinge（1 自由度旋转关节）、slide（1 自由度平移关节）。
        这里要求受控关节必须是单自由度的 hinge 或 slide，因为 PD 控制器只能处理单自由度关节。
        '''
        self.policy_qpos_addresses = self.model.jnt_qposadr[self.policy_joint_ids].copy()   # 每个策略关节在 data.qpos 中的起始地址
        self.policy_dof_addresses = self.model.jnt_dofadr[self.policy_joint_ids].copy()     # 每个策略关节在 data.qvel 中的起始地址

        # ======== [5] 解析硬件执行器 ID 并验证传动连接 ===========
        # actuator 名称与关节名称允许不同，因此分别读取两套 order 并验证传动连接。
        self.hardware_actuator_ids = np.asarray(        # 每个硬件执行器在 model 中的 ID
            [_named_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names],
            dtype=np.int32,
        )
        for actuator_id, actuator_name, joint_name in zip(
            self.hardware_actuator_ids, actuator_names, self.hardware_order, strict=True
        ):
            joint_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if int(self.model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise DeploymentError(f"执行器 {actuator_name!r} 没有连接到 Manifest 指定的关节 {joint_name!r}。")

        # ======== [6] 解析传感器和特殊对象 ===========
        self.orientation_sensor_id = _named_id(     # 姿态传感器的 ID
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, robot_model["orientation_sensor"]
        )
        self.angular_velocity_sensor_id = _named_id(    # 角速度传感器的 ID
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, robot_model["angular_velocity_sensor"]
        )
        self.base_joint_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, robot_model["base_joint"])    # 基座关节的 ID
        self.base_body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_model["base_body"])       # 基座刚体的 ID
        self.keyframe_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_KEY, robot_model["keyframe"])          # 初始 keyframe 的 ID

        # ======== [7] 执行初始复位 ===========
        self.reset()    # 把机器人放到初始姿态

    @staticmethod
    def _sensor_data(model: mujoco.MjModel, data: mujoco.MjData, sensor_id: int) -> np.ndarray:
        """从扁平 ``sensordata`` 中按传感器起始地址和维度切出一个副本。"""
        address = int(model.sensor_adr[sensor_id])
        dimension = int(model.sensor_dim[sensor_id])
        return np.asarray(data.sensordata[address : address + dimension]).copy()

    def reset(self) -> None:
        """从 Manifest 指定 keyframe 恢复状态，并调用 ``mj_forward`` 刷新派生量。"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        mujoco.mj_forward(self.model, self.data)

    def joint_state_policy(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``policy_action_order`` 下的关节位置和速度副本。"""
        q = np.asarray(self.data.qpos[self.policy_qpos_addresses]).copy()
        dq = np.asarray(self.data.qvel[self.policy_dof_addresses]).copy()
        return q, dq

    def orientation_wxyz(self) -> np.ndarray:
        """读取基座姿态传感器，返回 MuJoCo 约定的 ``wxyz`` 四元数。"""
        value = self._sensor_data(self.model, self.data, self.orientation_sensor_id)
        if value.shape != (4,):
            raise DeploymentError(f"姿态传感器必须输出 4 维四元数，实际为 {value.shape}。")
        return value

    def angular_velocity(self) -> np.ndarray:
        """读取机器人基座坐标系下的三维角速度。"""
        value = self._sensor_data(self.model, self.data, self.angular_velocity_sensor_id)
        if value.shape != (3,):
            raise DeploymentError(f"角速度传感器必须输出 3 维，实际为 {value.shape}。")
        return value

    @property
    def base_height(self) -> float:
        """返回基座 body 在世界坐标系中的 z 坐标，用于摔倒统计。"""
        return float(self.data.xpos[self.base_body_id, 2])

    def apply_hardware_torque(self, torque: np.ndarray) -> None:
        """把 ``hardware_actuator_order`` 下的力矩写入 ``data.ctrl``。"""
        torque = np.asarray(torque, dtype=np.float64)
        expected_shape = (len(self.hardware_actuator_ids),)
        if torque.shape != expected_shape:
            raise ValueError(f"硬件顺序力矩 shape 应为 {expected_shape}，实际为 {torque.shape}。")
        self.data.ctrl[self.hardware_actuator_ids] = torque

    def step(self) -> None:
        """推进一个 MuJoCo 物理步，并立即检查状态是否出现 NaN/Inf。"""
        mujoco.mj_step(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise FloatingPointError("MuJoCo 状态包含 NaN 或 Inf。")


# 主仿真循环
class Sim2SimRunner:
    """在一个进程中串联 MuJoCo、观测、Encoder、Actor 和 PD 控制。

    一次策略周期的顺序为：读取当前历史和观测 -> 双 ONNX 推理 -> 保存原始 action
    作为下一帧 ``previous_action`` -> 执行 decimation 个物理步 -> 构造并追加新观测。
    """

    '''
    整个 sim2sim 系统的**"总装车间"
        它把之前介绍的所有独立组件（物理引擎、策略网络、观测构造器、PD 控制器、键盘控制、动作延迟、历史窗口）逐一创建出来，
        然后做跨模块的维度一致性校验**，最后把所有状态初始化到"起跑线"上。
    输入参数：
        bundle（DeploymentBundle）：
            部署数据包，包含 Manifest JSON、两个 ONNX 模型路径和 MuJoCo XML 路径。
            它是整个系统的"唯一真相来源"，所有组件都从它读取自己需要的配置。
        runtime（RuntimeConfig）：
            用户在 sim2sim.py 中配置的运行参数。包含固定命令、键盘开关、实时倍率、动作延迟步数、摔倒高度阈值等。
            这些是"运行时可变"的参数，与 Manifest 中"固定不变"的模型参数形成互补。
    '''
    def __init__(self, bundle: DeploymentBundle, runtime: RuntimeConfig) -> None:
        """创建所有运行组件，并核对模型、观测、命令和控制维度。"""
        # =========== [1] 创建五大核心组件 ============
        self.bundle = bundle
        self.runtime = runtime
        self.deployment = bundle.deployment
        self.policy = SplitOnnxPolicy(bundle)   # Encoder + Actor 双 ONNX 推理
        self.adapter = MujocoAdapter(bundle)    # MuJoCo 物理引擎读写
        self.observation_builder = ObservationBuilder(self.deployment)  # 拼接 28 维 proprioception
        self.controller = GenericPDController(self.deployment)  # 动作→力矩的 PD 计算

        # =========== [2] 跨模块维度一致性校验（"合约验证"） ============
        # 当前历史缓冲的首行最旧、末行最新，必须与训练和导出约定一致。
        if self.bundle.manifest.get("history_order") not in (None, "oldest_to_newest"): # 校验 history_order
            raise DeploymentError("当前历史缓冲只支持 oldest_to_newest 顺序。")
        if self.policy.proprioception_dim != self.observation_builder.proprioception_dim:   # 校验 proprioception 维度
            raise DeploymentError("ONNX proprioception 维度与 proprioception_layout 不一致。")
        if self.policy.action_dim != self.controller.action_dim:    # 校验 action 维度
            raise DeploymentError("ONNX action 维度与 policy_action_order 不一致。")
        '''
        维度	                self.policy 的来源	        对比对象的来源
        proprioception_dim	    ONNX 模型的输入 shape	    Manifest proprioception_dim（经过 ObservationBuilder 解析）
        action_dim	            ONNX 模型的输出 shape	    len(policy_action_order)（经过 GenericPDController 解析）
        如果这两个维度对不上，说明 Manifest 和 ONNX 文件不匹配——比如训练时改了观测维度但忘记更新 Manifest。
        '''

        command_order = list(self.deployment["command_order"])
        if self.policy.command_dim != len(command_order):   # 校验 command 维度
            raise DeploymentError("ONNX command 维度与 command_order 不一致。")

        # =========== [3] 创建辅助组件 ============
        self.command = KeyboardCommand(     # 线程安全的命令管理
            runtime.fixed_command,                      # ← 用户配置的固定命令
            command_order,                              # ← Manifest 定义的命令顺序
            self.deployment.get("command_limits", {}),  # ← 训练时的命令范围
            runtime.keyboard_linear_step,
            runtime.keyboard_yaw_step,
        )
        delay_range = self.deployment.get("shared_action_delay_physics_steps", [0, 0])
        self.delay = ActionDelayBuffer(     # 动作延迟 FIFO 队列
            runtime.action_delay_steps,     # ← 用户配置的延迟步数
            self.policy.action_dim,         # ← 8（来自 ONNX 模型）
            delay_range)                    # ← Manifest 定义的允许范围
        self.history = ObservationHistory(  # 10 帧滑动观测窗口
            self.policy.history_samples,    # ← 10（来自 Encoder 模型的输入 shape 的第二维）
            self.policy.proprioception_dim) # ← 28（来自 Actor 模型的 proprioception 输入 shape 的第二维）

        # =========== [4] 初始化状态变量 ============
        self.previous_action = np.zeros(self.policy.action_dim, dtype=np.float32)   # 上一周期的 Actor 输出
        self.current_observation = np.zeros(self.policy.proprioception_dim, dtype=np.float32)   # 当前帧 proprioception
        # trace 按策略步记录，避免在高频物理循环中产生重复的大量数据。
        self.trace_path = runtime.trace_path
        self.trace: dict[str, list[Any]] = {}       # 空调试轨迹字典

        # =========== [5] 执行初始复位 ============
        self.reset()

    '''
    整个 sim2sim 系统中**"物理世界到神经网络语言"的翻译器**。
        它从 MuJoCo 仿真器中读取原始物理数据（关节角度、角速度、基座姿态四元数），打包后交给 ObservationBuilder 拼成策略网络能理解的 28 维 proprioception 向量。
    '''
    def _observe(self) -> np.ndarray:
        """读取最新 MuJoCo 状态，构造一帧 policy_order 下的 proprioception。"""
        q, dq = self.adapter.joint_state_policy()   # 读取关节状态
        return self.observation_builder.build(
            self.adapter.angular_velocity(),        # 读取角速度传感器
            self.adapter.orientation_wxyz(),        # 读取姿态传感器
            q,
            dq,
            self.previous_action,                   # 缓存的上周期动作
        )
    '''
    _observe()
    │
    ├─ [1] 从 MuJoCo 读取关节状态
    │   └─ self.adapter.joint_state_policy()
    │       │
    │       ├─ data.qpos[policy_qpos_addresses] → q [8]
    │       │     ← policy_order 下 8 个关节的当前位置
    │       │       例如: [0.05, 0.26, -0.56, -0.05, 0.26, -0.56, 0.0, 0.0]
    │       │
    │       └─ data.qvel[policy_dof_addresses] → dq [8]
    │             ← policy_order 下 8 个关节的当前速度
    │               例如: [0.01, -0.02, 0.0, -0.01, -0.02, 0.0, 5.2, 5.2]
    │
    ├─ [2] 从 MuJoCo 读取基座姿态传感器
    │   └─ self.adapter.angular_velocity()
    │       └─ data.sensordata[gyro_address:gyro_address+3] → [3]
    │             ← 基座三轴角速度（IMU 陀螺仪读数）
    │               例如: [0.001, 0.002, -0.001]
    │
    ├─ [3] 从 MuJoCo 读取基座姿态四元数
    │   └─ self.adapter.orientation_wxyz()
    │       └─ data.sensordata[quat_address:quat_address+4] → [4]
    │             ← 基座姿态四元数 wxyz
    │               例如: [0.999, 0.01, 0.02, 0.0]
    │
    ├─ [4] 读取上一周期动作缓存
    │   └─ self.previous_action → [8]
    │         ← 上一策略周期的 Actor 原始输出（未经 PD 处理）
    │           例如: [0.1, -0.2, 0.0, -0.1, -0.2, 0.0, 0.5, 0.5]
    │
    └─ [5] 交给 ObservationBuilder 拼接
        └─ self.observation_builder.build(
            angular_velocity=[3],      ← 基座角速度
            quaternion_wxyz=[4],       ← 姿态四元数
            joint_position=[8],        ← 关节位置
            joint_velocity=[8],        ← 关节速度
            previous_action=[8],       ← 上一动作
        )
        │
        ├─ 计算投影重力: quaternion → gravity [3]
        │     g_body = R(q).T @ [0, 0, -1]
        │
        ├─ 按 proprioception_layout 逐项拼接:
        │     term[0] = angular_velocity * 0.25       → 3 维
        │     term[1] = projected_gravity * 1.0       → 3 维
        │     term[2] = (q[leg_joints] - default_q) * 1.0  → 6 维
        │     term[3] = dq * 0.05                     → 8 维
        │     term[4] = previous_action * 1.0         → 8 维
        │
        └─ concatenate → [28] float32
    '''

    def reset(self) -> None:
        """同步复位 MuJoCo、动作延迟、上一动作和 Encoder 历史。"""
        self.adapter.reset()                            # ← MuJoCo 复位到 keyframe
        self.delay.reset()                              # ← 延迟队列清零
        self.previous_action.fill(0.0)                  # ← 上一动作归零
        self.history.reset()                            # ← 历史窗口清零
        self.current_observation = self._observe()      # ← 读取首帧观测
        # 首次 append 会复制当前帧填满窗口，与训练端 reset 后的历史语义一致。
        self.history.append(self.current_observation)   # ← 复制首帧填满 10 行窗口

    def _record(
        self,
        estimated_velocity: np.ndarray,
        action: np.ndarray,
        torque_policy: np.ndarray,
    ) -> None:
        """按策略步缓存可选调试轨迹，用于离线检查观测、动作和力矩。"""
        if self.trace_path is None:
            return
        values = {
            "time": float(self.adapter.data.time),
            "base_height": self.adapter.base_height,
            "qpos": self.adapter.data.qpos.copy(),
            "qvel": self.adapter.data.qvel.copy(),
            "command": self.command.value(),
            "proprioception": self.current_observation.copy(),
            "estimated_base_linear_velocity": estimated_velocity.copy(),
            "action": action.copy(),
            "torque_policy": torque_policy.copy(),
        }
        for key, value in values.items():
            self.trace.setdefault(key, []).append(value)

    def _save_trace(self) -> None:
        """将内存中的调试轨迹保存为压缩 NPZ；未配置路径时不执行。"""
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(values) for key, values in self.trace.items()}
        np.savez_compressed(self.trace_path, **arrays)

    '''
    整个 sim2sim 系统的**"指挥中心"**
        它执行仿真主循环，按照固定的节奏（策略周期 → 物理执行 → 观测构造 → 实时同步）不断迭代，直到仿真时长耗尽或用户关闭可视化窗口。
        运行结束后，它返回一份运行统计摘要，包括策略步数、物理步数、摔倒情况等。
    '''
    def run(self) -> RunSummary:
        """执行仿真主循环，直到达到时长或 viewer 被关闭。

        ``control_period_s = physics_period_s * decimation``。每个策略周期只执行一次
        Encoder 和 Actor，而其 action 会在 decimation 个物理步中保持；每个物理步
        仍重新读取 q/dq、应用动作延迟并计算 PD 力矩。
        """
        '''
        输出（返回值）：
            RunSummary：一个不可变数据类，包含以下字段：
                字段	            类型	    含义
                policy_steps	    int	    执行的策略周期总数
                physics_steps	    int	    执行的 MuJoCo 物理步总数
                resets	            int	    用户手动复位的次数
                min_base_height	    float	运行期间基座的最低高度（米）
                final_base_height	float	运行结束时基座的高度（米）
                fell	            bool	是否曾低于 fall_height_m 阈值
        '''

        # ========== [1] 运行参数校验 =============
        if self.runtime.realtime_factor <= 0.0:
            raise ValueError("REALTIME_FACTOR 必须大于零。")
        if self.runtime.duration_s < 0.0:
            raise ValueError("DURATION_S 不能小于零。")
        if self.runtime.headless and self.runtime.duration_s == 0.0:
            raise ValueError("HEADLESS=True 时 DURATION_S 必须大于零。")
        if self.runtime.keyboard_enabled and self.runtime.headless:
            raise ValueError("键盘控制依赖 MuJoCo viewer，不能与 HEADLESS=True 同时使用。")

        # ========== [2] 启动 MuJoCo viewer（可选） =============
        # passive viewer 不接管仿真循环；物理推进和实时同步仍由本类负责。
        viewer = None
        if not self.runtime.headless:
            from mujoco import viewer as mujoco_viewer

            callback = self.command.key_callback if self.runtime.keyboard_enabled else None
            viewer = mujoco_viewer.launch_passive(self.adapter.model, self.adapter.data, key_callback=callback)
            '''
            launch_passive：
                MuJoCo 的"被动模式"viewer。
                它不接管仿真循环，只负责渲染和接收键盘事件。
                物理推进仍然由我们的主循环控制，viewer 只在调用 viewer.sync() 时更新画面。
            '''

        # ========== [3] 读取仿真周期参数 =============
        physics_period = float(self.deployment["physics_period_s"])
        control_period = float(self.deployment["control_period_s"])
        decimation = int(self.deployment["decimation"])
        if decimation <= 0 or not np.isclose(control_period, physics_period * decimation, rtol=0.0, atol=1.0e-12):
            raise DeploymentError("control_period_s 必须等于 physics_period_s * decimation。")

        # ========== [4] 计算最大物理步数 =============
        max_physics_steps = None
        if self.runtime.duration_s > 0.0:
            max_physics_steps = int(round(self.runtime.duration_s / physics_period))

        # ========== [5] 初始化统计变量 =============
        policy_steps = 0
        physics_steps = 0
        resets = 0
        min_height = self.adapter.base_height
        fell = False

        # ========== [6] 主循环 =============
        try:
            while max_physics_steps is None or physics_steps < max_physics_steps:
                if viewer is not None and not viewer.is_running():
                    break
                if self.command.consume_reset():
                    self.reset()
                    resets += 1
                if self.command.is_paused():
                    if viewer is not None:
                        viewer.sync()   # 即使仿真暂停，也需要刷新 viewer 画面，否则窗口会卡住
                    time.sleep(0.01)    # 防止空转消耗 CPU
                    continue

                tick_start = time.perf_counter()    # 记录周期起始时间

                # ================================================================
                # 阶段 1：策略推理（一个策略周期只执行一次）
                # ================================================================
                # 数据流：history [1,10,28] → Encoder → estimated velocity [1,3]
                #         estimated velocity + proprioception [1,28] + command [1,3] → Actor → action [1,8]
                command = self.command.value()
                estimated_velocity, action = self.policy.infer(
                    self.history.batched(),
                    self.current_observation[None, :],
                    command[None, :],
                )
                # previous_action 保存的是 Actor 原始输出（下一帧观测会用到它），不是经过延迟缓冲后的实际控制动作。
                self.previous_action = action.copy()
                last_torque_policy = np.zeros(self.policy.action_dim, dtype=np.float64)

                # ================================================================
                # 阶段 2：物理执行（同一个 action 在 decimation 个物理步中保持）
                # ================================================================
                for _ in range(decimation):
                    # 如果 delay_steps=2: 当前 action 入队，弹出 2 步前的旧 action
                    delayed_action = self.delay.apply(action)       # → delayed_action [8]
                    # 从 data.qpos/qvel 中按预计算地址读取
                    q, dq = self.adapter.joint_state_policy()       # → q [8], dq [8]
                    # position 关节: kp*(q_target-q) + kd*(0-dq)
                    # velocity 关节: kd*(dq_target-dq)
                    last_torque_policy = self.controller.compute_policy_torque(delayed_action, q, dq)   # → torque [8]
                    # 重排为 hardware_dof_order
                    hardware_torque = self.controller.to_hardware_order(last_torque_policy) # → hardware_torque [8]
                    # 写入 data.ctrl[actuator_ids]
                    self.adapter.apply_hardware_torque(hardware_torque)
                    # mujoco.mj_step(model, data)  ← 真正的物理推进
                    self.adapter.step()
                    physics_steps += 1
                    min_height = min(min_height, self.adapter.base_height)
                    fell = fell or self.adapter.base_height < self.runtime.fall_height_m
                    if max_physics_steps is not None and physics_steps >= max_physics_steps:
                        break

                # ================================================================
                # 阶段 3：观测构造（物理推进完成后，为下一个策略周期准备）
                # ================================================================
                # 读取最新的 MuJoCo 状态，拼接成 28 维 proprioception，追加到历史窗口（10 帧 oldest-to-newest 滑动窗口）
                self.current_observation = self._observe()
                self.history.append(self.current_observation)
                self._record(estimated_velocity, action, last_torque_policy)
                policy_steps += 1
                if viewer is not None:
                    viewer.sync()   # 每个策略周期刷新一次画面
                    '''
                    为什么不在每个物理步刷新？
                        因为渲染比物理计算慢得多，20Hz 的策略周期刷新已经足够流畅，200Hz 刷新浪费性能。
                    '''

                # ================================================================
                # 阶段 4：实时同步（按 realtime_factor 控制墙钟速度）
                # ================================================================
                # 这步只调整 Python 进程的睡眠时间，不改变 MuJoCo 的物理步长或策略控制周期。
                # realtime_factor=2.0 意味着每个策略周期只睡一半的时间，让仿真跑得比现实快。
                sleep_time = control_period / self.runtime.realtime_factor - (time.perf_counter() - tick_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
        # ========== [7] finally 清理 =============
        finally:
            if viewer is not None:
                viewer.close()
                # launch_passive 使用异步 GLFW 线程，留出短暂时间完成窗口资源回收。
                time.sleep(0.25)
            self._save_trace()

        # ========== [8] 返回 运行摘要 =============
        return RunSummary(
            policy_steps=policy_steps,
            physics_steps=physics_steps,
            resets=resets,
            min_base_height=min_height,
            final_base_height=self.adapter.base_height,
            fell=fell,
        )


def run_simulation(runtime: RuntimeConfig) -> RunSummary:
    """直接入口：加载部署包，创建编排器并运行单进程 sim2sim。"""
    bundle = load_deployment(runtime.model_dir, runtime.mjcf_path)
    return Sim2SimRunner(bundle, runtime).run()
