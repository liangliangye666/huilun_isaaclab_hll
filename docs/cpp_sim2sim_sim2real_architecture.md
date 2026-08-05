# L5A C++ Sim2Sim / Sim2Real 架构与迁移索引

## 1. 文档范围与当前状态

本文分析 `origin_code/sim2sim_cpp_and_sim2real/`。该目录从旧 Gym 工程
`/mnt/isaacdata/马哥代码20260804/gac-robotics-mgc-y4a-rl/` 搬入，目的是保留原有
C++ MuJoCo Sim2Sim 和 RK3588 Sim2Real 框架，再逐步接入本仓库的 IsaacLab 训练导出。

必须先区分两个事实：

- C++ 工程内部的 8 关节顺序已经是 `[左三腿, 左轮, 右三腿, 右轮]`，与当前 IsaacLab 策略一致。
- C++ 工程仍使用旧 Gym 网络与观测契约，不能直接加载当前 IsaacLab 导出的模型。

本文是当前本地副本的静态源码索引。本次没有完成 C++ 全量构建、MuJoCo 闭环运行、
RK3588 部署或 500 Hz 实机抖动测试。

## 2. 一句话架构

`sim.cc` 和 `stand_mode.cc` 是同一个控制核心的两个运行时适配器：

```text
                         +-------------------+
MuJoCo state ----------> |                   | ----------> MuJoCo torque
                         | RobotModel        |
                         |   -> FSM          |
hardware/IMU feedback -> |      -> RL        | ----------> motor commands
                         |                   |
                         +-------------------+
```

- `sim.cc`：拥有 MuJoCo、渲染和仿真线程，把 `mjData` 适配成 `RobotModel` 状态。
- `stand_mode.cc`：暴露 initialize/step/terminate，由外部实机进程周期调用。
- `control/`：两条路径共用的 RobotModel、FSM、RL、PD、IMU 和估计器实现。

## 3. 目录职责

| 路径 | 作用 | 阅读优先级 |
|---|---|---:|
| `platforms/l5a/sim.cc` | C++ MuJoCo 入口、物理线程、ROS executor、控制回调 | 高 |
| `platforms/l5a/stand_mode.cc` | 实机 ABI 适配、手柄命令、电机模式和输出 | 高 |
| `platforms/l5a/control/robot_model.*` | MuJoCo/实机状态统一、Pinocchio 状态与动力学 | 高 |
| `platforms/l5a/control/fsm.*` | 安全检查、RL/EDamping 状态切换 | 高 |
| `platforms/l5a/control/rl.*` | 观测、历史、TorchScript 推理、动作和 PD | 高 |
| `platforms/l5a/control/*.yaml` | 仿真/实机 PD、缩放、decimation、模型文件名 | 高 |
| `platforms/l5a/control/module/` | 旧 Gym TorchScript 模型 | 高 |
| `platforms/l5a/deploy/standMode_types.h` | 外部实机调用的输入/输出 ABI 数据结构 | 高 |
| `platforms/l5a/deploy/drivers/` | ARM64 驱动程序、EtherCAT/手柄配置和板端应用 | 中 |
| `sim/model/l5a/` | C++ MuJoCo 使用的 XML、URDF、mesh 和 ROS 消息 | 高 |
| `sim/mujoco_simulate/` | MuJoCo simulate UI 外壳 | 低 |
| `publisher/`、`launch/`、`tools/` | ROS 发布、启动文件、手柄和辅助工具 | 中 |
| `third_party/` | x86/ARM 预编译依赖和上游源码镜像 | 默认跳过 |

`README_GAC.md` 主要描述旧 Gym 训练流程，其中部分部署说明已经落后于当前源码。
当前 `rl.cc` 明确通过 LibTorch 加载 TorchScript，不能按 README 中“不依赖第三方推理库”
的文字理解现有 L5A 实现。

## 4. 构建模式和运行资产

顶层 `CMakeLists.txt` 根据处理器选择默认模式：

| 平台 | CMake 模式 | 主要产物 |
|---|---|---|
| x86_64 | `SIM_ENABLE=ON` | `sim_l5a` MuJoCo 可执行程序 |
| aarch64 | `PHYSICS_ENABLE=ON` | `stand_mode_lib` 和 `stand_mode_test` |

共同约束：

- C++17、Release、共享库构建。
- `platforms/l5a/CMakeLists.txt` 将 `Torch_DIR` 固定为 `/usr/local/torch/share/cmake/Torch`。
- 依赖 ROS2/ament、LibTorch、Eigen、Pinocchio、yaml-cpp、glog；仿真额外依赖 MuJoCo/GLFW。
- `scripts/set_env.sh` 设置 `PROJECT_ROOT_DIR`，控制器用它拼接 YAML、模型和 URDF 路径。
- C++ MuJoCo XML 为 `sim/model/l5a/xml/l5aurdf20260521.xml`，物理步长 `0.002 s`。
- 该 XML 的 8 个 motor 使用固定硬件顺序，并额外包含 `wheel_left_site`、
  `wheel_right_site`，供 `sim.cc` 记录轮高度。

板端 `deploy/drivers/` 中的主要程序是 ARM aarch64 ELF，不是可移植的算法源码。
迁移或构建前必须重新核对目标板 ABI、动态库、EtherCAT 配置和外部调用程序。

## 5. 统一关节顺序

`robot_model.h` 中的 `Joints` 枚举、MuJoCo actuator、YAML 参数和实机字段映射均采用：

```text
0 left_hip_roll_joint
1 left_hip_pitch_joint
2 left_knee_joint
3 left_wheel_joint
4 right_hip_roll_joint
5 right_hip_pitch_joint
6 right_knee_joint
7 right_wheel_joint
```

因此迁移到当前 IsaacLab 策略时不需要动作重排。仍需保证状态、默认角、Kp/Kd、
力矩限制和实机反馈映射也使用同一顺序。

## 6. 共享控制核心

### 6.1 RobotModel

关键入口：

- `RobotModel::UpdateMujocoJointStates()`：读取 `mjData`，将 free-base 和关节状态写入
  `q_rpy`、`q_pino`、`qdot`，并把基座速度旋转到本体坐标系。
- `RobotModel::UpdateRealJointStates()`：从 `standmode_input_t` 读取 IMU 和 8 关节反馈，
  写入同一套内部状态。
- `RobotModel::UpdateModel()`：调用 `AddFrames()`、Pinocchio 运动学、动力学和状态发布。

两种运行时最终都向 FSM 提供相同的数据形态：

```text
q_rpy / q_pino / qdot
base orientation and angular velocity
desired vx / vy / yaw rate
8 joint position and velocity
```

### 6.2 FSM

`FSM::Run()` 每个控制周期先执行 `CheckSafety()`：

```text
safe   -> FsmId::Rl       -> RL::Run()
unsafe -> FsmId::RlEDamp  -> RL::RunEDamp()
```

FSM 对外保存：

- `tau()`：8 维内部 PD 力矩；
- `pos()`、`vel()`：8 维位置/速度目标；
- `pos_fb_kp_`、`pos_fb_kd_`：实机力位混合模式增益。

### 6.3 RL 与 PD

旧 C++ 策略动作缩放为：

```text
pos_ref = action * 0.25
vel_ref = action * 0.5
```

随后把轮关节位置目标清零、腿关节速度目标清零：

```text
腿：default_pos + position action，目标速度为 0
轮：位置参考为 0，使用 velocity action
```

`PdController` 根据实际位置/速度生成 8 维 `tau_`。Sim2Sim 把全部 8 维力矩写入
MuJoCo；Sim2Real 则把腿和轮拆成不同的电机命令语义。

## 7. C++ Sim2Sim 数据流

### 7.1 启动流程

`sim.cc::main()` 的启动链为：

```text
main
  -> initialize glog and signal handlers
  -> open log.csv
  -> MujocoSimulator
       -> rclcpp::init
       -> ROS MultiThreadedExecutor thread
       -> create MuJoCo Simulate UI
       -> PhysicsThread
       -> main thread blocks in RenderLoop
```

### 7.2 每个物理步

物理线程中的实际顺序是：

```text
mj_step(m, d)
  -> MyController(m, d)
       -> RobotModel::UpdateMujocoJointStates
       -> RobotModel::UpdateModel
       -> FSM::Run
            -> RL::Run or RL::RunEDamp
       -> d->ctrl[i] = tau_cmd[i]
```

由于当前代码先调用 `mj_step()`、后写 `d->ctrl`，本次算出的新力矩从下一个
MuJoCo 物理步开始生效。

### 7.3 线程模型

| 线程 | 创建位置 | 责任 |
|---|---|---|
| 主/UI 线程 | `main` | MuJoCo `RenderLoop`、信号处理 |
| PhysicsThread | `MujocoSimulator` | `mj_step`、控制回调、仿真时间同步 |
| ROS executor | `MujocoSimulator` | ROS2 回调和发布 |
| RL inference thread | `RL` 构造函数 | estimator + actor TorchScript 推理 |
| joystick 内部路径 | `JoyStick` | 仿真命令输入，具体线程归属需结合工具实现 |

## 8. Sim2Real ABI 和数据流

`stand_mode.cc` 没有 `main()`，它提供三个给外部板端程序调用的函数：

```text
standMode_initialize(output, input)
standMode_step(output, input)
standMode_terminate()
```

全局 `RobotModel`、`FSM` 和 RL 推理线程在共享库加载/全局对象构造阶段创建。
`standMode_initialize()` 只重新初始化 RobotModel/FSM 的一部分状态，并不会重新构造
RL 对象或完整清空历史与推理线程状态。

外部周期调用者拥有主循环；`RobotModel` 和步态代码把控制周期写死为 `0.002 s`，
所以它期望名义 500 Hz 调用，但实际频率由外部进程决定。`stand_mode.cc` 自己不创建
500 Hz 定时线程。
一次 `standMode_step()` 的主链为：

```text
standmode_input_t
  -> set motor enable/mode
  -> RobotModel::UpdateRealJointStates
  -> RobotModel::UpdateModel
  -> handle gamepad and smooth command
  -> FSM::Run
  -> standmode_output_t.joints_cmd
  -> observed_value debug channels
```

### 8.1 输入 ABI

`standmode_input_t` 主要包含：

- `Handle_signals`：手柄轴和按键；
- `IMU_signals`：姿态、角速度和加速度；
- `joints_status`：关节位置、速度、电流等反馈；
- 机械臂、质量传感器、导航和电池等兼容字段。

### 8.2 输出到电机

正常模式下：

| 关节 | operation_mode | 下发内容 |
|---|---:|---|
| 6 个腿关节 | 11，力位混合 | `pos_cmd` + KP/KD，速度和前馈力矩置零 |
| 2 个轮关节 | 4，力矩 | 内部速度 PD 算出的 `tau_cmd` |

紧急状态会调用 `setEmergencyParameters()`，把 8 个关节改为 mode 3。其物理效果和
驱动侧解释必须在目标硬件上确认，不能只根据枚举数字推断安全性。

## 9. 旧 C++ 推理线程与张量契约

### 9.1 旧观测

`rl.cc` 固定 `num_obs_=32`、`hist_len_=10`、`num_est_=3`：

```text
base angular velocity                 3   obs[0:3]
projected gravity                    3   obs[3:6]
vx, vy, yaw command                  3   obs[6:9]
fixed target height                  1   obs[9]
six leg position errors              6   obs[10:16]
all joint velocities                 8   obs[16:24]
previous actions                     8   obs[24:32]
                                         --------
                                         32
```

网络契约：

```text
history: 10 x 32 -> flattened [1, 320]
estimator: [1, 320] -> [1, 3]
actor: concat(current obs 32, estimated velocity 3) -> [1, 35]
policy output: [1, 8]
```

第一次触发推理时，当前 observation 被复制 10 次填满历史；之后按 oldest-to-newest 滑动。

### 9.2 异步 latest-result-hold

500 Hz 控制线程每 `decimation=10` 个周期尝试提交一次推理，即名义 50 Hz：

```text
control thread                         inference thread
--------------                         ----------------
build obs/history
lock SharedData
if inference_ready == false:
  copy inputs
  inference_ready = true  -----------> wake condition_variable
unlock                                   copy inputs
                                         estimator + actor
                                         write actions
                                         has_new_result = true
consume new result if available <------
hold previous action otherwise
```

如果上一轮推理还忙，新的请求会被跳过；控制循环继续保持最近一次动作。因此它不是
“每 20 ms 必然得到一个新动作”，也没有仅凭源码即可证明的硬实时保证。

## 10. 与当前 IsaacLab 导出的差异

当前真相来源是 `docs/l5a_wf_training.md` 和导出目录中的 `policy_manifest.json`。

| 契约 | 旧 C++ 基线 | 当前 IsaacLab |
|---|---:|---:|
| 物理周期 | `0.002 s`，500 Hz | `0.005 s`，200 Hz |
| decimation | 10 | 4 |
| 策略频率 | 50 Hz | 50 Hz |
| 单帧 proprioception | 32 | 28 |
| history | `[1,320]` flatten | `[1,10,28]` |
| estimator | 单输入 `[1,320] -> [1,3]` | 单输入 `[1,10,28] -> [1,3]` |
| actor | 单输入 `[1,35] -> [1,8]` | 三输入 `3 + 28 + 3 -> 8` |
| command | 包在 32 维 obs | Actor 独立 `[N,3]` 输入 |
| 固定 height | obs 中 1 维 | policy/history 中没有 |
| 轮动作 scale | 0.5 | 1.0 |
| 关节顺序 | 左腿、左轮、右腿、右轮 | 相同 |
| 参数契约 | 两份手写 YAML | 导出的 `policy_manifest.json` |

当前 TorchScript schema 已核对为：

```text
velocity_estimator.pt.forward(observation_history)
policy.pt.forward(estimated_base_linear_velocity, proprioception, commands)
```

所以只替换 `control/module/*.pt` 会在 shape 或 forward 参数上失败。

## 11. 从旧 Gym 迁移到当前 IsaacLab 的修改中心

建议保持外壳与核心边界，优先替换部署契约层：

1. `rl.h`：将固定维度、共享输入和模型接口改为当前 28/10/3/8 契约。
2. `rl.cc`：重写 observation layout、历史 tensor shape、estimator/actor 调用和轮 scale。
3. YAML/Manifest：以 `policy_manifest.json` 为部署真相，避免再次手工复制时序、缩放和顺序。
4. 时序：C++ MuJoCo XML/RobotModel 从 2 ms 调整到 5 ms，并把 decimation 从 10 调整到 4。
5. 模型文件：使用当前导出的 `velocity_estimator.pt` 和 `policy.pt`，不要沿用旧文件名含义。
6. Sim2Real 输出：保留 6 腿位置/KP/KD + 2 轮速度 PD 力矩的物理语义。
7. 验证顺序：先 C++ 离线 tensor 对拍，再 C++ MuJoCo，最后才进入实机低风险流程。

可大体复用的部分：

- `RobotModel` 的 MuJoCo/实机状态适配框架；
- `FSM` 和 EDamping 框架，但安全逻辑需要审计；
- `standMode_types.h` ABI 和手柄/电机字段映射；
- MuJoCo UI、URDF/XML/mesh 和构建骨架。

## 12. 已发现的源码风险点

这些是静态审计提示，不等于已经在运行中复现：

- `RobotModel::UpdateModel()` 每周期调用 `AddFrames()`，反复向 Pinocchio model 添加 frame
  并重建 data，可能产生开销和模型膨胀。
- `FSM::CheckSafety()` 中速度越限分支没有设置 `EDamp_signal_=true`；力矩检查使用的是
  FSM 已保存的上一周期 `tau_`。
- `robot_model.emergency_` 被置位后缺少清晰的在线恢复路径。
- `standMode_initialize()` 不完整重置 RL history/action/推理线程状态。
- `standMode_terminate()` 是空函数，全局线程的停止依赖进程/全局对象析构。
- `deploy/standMode.h` 仍带旧生成代码痕迹和不一致声明（包括 `standmode_inputvoid`、
  `y4a::RobotModel`）；当前 `stand_mode.cc` 没有包含它，测试程序反而直接包含
  `stand_mode.cc`。迁移实机 ABI 时应先建立一份与实现一致的干净公共头文件。
- Torch 推理、mutex、动态 Eigen/Pinocchio 操作和 CSV/控制台 I/O 都可能影响实时抖动，
  需要目标板测量。
- 构建脚本会删除 `install` 或 `install_arm64`；运行前应确认工作目录和目标路径。

## 13. 快速检索入口

```bash
# C++ 两个运行时入口
rg -n "MyController|PhysicsLoop|int main|standMode_initialize|standMode_step" \
  origin_code/sim2sim_cpp_and_sim2real/platforms/l5a

# RL 观测、线程和动作
rg -n "num_obs_|InferenceLoop|SharedData|decimation_|pos_ref|vel_ref" \
  origin_code/sim2sim_cpp_and_sim2real/platforms/l5a/control

# 实机 ABI 和电机模式
rg -n "standmode_input_t|standmode_output_t|operation_mode|torque_cmd" \
  origin_code/sim2sim_cpp_and_sim2real/platforms/l5a

# 构建目标和依赖
rg -n "SIM_ENABLE|PHYSICS_ENABLE|add_executable|add_library|Torch_DIR" \
  origin_code/sim2sim_cpp_and_sim2real --glob 'CMakeLists.txt'
```
