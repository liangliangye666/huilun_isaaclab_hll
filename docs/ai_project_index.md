# Huilun IsaacLab AI 项目索引

## 1. 项目定位

本仓库把四条链路放在同一个工程中：

```text
IsaacLab 训练与导出
        |
        +-> Python MuJoCo Sim2Sim
        |
        +-> C++ MuJoCo Sim2Sim（IsaacLab WF-Flat 策略）
        |
        +-> C++ Sim2Real（IsaacLab WF-Flat 策略）
```

当前训练任务是 L5A wheel-legged WF flat locomotion，采用 Manager-Based IsaacLab、
RSL-RL PPO 和一个受监督的 10 帧基座速度 Encoder。

## 2. 回答问题前的固定查找顺序

1. 先在 `source/**/tasks/**/__init__.py` 查找 `gym.register`。
2. 从注册项的 `env_cfg_entry_point` 打开环境配置。
3. 从 `rsl_rl_cfg_entry_point` 打开 Runner/Actor-Critic/PPO 配置。
4. 根据问题进入 asset、MDP term、训练脚本、导出器或部署端。
5. 若涉及时序、shape、关节顺序，交叉核对 `policy_manifest.json`，不要只看注释。

当前注册入口：

```text
source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/__init__.py
```

| Gym ID | 环境配置 | 算法配置 | 用途 |
|---|---|---|---|
| `Huilun-L5A-WF-Flat-v0` | `L5AWFFlatEnvCfg` | `L5AWFPPORunnerCfg` | 训练 |
| `Huilun-L5A-WF-Flat-Play-v0` | `L5AWFFlatEnvCfg_PLAY` | `L5AWFPPORunnerCfg` | 评估和导出 |

导入链为：

```text
import huilun_isaaclab
  -> huilun_isaaclab.tasks
  -> import_packages(...)
  -> tasks.locomotion.l5a.__init__
  -> gym.register(...)
  -> scripts/rsl_rl/train.py or play.py calls gym.make(...)
```

注册阶段只绑定字符串配置；`gym.make()` 时才创建 `ManagerBasedRLEnv`。

## 3. 端到端主链

```text
gym.register
  -> wf_flat_env_cfg.py
       -> L5AWFSceneCfg / CommandsCfg / ActionsCfg
       -> ObservationsCfg / EventCfg / RewardsCfg / TerminationsCfg
       -> L5AWFFlatEnvCfg
  -> agents/rsl_rl_ppo_cfg.py
       -> L5AWFPPORunnerCfg
  -> scripts/rsl_rl/train.py
       -> gym.make
       -> RslRlVecEnvWrapper
       -> VelocityEstimatorOnPolicyRunner
       -> VelocityEstimatorActorCritic + VelocityEstimatorPPO
  -> scripts/rsl_rl/play.py
       -> load checkpoint
       -> export_velocity_estimator_policy
       -> policy.pt / policy.onnx
       -> velocity_estimator.pt / velocity_estimator.onnx
       -> policy_manifest.json
  -> deployment runtime
```

## 4. 顶层目录地图

| 路径 | 状态 | 内容 |
|---|---|---|
| `source/huilun_isaaclab/` | 当前主线 | IsaacLab 扩展、任务、asset、MDP 和自定义 RSL-RL |
| `scripts/rsl_rl/` | 当前主线 | train/play/CLI 入口 |
| `scripts/devtools/` | 辅助 | AI 索引生成脚本 |
| `resources/robots/l5a/` | 当前资产 | USD、URDF、XML、mesh、ROS message |
| `sim2sim_mujoco_py/` | 当前部署主线 | Manifest 驱动的 Python MuJoCo + ONNX Runtime |
| `origin_code/sim2sim_mujoco_py/` | 兼容/参考 | 单文件 TorchScript Python Sim2Sim |
| `origin_code/sim2sim_cpp_and_sim2real/` | 当前 C++ 部署 | 保留旧外壳，已适配 WF-Flat 的 28 维观测、10 帧 Encoder 和三输入 Actor |
| `docs/` | 文档真相 | 训练、架构和 AI 索引 |
| `logs/rsl_rl/` | 生成物 | checkpoint、导出模型和 Manifest |
| `outputs/` | 生成物 | Hydra/运行输出，不作为源码入口 |

默认不要递归阅读 `.git/`、`logs/`、`outputs/`、缓存、C++ `third_party/`、编译目录和
二进制模型。只有问题明确涉及某个生成物、ABI 或依赖版本时才进入这些目录。

## 5. 当前训练主线文件索引

### 5.1 机器人和顺序

`source/huilun_isaaclab/huilun_isaaclab/assets/robots/l5a.py`

负责：

- L5A USD 资产、初始姿态、关节限制和 actuator；
- 腿/轮关节名分组；
- `HARDWARE_DOF_NAMES` 和 `WF_POLICY_DOF_NAMES`；
- 标称 `L5A_CFG` 与带动作延迟的 `L5A_WF_CFG`。

当前统一关节顺序：

```text
left_hip_roll_joint
left_hip_pitch_joint
left_knee_joint
left_wheel_joint
right_hip_roll_joint
right_hip_pitch_joint
right_knee_joint
right_wheel_joint
```

### 5.2 环境总配置

`source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/wf_flat_env_cfg.py`

这是回答训练行为问题时最重要的文件，包含：

- `L5AWFSceneCfg`：robot、terrain、contact sensor；
- `CommandsCfg`：速度/heading 命令范围；
- `ActionsCfg`：左腿、左轮、右腿、右轮四个 action term；
- `ObservationsCfg`：policy/history/critic/commands/estimator target；
- `EventCfg`：质量、摩擦、COM、Kp/Kd、零位、IMU bias、reset、push；
- `RewardsCfg`：速度跟踪、姿态、轮距、平滑、能耗和限制；
- `TerminationsCfg`、`CurriculumCfg`；
- `L5AWFFlatEnvCfg`：physics dt、decimation、episode、Manager 聚合；
- `build_l5a_wf_export_metadata()`：checkpoint/export 部署元数据。

### 5.3 自定义 MDP

路径：`source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a/mdp/`

| 文件 | 作用 |
|---|---|
| `actions.py` | 每环境随机默认角的位置 action |
| `observations.py` | IMU bias、特权 torque/acc/body/contact/mass |
| `events.py` | COM、mass/inertia、effort 和 IMU 随机化 |
| `rewards.py` | L5A 轮腿几何、平滑、功率和姿态奖励 |
| `terminations.py` | action 越界终止 |

`mdp/__init__.py` 汇总 IsaacLab 标准 MDP term 和本项目自定义 term，因此配置中常写
`mdp.some_function`。

### 5.4 延迟 actuator

`source/huilun_isaaclab/huilun_isaaclab/actuators/delayed_implicit.py`

在每个物理步对位置、速度和力矩目标应用同一个 per-environment lag。它与策略
`decimation` 是两个不同时间尺度。

## 6. 训练算法和导出索引

### 6.1 Agent 配置

`tasks/locomotion/l5a/agents/rsl_rl_ppo_cfg.py`

定义 `L5AWFPPORunnerCfg`，选择：

- `VelocityEstimatorOnPolicyRunner`；
- `VelocityEstimatorActorCritic`；
- `VelocityEstimatorPPO`；
- policy/history/command/critic/target observation group 名称。

### 6.2 自定义 RSL-RL

路径：`source/huilun_isaaclab/huilun_isaaclab/learning/rsl_rl/`

| 文件 | 关键类/函数 | 责任 |
|---|---|---|
| `estimator_actor_critic.py` | `VelocityEstimatorActorCritic` | Encoder、Actor、Critic 前向和 TensorDict 分组 |
| `estimator_ppo.py` | `VelocityEstimatorPPO` | PPO 之外独立优化 Encoder MSE |
| `estimator_runner.py` | `VelocityEstimatorOnPolicyRunner` | 构造算法、checkpoint metadata 兼容性 |
| `estimator_exporter.py` | `export_velocity_estimator_policy` | 导出 split JIT/ONNX 和 Manifest |
| `velocity_estimator_cfg.py` | 两个 config class | Encoder/PPO 的配置字段 |

### 6.3 运行脚本

| 文件 | 入口作用 |
|---|---|
| `scripts/rsl_rl/train.py` | 启动 Isaac Sim、Hydra 解析、`gym.make`、runner.learn |
| `scripts/rsl_rl/play.py` | 加载 checkpoint、播放、导出、可选调试记录 |
| `scripts/rsl_rl/cli_args.py` | RSL-RL CLI 到配置的映射 |
| `scripts/list_envs.py` | 查看 Gym registry 中的项目任务 |
| `scripts/zero_agent.py` / `random_agent.py` | 环境接口冒烟测试 |

## 7. 当前训练/部署硬契约

更完整说明见 `docs/l5a_wf_training.md`。

### 7.1 时序

```text
physics dt      = 0.005 s = 200 Hz
decimation      = 4
control period  = 0.020 s = 50 Hz
history         = 10 frames, oldest -> newest
```

### 7.2 网络

```text
proprioception:      [N, 28]
observation_history: [N, 10, 28]
commands:            [N, 3]
estimator target:    [N, 3]  # training only

velocity_estimator.pt(history) -> [N, 3]
policy.pt(estimated_velocity, proprioception, commands) -> [N, 8]
```

28 维布局：

```text
base angular velocity    3
projected gravity        3
six leg position errors  6
all joint velocities     8
previous action          8
                         --
                         28
```

### 7.3 动作语义

```text
indices 0,1,2,4,5,6: leg position, scale 0.25 + default position
indices 3,7:           wheel velocity, scale 0.5
```

不要把 8 维 action 全部解释为位置目标，也不要在当前策略和硬件之间再次换序。

## 8. Python MuJoCo 部署索引

### 8.1 当前通用版

路径：`sim2sim_mujoco_py/`

```text
sim2sim.py
  -> src/model.py          Manifest + split ONNX
  -> src/observation.py    28-dim obs + 10-frame history
  -> src/control.py        6 position + 2 velocity PD
  -> src/simulator.py      MuJoCo adapter and main loop
```

它以导出目录中的 `policy_manifest.json` 为运行契约，适合当前 IsaacLab 导出。

### 8.2 单文件兼容版

`origin_code/sim2sim_mujoco_py/sim2sim_l5a.py`

保留旧脚本形态，使用两个 TorchScript 文件。当前约定策略、MuJoCo 和硬件均直接采用
统一关节顺序，不设置运行时换序层。

## 9. C++ Sim2Sim / Sim2Real 索引

路径：`origin_code/sim2sim_cpp_and_sim2real/`

详细阅读 [C++ Sim2Sim/Sim2Real 架构文档](cpp_sim2sim_sim2real_architecture.md)。

最重要的入口：

- `platforms/l5a/sim.cc`：C++ MuJoCo executable；
- `platforms/l5a/stand_mode.cc`：实机共享库 ABI；
- `platforms/l5a/control/robot_model.*`：状态适配；
- `platforms/l5a/control/fsm.*`：安全/FSM；
- `platforms/l5a/control/rl.*`：当前 WF-Flat TorchScript 推理契约；
- `platforms/l5a/deploy/standMode_types.h`：板端输入输出结构。

当前状态：关节顺序无需换序；C++ 已使用当前导出的 `velocity_estimator.pt`、`policy.pt`
和两份手写部署 YAML。C++ MuJoCo 与真机刻意保留旧 2 ms × 10 的 500 Hz 底层、
50 Hz 策略时序，而 Python MuJoCo 保持训练一致的 5 ms × 4。

构建入口仍是小工程自己的脚本：Docker 内运行 `./scripts/build_and_install.sh` 生成
`install/bin/sim_l5a`，运行 `./scripts/arm64_build_and_install.sh` 只生成
`install_arm64/`。换模型时手工覆盖 `platforms/l5a/control/module/` 中的
`policy.pt`、`velocity_estimator.pt`、`policy_manifest.json`，随后重新构建安装。

2026-08-06 已验证新 TorchScript shape/SHA256、x86 配置编译安装和模型预热/异步前向，
也已完成 ARM64 交叉编译并确认 aarch64 产物及 Torch 动态依赖。短时 MuJoCo 已加载
ROS、XML 和 MuJoCo 3.2.2，但连续闭环稳定性、旧 UI/ROS 清理路径以及 RK3588 板端
LibTorch 兼容性仍未验证；不能把本地构建通过解释为实机验证通过。

## 10. 资产和生成物

### 10.1 当前资产

`resources/robots/l5a/`：

- `usd/`：IsaacLab/PhysX 资产；
- `urdf/`：机器人结构和 Pinocchio/其他工具输入；
- `xml/`：Python MuJoCo 模型；
- `meshes/`：几何模型；
- `msg/`、`launch/`：历史 ROS 兼容资源。

C++ 搬入工程还有一份独立的 `sim/model/l5a/`。修改模型时不要默认两份会自动同步。

### 10.2 日志与导出

典型导出目录：

```text
logs/rsl_rl/l5a_wf_flat/<run>/exported/
  policy.pt
  policy.onnx
  velocity_estimator.pt
  velocity_estimator.onnx
  policy_manifest.json
```

`logs/` 和 `outputs/` 会随运行变化。回答“当前代码怎么定义”时优先看 source；回答
“某个模型实际是什么契约”时读取该模型旁边的 Manifest。

## 11. 按问题快速定位

| 问题 | 第一入口 | 继续追踪 |
|---|---|---|
| 有哪些训练任务？ | `tasks/locomotion/l5a/__init__.py` | `gym.register` kwargs |
| 环境每步发生什么？ | `wf_flat_env_cfg.py:L5AWFFlatEnvCfg` | Managers、MDP term |
| action 顺序/类型？ | `assets/robots/l5a.py` | `ActionsCfg`、Manifest |
| observation shape？ | `ObservationsCfg` | Actor-Critic、exporter |
| reward 为什么这样？ | `RewardsCfg` | `mdp/rewards.py` |
| 随机化在哪里？ | `EventCfg` | `mdp/events.py`、actuator |
| PPO/Encoder 如何更新？ | `rsl_rl_ppo_cfg.py` | `estimator_ppo.py`、runner |
| checkpoint 为什么拒绝？ | `estimator_runner.py` | deployment metadata diff |
| 导出文件怎么生成？ | `scripts/rsl_rl/play.py` | `estimator_exporter.py` |
| Python sim2sim 数据流？ | `sim2sim_mujoco_py/sim2sim.py` | `src/` 四个模块 |
| C++ sim2sim 线程？ | `platforms/l5a/sim.cc` | C++ 架构文档 |
| 实机输入输出？ | `stand_mode.cc` | `standMode_types.h` |

## 12. 常用检索命令

```bash
# 必须优先做：定位任务注册
rg -n "gym\.register" source

# 环境配置和 Manager term
rg -n "^class |@configclass|ObsTerm|RewTerm|EventTerm|DoneTerm" \
  source/huilun_isaaclab/huilun_isaaclab/tasks/locomotion/l5a

# 训练/播放主链
rg -n "gym\.make|RslRlVecEnvWrapper|OnPolicyRunner|export" scripts/rsl_rl

# 当前模型的部署契约
rg -n "physics_period_s|decimation|policy_action_order|proprioception_layout" \
  logs/rsl_rl/l5a_wf_flat/*/exported/policy_manifest.json

# 部署侧关节、观测和力矩
rg -n "joint_order|action_dim|proprioception|compute_torque|data.ctrl" \
  sim2sim_mujoco_py origin_code/sim2sim_mujoco_py
```

## 13. 维护规则

以下变化发生后应同步更新本文档和相关专项文档：

- 增删 `gym.register` 任务；
- 改变 observation/action shape、顺序或语义；
- 改变 physics dt、decimation 或 history length；
- 改变导出模型 forward 签名或 Manifest 字段；
- C++ 基线完成 IsaacLab 迁移；
- 实机 ABI、operation mode 或安全逻辑发生变化。
