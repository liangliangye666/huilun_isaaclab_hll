# L5A WF Blind-Flat 训练说明

本文档描述 `Huilun-L5A-WF-Flat-v0` 的训练数据流、TRON2 迁移边界和
L5A 部署契约。代码结构以本项目的 Manager-Based IsaacLab 框架为准，
没有引入 TRON2 仓库中自带的旧版 RSL-RL 分叉。

## 1. 任务入口

- 训练：`Huilun-L5A-WF-Flat-v0`
- 播放/导出：`Huilun-L5A-WF-Flat-Play-v0`
- 原有平衡任务仍保留：
  - `Huilun-L5A-Balance-v0`
  - `Huilun-L5A-Balance-Play-v0`

训练命令：

```bash
/mnt/isaacdata/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Huilun-L5A-WF-Flat-v0 \
  --headless
```

指定环境数或训练迭代数：

```bash
/mnt/isaacdata/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Huilun-L5A-WF-Flat-v0 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --headless
```

播放并导出：

```bash
/mnt/isaacdata/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Huilun-L5A-WF-Flat-Play-v0 \
  --checkpoint /absolute/path/to/model_XXXX.pt
```

## 2. Encoder、Actor 和 Critic 数据流

单帧本体观测为 28 维：

```text
base_ang_vel * 0.25          3
projected_gravity            3
leg_joint_pos_rel            6
all_joint_vel_rel * 0.05     8
last_action                  8
                              --
                              28
```

Actor 数据流：

```text
obs_history [N, 10, 28]
        |
        v
Encoder: 280 -> 256 -> 128 -> 3
        |
        +---- estimated base_lin_vel [N, 3]
                         |
                         v
[estimated velocity 3, proprioception 28, commands 3]
                         |
                         v
                 Actor input [N, 34]
                         |
                         v
                    actions [N, 8]
```

Encoder 的监督目标是独立的无噪声 observation group：

```text
base_lin_vel_target [N, 3]
```

它不再依赖“critic 前三维必须恰好是 base velocity”这样的隐式位置约定。
Encoder 使用独立 Adam（默认 `1e-3`）和 MSE loss；Actor 使用
`estimate.detach()`，因此 PPO 与 Encoder 的参数更新互不混合。

Critic 使用 68 维特权观测加 3 维 command，共 71 维。特权内容包括真实
base velocity、关节 torque/acceleration、轮体速度、随机化后的当前质量和
轮地接触力。

L5A 保持 `dt=0.005 s`、`decimation=2`，所以策略频率为 100 Hz。10 帧历史
是一个 0.10 s 的采样窗口（最旧与最新样本时间戳相差 0.09 s）；TRON2 的
`decimation=4` 对应 0.20 s 采样窗口。这是为了同时满足
“不修改 L5A decimation”和“使用 10 帧 Encoder”的明确选择。若后续实测
发现速度估计需要更长时间上下文，应单独比较 10/15/20 帧，而不是修改
L5A 控制周期。

## 3. 动作与 L5A 硬约束

策略动作顺序以当前 IsaacLab 框架为准：

```text
0 left_hip_roll_joint       position, scale 0.25 + randomized default
1 left_hip_pitch_joint      position, scale 0.25 + randomized default
2 left_knee_joint           position, scale 0.25 + randomized default
3 right_hip_roll_joint      position, scale 0.25 + randomized default
4 right_hip_pitch_joint     position, scale 0.25 + randomized default
5 right_knee_joint          position, scale 0.25 + randomized default
6 left_wheel_joint          velocity, scale 0.5
7 right_wheel_joint         velocity, scale 0.5
```

以下 L5A 参数保持不变：

- physics `dt=0.005 s`
- policy `decimation=2`
- wheel radius `0.127 m`
- nominal track width `0.28 m`，允许范围 `0.27–0.30 m`
- target base height `0.645 m`
- 原有关节默认角、effort/velocity limits 和 Kp/Kd
- 关闭 self-collision，soft position limit factor `0.95`

训练使用一个统一的 delayed implicit actuator。每个环境为完整 8 维命令
采样同一个 `0–6` physics-step lag，即 `0–30 ms`。这样既保留 TRON2 的
延迟执行器思想，也保留旧 L5A 的腿轮共享延迟语义。

## 4. WF 训练内容

### Commands

- `vx`: `[-1, 1] m/s`
- `vy`: 固定 `0`
- `yaw rate`: `[-1, 1] rad/s`
- heading: `[-pi, pi]`
- 10% standing environments
- 10 s 重采样周期

L5A 是两轮非完整约束系统，不能在不发生轮胎侧滑的情况下直接跟踪横向
速度；TRON2 注册的 WF Blind-Flat 任务同样将 `vy` 设为 0。因此这里的
“全向 WF”表示前进/后退、原地及行进转向、任意 heading，而不是平移侧行。

### Rewards

迁移了 TRON2 WF Blind-Flat 中适用于 L5A 的全部奖励类别：

- xy/yaw 速度跟踪、存活、静止
- 左右轮对称、前后对齐、轮距、base 位于支撑中点
- base 高度、垂向速度、roll/pitch 角速度、姿态
- action rate、二阶 action smoothness
- 非期望接触
- torque、acceleration、joint limit、power、轮/腿速度

TRON2 的 `proximal_yaw_init_offset` 不适用，因为 L5A 没有对应 yaw 关节。
轮距指数核修正了 TRON2 源码中 nominal 项等效使用 `std^4` 的问题。

### Domain randomization

- base mass、link mass、全身 mass/inertia
- friction 和有效 restitution
- Kp/Kd
- per-environment motor effort capability
- base/link COM
- 每环境固定的腿关节零位误差 `±0.05 rad`
- 每环境固定 IMU roll/pitch mounting bias `±1.2 deg`
- 完整动作共享 `0–30 ms` delay
- reset pose/velocity/joints
- 周期性 base velocity push

Play 配置关闭上述训练随机化并换回无延迟的标称 `L5A_CFG`。

注册的 TRON2 WF 任务本身是 blind flat：height scanner、rough terrain、
terrain curriculum、gait command、air-time/slide reward 均没有实际启用，
因此本任务也没有把这些未启用代码机械搬进来。

## 5. 部署输入输出

播放脚本会在 checkpoint 同目录的 `exported/` 中生成：

```text
policy.pt
policy.onnx
velocity_estimator.pt
velocity_estimator.onnx
policy_manifest.json
```

组合策略的三个输入为：

```text
proprioception       [N, 28]
observation_history  [N, 10, 28]  # oldest -> newest
commands             [N, 3]
```

旧 Gym/硬件 DOF 顺序为：

```text
[L_roll, L_pitch, L_knee, L_wheel, R_roll, R_pitch, R_knee, R_wheel]
```

策略输出转硬件顺序：

```text
[a0, a1, a2, a6, a3, a4, a5, a7]
```

硬件 joint state 转策略顺序：

```text
[q0, q1, q2, q4, q5, q6, q3, q7]
```

这些顺序、观测布局、缩放和控制周期会同时写入
`policy_manifest.json`，部署端不应再依赖口头约定。manifest 还记录每个
模型文件的 SHA-256；所有模型先在临时目录生成，manifest 最后发布，避免
导出中断后把半套新模型误认为完整结果。

## 6. Checkpoint 与兼容性

- 使用系统安装的 `rsl-rl-lib >= 3.1.2`，不使用 TRON2 vendored RSL-RL。
- checkpoint 同时保存：
  - policy/critic/Encoder parameters
  - PPO optimizer state
  - Encoder optimizer state
  - 训练时的部署 metadata
- 恢复训练时按目标 device 加载。
- 若旧 checkpoint 的部署 metadata 与当前配置不同，加载时会警告，并以
  checkpoint 中的契约生成 manifest，避免给旧权重贴上新观测布局。
- PPO rollout 通过 TensorDict 原样保存 history 和显式 estimator target。
- 导出模型已将 Encoder 和 Actor 串成一个可直接推理的组合模型，同时保留
  单独 Encoder 文件用于速度估计诊断。

端到端训练仍必须在能正常启动 Isaac Sim/PhysX 的 IsaacLab Python 环境中
进行；普通 Python 解释器只能完成静态和纯网络验证。
