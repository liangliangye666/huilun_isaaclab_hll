# MuJoCo Sim2Sim

这是独立的 Python MuJoCo 部署目录，适用于以下导出形式：

- `velocity_estimator.onnx`：历史观测进入 Encoder，输出估计基座线速度。
- `policy.onnx`：估计速度、当前 proprioception 和 commands 进入 Actor，输出动作。
- `policy_manifest.json`：记录观测布局、关节顺序、控制模式、PD 参数和 MuJoCo 名称。

运行时只依赖 NumPy、MuJoCo 和 ONNX Runtime，不依赖 IsaacLab、Torch 或训练代码。

## 安装

```bash
source ~/.bashrc
lab
cd /mnt/isaacdata/myproject/huilun_isaaclab/sim2sim_mujoco_py
python -m pip install -e .
python -m pip check
```

## 配置和运行

打开 `sim2sim.py`，在文件顶部修改：

- `MODEL_DIR` 和 `MJCF_PATH`：选择导出结果和机器人 XML。
- `FIXED_COMMAND`：固定速度命令，顺序与 Manifest 的 `command_order` 相同。
- `KEYBOARD_ENABLED`：是否启用 viewer 键盘控制。
- `HEADLESS`、`DURATION_S` 和 `REALTIME_FACTOR`：运行方式和速度。
- `ACTION_DELAY_STEPS`：复现训练动作延迟。
- `TRACE_PATH`：可选调试轨迹输出。

然后直接执行：

```bash
python sim2sim.py
```

键盘模式下，`W/S` 调整前进速度，`A/D` 调整偏航速度，`C` 清零，`R` 重置，
空格暂停或继续。

## 适配边界

关节数量、位置/速度控制模式和张量维度都从 Manifest 读取。当前运行时要求
``policy_action_order`` 与 ``hardware_dof_order`` 逐项相同，两个映射字段也必须是恒等
映射；动作、关节状态和力矩全程直接使用 ``[左三腿, 左轮, 右三腿, 右轮]``，
不提供运行时重排。当前通用观测构造器
支持基座角速度、投影重力、指定关节相对位置、指定关节速度和上一动作。新任务如果加入
其他观测类型，需要在 `src/observation.py` 中增加对应构造逻辑。
