# L5A WF-Flat MuJoCo Sim2Sim

该目录是独立的 Python 部署包。运行时只依赖 NumPy、MuJoCo、ONNX Runtime 和
PyYAML，不导入 IsaacLab、Torch 或训练代码。`policy_manifest.json` 是模型、观测、
关节顺序、控制参数和 MJCF 的唯一契约来源。

## 安装

```bash
source ~/.bashrc
lab
cd /mnt/isaacdata/myproject/huilun_isaaclab/sim2sim_mujoco_py
python -m pip install -e ".[test]"
python -m pip check
```

## 运行

固定零速度平衡：

```bash
python -m l5a_sim2sim --vx 0.0 --wz 0.0
```

固定前进命令和无界面验证：

```bash
python -m l5a_sim2sim --headless --duration 20 --vx 0.2 --wz 0.0
```

键盘控制：

```bash
python -m l5a_sim2sim --keyboard
```

`W/S` 增减前进速度，`A/D` 增减偏航速度，`C` 清零，`R` 重置，空格暂停。
横向速度始终为零。`--action-delay-steps` 可设为 0 到 6 个 5 ms 物理步。

默认模型目录和 MJCF 指向本仓库的目标训练日志与 L5A XML，也可以使用
`--model-dir` 和 `--mjcf` 显式覆盖。`--trace output.npz` 会保存策略步级调试轨迹。

