# 项目文档导航

回答本仓库问题前，先阅读本页，再按问题类型进入对应文档。

## 推荐阅读顺序

1. [AI 项目索引](ai_project_index.md)：项目边界、任务入口、核心调用链和按问题查文件。
2. [L5A WF 训练说明](l5a_wf_training.md)：当前 IsaacLab 训练、观测、动作、时序和导出契约。
3. [C++ Sim2Sim/Sim2Real 架构](cpp_sim2sim_sim2real_architecture.md)：搬入的旧 Gym C++ 部署工程、线程、ABI 和迁移差异。
4. [自动生成索引](ai_context/)：目录树、Gym 注册附近源码和原始符号检索结果。

## 代码状态约定

- `source/`、`scripts/`：当前 IsaacLab 训练主线。
- `sim2sim_mujoco_py/`：当前 Manifest 驱动的 Python MuJoCo 部署主线。
- `origin_code/sim2sim_mujoco_py/`：保留旧代码形态的单文件 Python 部署。
- `origin_code/sim2sim_cpp_and_sim2real/`：从旧 Gym 工程搬入的 C++ Sim2Sim/Sim2Real 基线，尚未完成现行 IsaacLab 模型契约迁移。
- `resources/`：当前 IsaacLab/Python 部署使用的 L5A 资产。
- `logs/`、`outputs/`：运行生成物，不是源码真相来源。

文档结论默认来自静态源码检查。涉及 Isaac Sim、MuJoCo 完整运行、C++ 构建、实机时序或安全性的结论，必须单独说明是否做过对应运行验证。
