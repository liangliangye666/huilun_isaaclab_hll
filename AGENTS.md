# Repository instructions

## 项目范围

本项目包含基于 IsaacLab 的训练代码框架，以及三类部署路径：

- `sim2sim_python`：Python 版 MuJoCo 仿真部署；
- `sim2sim_cpp`：C++ 版 MuJoCo 仿真部署；
- `sim2real`：C++ 实机部署。

## 工作要求

1. 回答问题前，优先阅读 `docs/README.md`、`docs/ai_project_index.md` 以及 `docs/` 下与任务相关的专项文档。
2. 不要随意修改原始代码。`origin_code/` 默认视为迁移参考或兼容基线；只有用户明确要求时才修改。对 `source/`、部署代码、模型和硬件 ABI 的修改也必须严格限制在请求范围内。
3. 分析训练任务时，第一步先查找 `gym.register`，再沿 `env_cfg_entry_point` 和 `rsl_rl_cfg_entry_point` 追踪环境与算法配置。
4. 默认使用中文解释。涉及代码流时优先给出真实文件、类、函数、tensor shape、关节顺序、线程所有权和物理输出语义。
5. 区分静态源码结论与运行验证；没有实际运行 Isaac Sim、MuJoCo、C++ 程序或实机时，不要声称已经完成对应验证。

## 关键边界

- 当前 IsaacLab 训练主线在 `source/huilun_isaaclab/` 和 `scripts/rsl_rl/`。
- 当前 Python MuJoCo 主线在 `sim2sim_mujoco_py/`。
- `origin_code/sim2sim_cpp_and_sim2real/` 当前仍是旧 Gym 部署契约，不能仅替换模型就视为已适配当前 IsaacLab 导出。
- L5A 的 8 维动作必须保留 6 个腿位置目标与 2 个轮速度目标的区别。
- 当前统一关节顺序为 `[左三腿, 左轮, 右三腿, 右轮]`。
