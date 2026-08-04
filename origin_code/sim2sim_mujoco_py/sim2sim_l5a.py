# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import json
import os

import mujoco
import mujoco_viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# 当前文件位于 <project_root>/origin_code/sim2sim_mujoco_py/，向上两级即项目根目录。
HUILUN_ISAACLAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class cmd:
    """固定的平衡指令：[前进速度, 横向速度, 偏航角速度]。"""

    vel_x = 0.0
    vel_y = 0.0
    vel_yaw = 0.0


def get_obs(data):
    """Extracts an observation from the mujoco data structure."""
    # [1] 读取关节状态
    q = data.qpos.astype(np.double)     # ← 全部关节位置（包含基座 free joint 的 7 个分量（3 位置 + 4 四元数）+ 8 个 hinge 关节各 1 个角度 = 共 15 个元素）
    dq = data.qvel.astype(np.double)    # ← 全部关节速度（包含基座 free joint 的 6 个速度分量 + 8 个 hinge 关节各 1 个角速度 = 共 14 个元素）
    # [2] 读取姿态四元数（传感器）
    # MuJoCo framequat 输出 [w, x, y, z]，SciPy 的 Rotation.from_quat()，它要求输入顺序是 [x, y, z, w]
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    # [3] 创建旋转对象，表示基座→世界的旋转
    # 给定基座坐标系下的一个向量 v，r.apply(v) 能算出它在世界坐标系下的方向；
    # 反过来 r.apply(v, inverse=True) 能把世界坐标系下的向量转到基座坐标系下。
    r = R.from_quat(quat)
    # [4] 计算基座坐标系下的线速度
    # data.qvel[:3]：MuJoCo 中 qvel 的前 3 个分量是基座 free joint 的线速度 [vx, vy, vz]，在世界坐标系下表示。
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # 世界坐标系下的基座线速度 [vx, vy, vz] 旋转到基座坐标系
    # [5] 读取角速度
    # 直接读取陀螺仪传感器数据，得到基座三轴角速度 [ωx, ωy, ωz]。注意陀螺仪输出已经是基座坐标系下的值，不需要做坐标变换
    omega = data.sensor("angular-velocity").data.astype(np.double)  # ← 陀螺仪读数 [ωx, ωy, ωz]
    # [6] 计算投影重力
    # 世界坐标系下的重力：[0.0, 0.0, -1.0]，指向正下方，转换后表示"重力在基座坐标系中指向哪个方向"
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return q, dq, quat, v, omega, gvec


def pd_control(target_q, default_q, q, kp, target_dq, dq, kd):  # target_q:目标位置增量,default_q:关节默认位置/中立位置
    """Calculates torques from position and velocity commands."""
    return (target_q + default_q - q) * kp + (target_dq - dq) * kd


def initialize_qpos(model, data):
    """Initialize the robot with the first keyframe in the MuJoCo model."""
    data.qpos[:] = model.key_qpos[0]


def run_mujoco(policy, estimator, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The TorchScript actor policy.
        estimator: The TorchScript base linear velocity estimator.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    # [1] 加载 MuJoCo 物理引擎
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)

    # # 获取关节名称和对应的索引
    # for joint_id in range(model.njnt):
    #     joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    #     qpos_index = model.jnt_qposadr[joint_id]
    #     print(f"Joint {joint_id}: {joint_name}, qpos index: {qpos_index}")

    # # 遍历所有 actuators
    # motor_names = []
    # for actuator_id in range(model.nu):
    #     actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    #     motor_names.append(actuator_name)
    # print("Motors in order:", motor_names)

    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    # [2] 初始化机器人状态
    # Set the initial state
    initialize_qpos(model, data)    # ← 设置初始关节角度

    # default_q、q、dq、target_q、target_dq 和 tau 均使用 MuJoCo/硬件顺序：
    # [左 roll, 左 pitch, 左 knee, 左轮, 右 roll, 右 pitch, 右 knee, 右轮]。
    default_q = cfg.asset.default_q_policy[cfg.asset.policy_to_mujoco_indices]

    # [3] 首次物理步 + 启动 viewer
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # [4] 初始化控制变量
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)   # ← 位置目标
    target_dq = np.zeros(cfg.env.num_actions, dtype=np.double)  # ← 速度目标
    action = np.zeros(cfg.env.num_actions, dtype=np.double)     # ← 上一动作

    count_lowlevel = 0      # ← 物理步计数器
    obs_history = None      # ← 历史窗口（首次填满）

    # [5] 主循环
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        # [5a] 读取 MuJoCo 观测
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions :]       # ← 只取最后 8 个关节
        dq = dq[-cfg.env.num_actions :]     # ← 只取最后 8 个关节

        # MuJoCo 顺序转换为训练策略顺序：六个腿关节在前，两个轮关节在后。
        q_policy = q[cfg.asset.mujoco_to_policy_indices]
        dq_policy = dq[cfg.asset.mujoco_to_policy_indices]

        # [5b] 策略推理（每 decimation 步一次）
        # 200 Hz physics -> 100 Hz policy
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # 当前 WF policy 的单帧 proprioception 共 28 维：
            # 拼接 proprio_obs [1, 28]：3 维角速度 + 3 维投影重力 + 6 维腿位置 + 8 维关节速度 + 8 维上一动作。
            proprio_obs = np.zeros((1, cfg.env.num_observations), dtype=np.float32)
            proprio_obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
            proprio_obs[0, 3:6] = gvec * cfg.normalization.obs_scales.gravity
            proprio_obs[0, 6:12] = (
                q_policy[cfg.asset.policy_leg_indices]
                - cfg.asset.default_q_policy[cfg.asset.policy_leg_indices]
            ) * cfg.normalization.obs_scales.dof_pos
            proprio_obs[0, 12:20] = dq_policy * cfg.normalization.obs_scales.dof_vel
            proprio_obs[0, 20:28] = action * cfg.normalization.obs_scales.last_action
            proprio_obs = np.clip(
                proprio_obs,
                -cfg.normalization.clip_observations,
                cfg.normalization.clip_observations,
            )

            # 更新历史窗口：第一次推理用当前帧填满 10 帧历史；之后按 oldest -> newest 滑动。
            if obs_history is None:
                obs_history = np.tile(
                    proprio_obs[:, np.newaxis, :],
                    (1, cfg.env.obs_history_length, 1),
                )
            else:
                obs_history = np.concatenate(
                    (obs_history[:, 1:, :], proprio_obs[:, np.newaxis, :]),
                    axis=1,
                )

            commands = np.array([[cmd.vel_x, cmd.vel_y, cmd.vel_yaw]], dtype=np.float32)

            # ONNX 推理
            with torch.no_grad():
                latent = estimator(torch.tensor(obs_history, dtype=torch.float32))
                action[:] = policy(
                    latent,
                    torch.tensor(proprio_obs, dtype=torch.float32),
                    torch.tensor(commands, dtype=torch.float32),
                )[0].detach().numpy()
            # 裁剪 action
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

        # [5c] 计算控制目标（每个物理步都执行）：策略动作顺序转换为 MuJoCo/硬件顺序。
        action_mujoco = action[cfg.asset.policy_to_mujoco_indices]
        target_q[cfg.asset.mujoco_leg_indices] = (      # target_q[腿] = action * action_scale_pos
            action_mujoco[cfg.asset.mujoco_leg_indices] * cfg.control.action_scale_pos
        )
        target_dq[cfg.asset.mujoco_wheel_indices] = (   # target_dq[轮] = action * action_scale_vel
            action_mujoco[cfg.asset.mujoco_wheel_indices] * cfg.control.action_scale_vel
        )
        print("action:", action[cfg.asset.policy_wheel_indices])

        # [5d] PD 控制 + 写入力矩
        # Generate PD control
        tau = pd_control(
            target_q,
            default_q,
            q,
            cfg.robot_config.kps,
            target_dq,
            dq,
            cfg.robot_config.kds,
        )
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)

        data.ctrl = tau
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == "__main__":
    import argparse

    '''
    [1] 命令行参数解析
     ├─ --load_model: policy.pt 路径（必填）
     └─ --load_estimator: velocity_estimator.pt 路径（可选，默认同目录）
    '''
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--load_model", type=str, required=True, help="Path to policy.pt.")
    parser.add_argument(
        "--load_estimator",
        type=str,
        default=None,
        help="Path to velocity_estimator.pt. Defaults to the file next to --load_model.",
    )
    args = parser.parse_args()

    '''
    [2] 定位文件
     ├─ model_dir = policy.pt 所在目录
     ├─ estimator_path = 用户指定的或 model_dir/velocity_estimator.pt
     └─ manifest_path = model_dir/policy_manifest.json
    '''
    model_dir = os.path.dirname(os.path.abspath(args.load_model))
    estimator_path = args.load_estimator
    if estimator_path is None:
        estimator_path = os.path.join(model_dir, "velocity_estimator.pt")

    manifest_path = os.path.join(model_dir, "policy_manifest.json")

    '''
    [3] 读取 Manifest JSON
    '''
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    deployment = manifest["deployment"]
    proprioception_layout = deployment["proprioception_layout"]

    '''
    [4] 组装 Sim2simCfg 配置对象
    '''
    class Sim2simCfg:
        """Configuration matching the exported Huilun L5A WF-Flat policy."""

        class env:
            num_actions = deployment["action_dim"]
            num_observations = deployment["proprioception_dim"]
            obs_history_length = deployment["history_samples"]
            '''
            action_dim = 8：8 个受控关节
            proprioception_dim = 28：5 项观测拼接
            history_samples = 10：Encoder 需要 10 帧历史
            '''

        class sim_config:
            mujoco_model_path = os.path.join(
                HUILUN_ISAACLAB_ROOT_DIR,
                deployment["robot_model"]["mjcf_path"],
            )
            sim_duration = 500.0    # 仿真时长
            dt = deployment["physics_period_s"]
            decimation = deployment["decimation"]

        # 关节顺序映射 + 默认关节角
        class asset:
            # 策略顺序：[左三腿, 右三腿, 左轮, 右轮]
            # MuJoCo 顺序：[左三腿, 左轮, 右三腿, 右轮]
            mujoco_to_policy_indices = np.array([0, 1, 2, 4, 5, 6, 3, 7], dtype=np.int64)
            policy_to_mujoco_indices = np.array([0, 1, 2, 6, 3, 4, 5, 7], dtype=np.int64)
            policy_leg_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
            policy_wheel_indices = np.array([6, 7], dtype=np.int64)
            mujoco_leg_indices = np.array([0, 1, 2, 4, 5, 6], dtype=np.int64)
            mujoco_wheel_indices = np.array([3, 7], dtype=np.int64)
            default_q_policy = np.array(
                deployment["default_joint_positions"]["values"],
                dtype=np.double,
            )

        # 观测缩放系数 + 裁剪范围
        class normalization:
            class obs_scales:
                ang_vel = proprioception_layout[0]["scale"]
                gravity = proprioception_layout[1]["scale"]
                dof_pos = proprioception_layout[2]["scale"]
                dof_vel = proprioception_layout[3]["scale"]
                last_action = proprioception_layout[4]["scale"]

            # 当前训练观测项未设置额外 clip；保留旧 sim2sim 的宽松观测保护边界。
            clip_observations = 100.0
            clip_actions = deployment["policy_output_clip"]

        # 动作缩放系数
        class control:
            action_scale_pos = deployment["policy_action_semantics"]["leg_position"]["scale"]
            action_scale_vel = deployment["policy_action_semantics"]["wheel_velocity"]["scale"]

        # PD 参数（kps, kds, tau_limit）
        class robot_config:
            # Manifest 中的控制参数是策略顺序，PD 计算前转换为 MuJoCo 顺序。
            kps = np.array(deployment["joint_control"]["stiffness"], dtype=np.double)[
                [0, 1, 2, 6, 3, 4, 5, 7]
            ]
            kds = np.array(deployment["joint_control"]["damping"], dtype=np.double)[
                [0, 1, 2, 6, 3, 4, 5, 7]
            ]
            tau_limit = np.array(deployment["joint_control"]["effort_limits"], dtype=np.double)[
                [0, 1, 2, 6, 3, 4, 5, 7]
            ]

    '''
    [5] 加载 TorchScript 模型
    '''
    policy = torch.jit.load(args.load_model)
    estimator = torch.jit.load(estimator_path)

    '''
    [6] 启动仿真
    '''
    run_mujoco(policy, estimator, Sim2simCfg())
