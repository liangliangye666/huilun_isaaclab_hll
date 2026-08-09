#include "rl.h"

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace l5a {

RL::RL(RobotModel& robot_model) {
  // set_env.sh 会在运行前设置 PROJECT_ROOT_DIR；控制器用它定位 YAML、URDF 和 TorchScript 模型。
  const char* project_root = std::getenv("PROJECT_ROOT_DIR");
  if (project_root == nullptr) {
    throw std::runtime_error("PROJECT_ROOT_DIR is not set");
  }
  const std::string workspace_path(project_root);

#if SIM_ENABLE
  config_ = YAML::LoadFile(workspace_path + "/platforms/l5a/control/rl_parameters_sim.yaml");
#elif PHYSICS_ENABLE
  config_ = YAML::LoadFile(workspace_path + "/platforms/l5a/control/rl_parameters_physics.yaml");
#endif

  // 网络维度、归一化、动作和控制参数统一从当前部署 YAML 读取。
  LoadParameters();

  obs_ = Eigen::VectorXd::Zero(num_obs_);
  obs_hist_ = Eigen::VectorXd::Zero(num_obs_ * hist_len_);
  cmd_ = Eigen::VectorXd::Zero(num_cmd_);
  est_lin_vel_ = Eigen::VectorXd::Zero(num_est_);
  actions_ = Eigen::VectorXd::Zero(num_actions_);
  tau_ = Eigen::VectorXd::Zero(num_actions_);
  pos_target_ = Eigen::VectorXd::Zero(num_actions_);
  vel_target_ = Eigen::VectorXd::Zero(num_actions_);

  shared_data_.obs = Eigen::VectorXd::Zero(num_obs_);
  shared_data_.obs_hist = Eigen::VectorXd::Zero(num_obs_ * hist_len_);
  shared_data_.cmd = Eigen::VectorXd::Zero(num_cmd_);
  shared_data_.actions = Eigen::VectorXd::Zero(num_actions_);
  shared_data_.est_lin_vel = Eigen::VectorXd::Zero(num_est_);

  iter = 1;

  pd_controller_joints_ = PdController<Eigen::VectorXd>(kp_joints_, kd_joints_);

  const auto start = std::chrono::high_resolution_clock::now();
  controller_ = torch::jit::load(workspace_path + "/platforms/l5a/control/module/" + model_ctrl_);
  estimator_ = torch::jit::load(workspace_path + "/platforms/l5a/control/module/" + model_est_);
  WarmUpModels();
  const auto end = std::chrono::high_resolution_clock::now();
  const double initialization_time_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0;
  std::cout << "RL initialization time: " << initialization_time_ms << " ms" << std::endl;

  // 保留旧部署的异步 latest-result-hold 推理结构。
  inference_thread_ = std::thread(&RL::InferenceLoop, this);
}

RL::~RL() {
  should_stop_.store(true, std::memory_order_release);
  shared_data_.cv.notify_all();
  if (inference_thread_.joinable()) {
    inference_thread_.join();
  }
}

void RL::InferenceLoop() {
  while (!should_stop_) {
    Eigen::VectorXd observation;
    Eigen::VectorXd observation_history;
    Eigen::VectorXd commands;

    {
      std::unique_lock<std::mutex> lock(shared_data_.mutex);
      // cv.wait 会在阻塞时释放 mutex，被唤醒后重新加锁；这样控制线程可以继续写入新请求或析构时退出。
      shared_data_.cv.wait(lock, [this]() { return shared_data_.inference_ready || should_stop_; });
      if (should_stop_) {
        break;
      }

      // 只在锁内拷贝输入快照，Torch forward 不占用共享锁，避免阻塞 500 Hz 控制线程。
      observation = shared_data_.obs;
      observation_history = shared_data_.obs_hist;
      commands = shared_data_.cmd;
    }

    const auto start = std::chrono::high_resolution_clock::now();

    try {
      // 推理阶段不需要梯度，禁用 autograd 可以减少内存占用和额外计算。
      torch::NoGradGuard no_grad;
      const auto double_options = torch::TensorOptions().dtype(torch::kDouble).requires_grad(false);

      // Encoder 输入是 10 帧 proprioception，按 oldest -> newest 排列。
      // shape: history [1, 10, 28] -> estimated base linear velocity [1, 3]。
      torch::Tensor history_tensor =
          torch::from_blob(observation_history.data(), {1, hist_len_, num_obs_}, double_options)
              .clone()
              .toType(torch::kFloat);
      std::vector<torch::jit::IValue> estimator_inputs{history_tensor};
      torch::Tensor estimated_velocity = estimator_.forward(estimator_inputs).toTensor().contiguous().cpu();

      // Actor 是 IsaacLab 导出的三输入模型，不再使用旧 Gym 的单个拼接 Tensor。
      // 输入顺序必须和 policy_manifest.json 一致：
      // estimated velocity [1, 3] + current proprioception [1, 28] + commands [1, 3] -> actions [1, 8]。
      torch::Tensor observation_tensor =
          torch::from_blob(observation.data(), {1, num_obs_}, double_options).clone().toType(torch::kFloat);
      torch::Tensor command_tensor =
          torch::from_blob(commands.data(), {1, num_cmd_}, double_options).clone().toType(torch::kFloat);
      std::vector<torch::jit::IValue> controller_inputs{
          estimated_velocity,
          observation_tensor,
          command_tensor,
      };
      torch::Tensor controller_outputs =
          controller_.forward(controller_inputs).toTensor().contiguous().cpu().toType(torch::kDouble);

      const double* action_data = controller_outputs.data_ptr<double>();
      Eigen::Map<const Eigen::VectorXd> action_map(action_data, num_actions_);
      const Eigen::VectorXd actions = action_map;

      torch::Tensor estimated_velocity_double = estimated_velocity.toType(torch::kDouble);
      const double* velocity_data = estimated_velocity_double.data_ptr<double>();
      Eigen::Map<const Eigen::VectorXd> velocity_map(velocity_data, num_est_);
      const Eigen::VectorXd estimated_linear_velocity = velocity_map;

      const auto end = std::chrono::high_resolution_clock::now();
      const double inference_time_ms =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0;

      {
        std::lock_guard<std::mutex> lock(shared_data_.mutex);
        shared_data_.est_lin_vel = estimated_linear_velocity;
        shared_data_.actions = actions;
        shared_data_.inference_time_ms = inference_time_ms;
        shared_data_.inference_ready = false;
        shared_data_.has_new_result = true;
      }
    } catch (const c10::Error& error) {
      std::cerr << "Torch error in inference thread: " << error.what() << std::endl;
      std::lock_guard<std::mutex> lock(shared_data_.mutex);
      // 失败后清掉 ready，避免控制线程永久认为推理线程还在忙。
      shared_data_.inference_ready = false;
    }
  }
  std::cout << "InferenceLoop exited\n";
}

void RL::Run(RobotModel& robot_model) {
  // RobotModel 已经把基座角速度整理到机体系；这里直接按训练观测缩放。
  const Eigen::Vector3d base_omega = robot_model.qdot.segment(3, 3);
  const Eigen::VectorXd pos = robot_model.q_rpy.tail(num_actions_);
  const Eigen::VectorXd vel = robot_model.qdot.tail(num_actions_);

  Eigen::Vector3d gravity;
  gravity << 0.0, 0.0, -1.0;
  const Eigen::Vector3d projected_gravity = robot_model.R_BW * gravity;

  // 当前 IsaacLab WF-Flat 的 proprioception：3 + 3 + 6 + 8 + 8 = 28。
  obs_.segment(0, 3) = base_omega * obs_scales_ang_vel_;
  obs_.segment(3, 3) = projected_gravity * obs_scales_gravity_;

  // 训练只把六个腿关节的位置误差放入 proprioception，两个轮子的位置不进入该段。
  const Eigen::VectorXd relative_pos = pos - default_pos_;
  obs_[6] = relative_pos[static_cast<int>(Joints::left_hip_roll_joint)] * obs_scales_dof_pos_;
  obs_[7] = relative_pos[static_cast<int>(Joints::left_hip_pitch_joint)] * obs_scales_dof_pos_;
  obs_[8] = relative_pos[static_cast<int>(Joints::left_knee_joint)] * obs_scales_dof_pos_;
  obs_[9] = relative_pos[static_cast<int>(Joints::right_hip_roll_joint)] * obs_scales_dof_pos_;
  obs_[10] = relative_pos[static_cast<int>(Joints::right_hip_pitch_joint)] * obs_scales_dof_pos_;
  obs_[11] = relative_pos[static_cast<int>(Joints::right_knee_joint)] * obs_scales_dof_pos_;

  // 8 个关节速度和上一帧 actor 原始动作都按统一关节顺序进入观测。
  obs_.segment(12, num_actions_) = vel * obs_scales_dof_vel_;
  obs_.segment(20, num_actions_) = actions_ * obs_scales_last_action_;
  obs_ = obs_.cwiseMin(clip_obs_).cwiseMax(-clip_obs_);

  // 命令不属于 Encoder 历史，作为 Actor 的独立输入，单位保持 m/s、rad/s。
#if SIM_ENABLE
  cmd_[0] = robot_model.vel_x_des_ + lin_vel_x_com_;
  cmd_[1] = robot_model.vel_y_des_ + lin_vel_y_com_;
  cmd_[2] = robot_model.omega_des_ + omega_com_;
#elif PHYSICS_ENABLE
  cmd_[0] = robot_model.vel_x_des_ + lin_vel_x_com_;
  cmd_[1] = robot_model.vel_y_des_ + lin_vel_y_com_;
  cmd_[2] = robot_model.omega_des_ + omega_com_;
#endif

  // 500 Hz 控制循环每 10 步提交一次推理请求；推理忙时跳过并保持最近动作。
  if (iter >= decimation_) {
    std::lock_guard<std::mutex> lock(shared_data_.mutex);
    if (!is_init_hist_) {
      is_init_hist_ = true;
      // 第一帧没有历史时，用当前观测填满 10 帧，和 Python 部署端的冷启动逻辑一致。
      obs_hist_ = obs_.replicate(hist_len_, 1);
    } else {
      UpdateHistoryBuffer(obs_hist_, obs_, num_obs_);
    }

    if (!shared_data_.inference_ready) {
      shared_data_.obs = obs_;
      shared_data_.obs_hist = obs_hist_;
      shared_data_.cmd = cmd_;
      shared_data_.inference_ready = true;
      shared_data_.cv.notify_one();
    }
    iter = 0;
  }

  // 控制线程不等待本轮推理完成；有新结果就更新，否则继续使用上一帧动作。
  {
    std::lock_guard<std::mutex> lock(shared_data_.mutex);
    if (shared_data_.has_new_result) {
      actions_ = shared_data_.actions;
      est_lin_vel_ = shared_data_.est_lin_vel;
      time_ms_ = shared_data_.inference_time_ms;
      shared_data_.has_new_result = false;
    }
  }

  actions_ = actions_.cwiseMin(clip_actions_).cwiseMax(-clip_actions_);

  // 调试通道：action[8]、command[3]、estimated velocity[3]、推理耗时、obs[28]。
  robot_model.observed_value.segment(1, num_actions_) = actions_;
  robot_model.observed_value.segment(9, num_cmd_) = cmd_;
  robot_model.observed_value.segment(12, num_est_) = est_lin_vel_;
  robot_model.observed_value[15] = time_ms_;
  robot_model.observed_value.segment(16, num_obs_) = obs_;

  Eigen::VectorXd pos_ref = actions_ * action_scales_pos_;
  Eigen::VectorXd vel_ref = actions_ * action_scales_vel_;

  // 统一顺序下，腿输出位置目标，轮输出速度目标，不做动作重排。
  pos_ref[static_cast<int>(Joints::left_wheel_joint)] = 0.0;
  pos_ref[static_cast<int>(Joints::right_wheel_joint)] = 0.0;
  // 腿部速度参考清零；轮子速度参考保留为 action * action_scales_vel_。
  vel_ref.segment(0, 3).setZero();
  vel_ref.segment(4, 3).setZero();

  pos_target_ = default_pos_ + pos_ref;
  vel_target_ = vel_ref;

  pd_controller_joints_.set_x_actual(pos);
  pd_controller_joints_.set_x_desired(pos_target_);
  pd_controller_joints_.set_xdot_actual(vel);
  pd_controller_joints_.set_xdot_desired(vel_target_);
  tau_ = pd_controller_joints_.Update();

  ++iter;
}

void RL::UpdateHistoryBuffer(Eigen::VectorXd& history, const Eigen::VectorXd& observation, int observation_dim) {
  const int total_length = history.size();
  // head 和 tail 引用同一底层缓冲区；eval() 保证滑动时不会被 Eigen 的别名覆盖。
  history.head(total_length - observation_dim) = history.tail(total_length - observation_dim).eval();
  history.tail(observation_dim) = observation;
}

void RL::RunEDamp(RobotModel& robot_model) {
  const Eigen::VectorXd vel = robot_model.qdot.segment(6, num_actions_);
  tau_[static_cast<int>(Joints::left_hip_pitch_joint)] =
      edamp_kd_hip_ * (0.0 - vel[static_cast<int>(Joints::left_hip_pitch_joint)]);
  tau_[static_cast<int>(Joints::left_knee_joint)] =
      edamp_kd_knee_ * (0.0 - vel[static_cast<int>(Joints::left_knee_joint)]);
  tau_[static_cast<int>(Joints::left_wheel_joint)] =
      edamp_kd_wheel_ * (0.0 - vel[static_cast<int>(Joints::left_wheel_joint)]);
  tau_[static_cast<int>(Joints::right_hip_pitch_joint)] =
      edamp_kd_hip_ * (0.0 - vel[static_cast<int>(Joints::right_hip_pitch_joint)]);
  tau_[static_cast<int>(Joints::right_knee_joint)] =
      edamp_kd_knee_ * (0.0 - vel[static_cast<int>(Joints::right_knee_joint)]);
  tau_[static_cast<int>(Joints::right_wheel_joint)] =
      edamp_kd_wheel_ * (0.0 - vel[static_cast<int>(Joints::right_wheel_joint)]);
}

void RL::LoadParameters() {
  num_obs_ = config_["model"]["proprioception_dim"].as<int>();
  hist_len_ = config_["model"]["history_length"].as<int>();
  num_est_ = config_["model"]["estimated_velocity_dim"].as<int>();
  num_cmd_ = config_["model"]["command_dim"].as<int>();
  num_actions_ = config_["model"]["action_dim"].as<int>();
  model_est_ = config_["model"]["estimator"].as<std::string>();
  model_ctrl_ = config_["model"]["controller"].as<std::string>();

  const std::vector<double> kp = config_["control"]["pd_controller"]["kp"].as<std::vector<double>>();
  const std::vector<double> kd = config_["control"]["pd_controller"]["kd"].as<std::vector<double>>();
  const std::vector<double> pos_fb_kp = config_["control"]["pos_fb_controller"]["kp"].as<std::vector<double>>();
  const std::vector<double> pos_fb_kd = config_["control"]["pos_fb_controller"]["kd"].as<std::vector<double>>();
  const std::vector<double> default_pos = config_["default_pos"].as<std::vector<double>>();

  kp_joints_ = Eigen::Map<const Eigen::VectorXd>(kp.data(), kp.size());
  kd_joints_ = Eigen::Map<const Eigen::VectorXd>(kd.data(), kd.size());
  pos_fb_kp_ = Eigen::Map<const Eigen::VectorXd>(pos_fb_kp.data(), pos_fb_kp.size());
  pos_fb_kd_ = Eigen::Map<const Eigen::VectorXd>(pos_fb_kd.data(), pos_fb_kd.size());
  default_pos_ = Eigen::Map<const Eigen::VectorXd>(default_pos.data(), default_pos.size());

  obs_scales_ang_vel_ = config_["obs_scales"]["ang_vel"].as<double>();
  obs_scales_gravity_ = config_["obs_scales"]["gravity"].as<double>();
  obs_scales_dof_pos_ = config_["obs_scales"]["dof_pos"].as<double>();
  obs_scales_dof_vel_ = config_["obs_scales"]["dof_vel"].as<double>();
  obs_scales_last_action_ = config_["obs_scales"]["last_action"].as<double>();

  action_scales_pos_ = config_["control"]["action_scales"]["pos"].as<double>();
  action_scales_vel_ = config_["control"]["action_scales"]["vel"].as<double>();
  decimation_ = config_["control"]["decimation"].as<int>();

  edamp_kd_hip_ = config_["control"]["EDamping"]["edamp_hip"].as<double>();
  edamp_kd_knee_ = config_["control"]["EDamping"]["edamp_knee"].as<double>();
  edamp_kd_wheel_ = config_["control"]["EDamping"]["edamp_wheel"].as<double>();

  clip_obs_ = config_["clip_obs"].as<double>();
  clip_actions_ = config_["clip_actions"].as<double>();
  lin_vel_x_com_ = config_["lin_vel_x_com"].as<double>();
  lin_vel_y_com_ = config_["lin_vel_y_com"].as<double>();
  omega_com_ = config_["omega_com"].as<double>();
}

void RL::WarmUpModels() {
  controller_.eval();
  controller_.to(torch::kCPU);
  estimator_.eval();
  estimator_.to(torch::kCPU);

  torch::NoGradGuard no_grad;
  torch::Tensor dummy_history = torch::rand({1, hist_len_, num_obs_});
  torch::Tensor dummy_observation = torch::rand({1, num_obs_});
  torch::Tensor dummy_commands = torch::rand({1, num_cmd_});

  std::vector<torch::jit::IValue> estimator_inputs{dummy_history};
  torch::Tensor dummy_estimated_velocity = estimator_.forward(estimator_inputs).toTensor();
  std::vector<torch::jit::IValue> controller_inputs{
      dummy_estimated_velocity,
      dummy_observation,
      dummy_commands,
  };
  controller_.forward(controller_inputs);
}

}  // namespace l5a
