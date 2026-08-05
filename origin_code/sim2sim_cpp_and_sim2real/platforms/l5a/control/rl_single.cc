#include "rl.h"

namespace l5a {
RL::RL(RobotModel& robot_model) {
  // FLAGS_logtostderr = 1;
  // FLAGS_colorlogtostderr = 1;

  num_obs_ = 36;                                   // 观测值数目,输入到actor网络
  num_actions_ = robot_model.pino_model().nv - 6;  // 自由度数目,输出到电机执行
  hist_len_ = 10;                                   // 历史观测长度,输入到critic和actor网络
  num_est_ = 3;                                    // 估计状态数目,输入到critic和actor网络
  num_ctrl_ = num_obs_ + num_est_;                 // 控制输入数目

  obs_ = Eigen::VectorXd::Zero(num_obs_);                   // 观测值
  obs_hist_ = Eigen::VectorXd::Zero(num_obs_ * hist_len_);  // 观测历史
  est_latent_ = Eigen::VectorXd::Zero(num_est_);            // 估计值
  est_lin_vel_ = Eigen::VectorXd::Zero(num_est_);           // 估计线速度
  actions_ = Eigen::VectorXd::Zero(num_actions_);           // 动作

  default_pos_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);  // 默认位置
  tau_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);          // 力矩
  pos_target_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);   // 位置期望
  vel_target_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);   // 速度期望

  forward_back_error_ = 0.0;
  left_right_error_ = 0.0;
  lin_vel_x_com_ = 0.0;
  lin_vel_y_com_ = 0.0;
  omega_com_ = 0.0;

  kp_joints_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);
  kd_joints_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);
  pos_fb_kp_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);
  pos_fb_kd_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);

  iter = 1;

  // ros_publisher_ = ROS::Publisher::getInstance();

// Load parameters
#if SIM_ENABLE
  std::string workspacePath = std::getenv("PROJECT_ROOT_DIR");
  config_ = YAML::LoadFile(workspacePath + "/platforms/l5a/control/rl_parameters_sim.yaml");
#elif PHYSICS_ENABLE
  std::string workspacePath = std::getenv("PROJECT_ROOT_DIR");
  config_ = YAML::LoadFile(workspacePath + "/platforms/l5a/control/rl_parameters_physics.yaml");
#endif
  LoadParameters();
  pd_controller_joints_ = PdController<Eigen::VectorXd>(kp_joints_, kd_joints_);
  auto start = std::chrono::high_resolution_clock::now();
  controller_ = torch::jit::load(workspacePath + "/platforms/l5a/control/module/" + model_ctrl_);
  estimator_ = torch::jit::load(workspacePath + "/platforms/l5a/control/module/" + model_est_);
  WarmUpModels();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  double time_ns = duration.count();
  double time_ms = time_ns / 1000000;
  std::cout << "RL initialization time: " << time_ms << " ms" << std::endl;
}

RL::~RL() {}

void RL::InferenceLoop() {}

void RL::Run(RobotModel& robot_model) {
  // 基坐标系的机身线速度和角速度
  Eigen::Vector3d base_vel = robot_model.qdot.segment(0, 3);
  Eigen::Vector3d base_omega = robot_model.qdot.segment(3, 3);

  // 关节位置和速度
  Eigen::VectorXd pos = robot_model.q_rpy.tail(robot_model.pino_model().nv - 6);
  Eigen::VectorXd vel = robot_model.qdot.tail(robot_model.pino_model().nv - 6);
  // Eigen::VectorXd vel = robot_model.joint_vel_;
  // vel[static_cast<int>(Joints::left_wheel_joint)] = robot_model.qdot[static_cast<int>(Joints::left_wheel_joint) + 6];
  // vel[static_cast<int>(Joints::right_wheel_joint)] = robot_model.qdot[static_cast<int>(Joints::right_wheel_joint) + 6];

  Eigen::Vector3d gravity;
  gravity << 0.0, 0.0, -1;
  Eigen::Vector3d projected_gravity = robot_model.R_BW * gravity;

  // base的线速度与角速度
  // obs_.segment(0, 3) = base_vel * obs_scales_lin_vel_;
  obs_.segment(0, 3) = base_omega * obs_scales_ang_vel_;
  obs_.segment(3, 3) = projected_gravity;
  // for (int i = 3; i < 7; i++) {
  //   obs_[i] = robot_model.q_pino[i] * obs_scales_quat_;  // quat
  // }
#if SIM_ENABLE
  obs_[6] = robot_model.vel_x_des_ * obs_scales_lin_vel_;
  obs_[7] = (robot_model.vel_y_des_ - 0.0) * obs_scales_lin_vel_y_;
  obs_[8] = robot_model.omega_des_ * obs_scales_ang_vel_;
#else
  obs_[6] = (robot_model.vel_x_des_ + lin_vel_x_com_) * obs_scales_lin_vel_;
  obs_[7] = (robot_model.vel_y_des_ + lin_vel_y_com_) * obs_scales_lin_vel_y_;
  obs_[8] = (robot_model.omega_des_ + omega_com_) * obs_scales_ang_vel_;
#endif
  obs_[9] = 0.645 * 5.0;
  obs_[10] = (pos - default_pos_)[static_cast<int>(Joints::left_hip_roll_joint)] * obs_scales_dof_pos_;
  obs_[11] = (pos - default_pos_)[static_cast<int>(Joints::left_hip_pitch_joint)] * obs_scales_dof_pos_;
  obs_[12] = (pos - default_pos_)[static_cast<int>(Joints::left_knee_joint)] * obs_scales_dof_pos_;
  obs_[13] = (pos - default_pos_)[static_cast<int>(Joints::right_hip_roll_joint)] * obs_scales_dof_pos_;
  obs_[14] = (pos - default_pos_)[static_cast<int>(Joints::right_hip_pitch_joint)] * obs_scales_dof_pos_;
  obs_[15] = (pos - default_pos_)[static_cast<int>(Joints::right_knee_joint)] * obs_scales_dof_pos_;
  // 关节速度
  obs_[16] = vel[static_cast<int>(Joints::left_hip_roll_joint)] * obs_scales_dof_vel_;
  obs_[17] = vel[static_cast<int>(Joints::left_hip_pitch_joint)] * obs_scales_dof_vel_;
  obs_[18] = vel[static_cast<int>(Joints::left_knee_joint)] * obs_scales_dof_vel_;
  obs_[19] = vel[static_cast<int>(Joints::left_wheel_joint)] * obs_scales_dof_vel_;
  obs_[20] = vel[static_cast<int>(Joints::right_hip_roll_joint)] * obs_scales_dof_vel_;
  obs_[21] = vel[static_cast<int>(Joints::right_hip_pitch_joint)] * obs_scales_dof_vel_;
  obs_[22] = vel[static_cast<int>(Joints::right_knee_joint)] * obs_scales_dof_vel_;
  obs_[23] = vel[static_cast<int>(Joints::right_wheel_joint)] * obs_scales_dof_vel_;
  obs_.segment(24, 8) = actions_;
  obs_[32] = robot_model.gait_enable_;
  obs_[33] = sin(2 * M_PI * robot_model.phase_) * robot_model.gait_enable_;
  obs_[34] = cos(2 * M_PI * robot_model.phase_) * robot_model.gait_enable_;
  obs_[35] = 1;
  obs_ = obs_.cwiseMin(clip_obs_).cwiseMax(-clip_obs_);

  robot_model.observed_value[7] = obs_[0];
  robot_model.observed_value[8] = obs_[1];
  robot_model.observed_value[9] = obs_[2];
  robot_model.observed_value[10] = obs_[3];
  robot_model.observed_value[11] = obs_[4];
  robot_model.observed_value[12] = obs_[5];
  robot_model.observed_value[13] = obs_[6];
  robot_model.observed_value[14] = obs_[7];
  robot_model.observed_value[15] = obs_[8];
  robot_model.observed_value[16] = obs_[9];

  robot_model.observed_value[17] = obs_[10];
  robot_model.observed_value[18] = obs_[11];
  robot_model.observed_value[19] = obs_[12];
  robot_model.observed_value[20] = obs_[13];
  robot_model.observed_value[21] = obs_[14];
  robot_model.observed_value[22] = obs_[15];
  robot_model.observed_value[23] = obs_[16];
  robot_model.observed_value[24] = obs_[17];
  robot_model.observed_value[25] = obs_[18];
  robot_model.observed_value[26] = obs_[19];
  robot_model.observed_value[27] = obs_[20];

  robot_model.observed_value[28] = robot_model.vel_x_des_;
  robot_model.observed_value[29] = robot_model.omega_des_;




  if (iter >= decimation_) {
    if (!isInit_hist) {
      isInit_hist = true;
      obs_hist_ = obs_.replicate(hist_len_, 1);
      est_latent_hist_ = est_latent_.replicate(hist_len_, 1);
    } else {
      UpdateHistoryBuffer(obs_hist_, obs_, num_obs_);
      UpdateHistoryBuffer(est_latent_hist_, est_latent_, num_est_);
    }

    auto start = std::chrono::high_resolution_clock::now();

    try {
      torch::NoGradGuard no_grad;

      Eigen::VectorXd est_input = CreateEstimatorInput(obs_hist_, num_obs_, num_actions_, hist_len_);
      torch::Tensor est_input_tensor =
          torch::from_blob(est_input.data(), {1, num_obs_ * hist_len_}, torch::TensorOptions().dtype(torch::kDouble).requires_grad(false)).clone();
      est_input_tensor = est_input_tensor.toType(torch::kFloat);

      std::vector<torch::jit::IValue> est_inputs;
      est_inputs.push_back(est_input_tensor);
      at::Tensor est_outputs = estimator_.forward(est_inputs).toTensor();
      est_outputs = est_outputs.contiguous().cpu().toType(torch::kDouble);

      const double* est_outputs_ptr = est_outputs.data_ptr<double>();
      Eigen::Map<const Eigen::VectorXd> est_outputs_map(est_outputs_ptr, num_est_);
      est_lin_vel_ = est_outputs_map / obs_scales_lin_vel_;
      est_latent_ = est_lin_vel_ * obs_scales_lin_vel_;
      // est_latent_ = robot_model.qdot.segment(0, 3) * obs_scales_lin_vel_;  // 数据缩放,直接使用仿真器真值

      Eigen::VectorXd ctrl_input(num_ctrl_);
      ctrl_input << obs_, est_latent_;
      torch::Tensor ctrl_inputs_tensor =
          torch::from_blob(ctrl_input.data(), {1, num_ctrl_}, torch::TensorOptions().dtype(torch::kDouble).requires_grad(false)).clone();
      ctrl_inputs_tensor = ctrl_inputs_tensor.toType(torch::kFloat);

      std::vector<torch::jit::IValue> ctrl_inputs;
      ctrl_inputs.push_back(ctrl_inputs_tensor);
      at::Tensor ctrl_outputs = controller_.forward(ctrl_inputs).toTensor();
      ctrl_outputs = ctrl_outputs.contiguous().cpu().toType(torch::kDouble);

      const double* ctrl_outputs_ptr = ctrl_outputs.data_ptr<double>();
      Eigen::Map<const Eigen::VectorXd> actions_map(ctrl_outputs_ptr, num_actions_);
      actions_ = actions_map;

      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      time_ms_ = static_cast<double>(duration.count()) / 1000000.0;
    } catch (const c10::Error& e) {
      std::cerr << "Torch error in single-thread inference: " << e.what() << std::endl;
    }

    iter = 0;
  }
  actions_ = actions_.cwiseMin(clip_actions_).cwiseMax(-clip_actions_);
  Eigen::VectorXd pos_ref = actions_ * action_scales_pos_;
  Eigen::VectorXd vel_ref = actions_ * action_scales_vel_;

#if SIM_ENABLE
  // 在仿真中添加延迟,部署时无须添加
  // 缓存队列，长度为20，先进先出
  const int kDelayLen = int(0.020 / robot_model.control_dt);
  int kNumJoints = robot_model.pino_model().nv - 6;
  using ActVec = Eigen::VectorXd;
  static std::deque<ActVec> actions_fifo(kDelayLen, ActVec::Zero(kNumJoints));
  // push新actions_到队尾
  actions_fifo.push_back(actions_);
  // 若队列超长，弹出队首
  if (actions_fifo.size() > kDelayLen) {
    actions_fifo.pop_front();
  }
  // 使用队首（最旧）actions_
  const ActVec& delayed_actions = actions_fifo.front();

  pos_ref = delayed_actions * action_scales_pos_;
  vel_ref = delayed_actions * action_scales_vel_;
#endif

  pos_ref.segment(3, 1).setZero();  // wheel pos ref set to zero
  pos_ref.segment(7, 1).setZero();

  vel_ref.segment(0, 3).setZero();  // joint vel ref set to zero
  vel_ref.segment(4, 3).setZero();

  pos_target_ = pos_ref + default_pos_;
  vel_target_ = vel_ref;
  // std::cout << "pos_tar: " << pos_target_ << std::endl;
  // std::cout << "vel_tar: " << vel_target_ << std::endl;

  pd_controller_joints_.set_x_actual(pos);
  pd_controller_joints_.set_x_desired(pos_target_);
  pd_controller_joints_.set_xdot_actual(vel);
  pd_controller_joints_.set_xdot_desired(vel_target_);
  tau_ = pd_controller_joints_.Update();
  // std::cout << "tau_: " << tau_ << std::endl;
  ++iter;
}

void RL::UpdateHistoryBuffer(Eigen::VectorXd& hist_buf, const Eigen::VectorXd& buf, int num_buf) {
  int total_len = hist_buf.size();
  // 滑动窗口：向上移动，丢弃最老的一帧（前 num_buf 个元素）
  hist_buf.head(total_len - num_buf) = hist_buf.tail(total_len - num_buf);
  // 添加新的观测帧到最后
  hist_buf.tail(num_buf) = buf;
}

Eigen::VectorXd RL::CreateEstimatorInput(const Eigen::VectorXd& obs_hist, int num_obs, int num_actions, int hist_len) {
  // 计算每个观测中非动作部分的大小
  // int obs_wo_actions_size = num_obs - num_actions;  // 22-6=16
  int obs_wo_actions_size = num_obs;  // 22-6=16
  // 创建结果向量
  Eigen::VectorXd est_input(obs_wo_actions_size * hist_len);  // 16*10=160
  for (int t = 0; t < hist_len; ++t) {
    // 使用Eigen的分块操作提取每个时间步中的非动作部分
    est_input.segment(t * obs_wo_actions_size,
                      obs_wo_actions_size) =  // 挑选出除去actions后的160个信息作为状态估计的输入
        obs_hist.segment(t * num_obs, obs_wo_actions_size);
  }
  return est_input;
}

void RL::RunEDamp(RobotModel& robot_model) {
  Eigen::VectorXd vel = robot_model.qdot.segment(6, robot_model.pino_model().nv - 6);
  tau_[static_cast<int>(Joints::left_hip_pitch_joint)] = edamp_kd_hip_ * (0 - vel[static_cast<int>(Joints::left_hip_pitch_joint)]);

  tau_[static_cast<int>(Joints::left_knee_joint)] = edamp_kd_knee_ * (0 - vel[static_cast<int>(Joints::left_knee_joint)]);

  tau_[static_cast<int>(Joints::left_wheel_joint)] = edamp_kd_wheel_ * (0 - vel[static_cast<int>(Joints::left_wheel_joint)]);

  tau_[static_cast<int>(Joints::right_hip_pitch_joint)] = edamp_kd_hip_ * (0 - vel[static_cast<int>(Joints::right_hip_pitch_joint)]);

  tau_[static_cast<int>(Joints::right_knee_joint)] = edamp_kd_knee_ * (0 - vel[static_cast<int>(Joints::right_knee_joint)]);

  tau_[static_cast<int>(Joints::right_wheel_joint)] = edamp_kd_wheel_ * (0 - vel[static_cast<int>(Joints::right_wheel_joint)]);
}

void RL::LoadParameters() {
  kp_joints_ = Eigen::VectorXd::Map(config_["control"]["pd_controller"]["kp"].as<std::vector<double>>().data(),
                                    config_["control"]["pd_controller"]["kp"].as<std::vector<double>>().size());
  kd_joints_ = Eigen::VectorXd::Map(config_["control"]["pd_controller"]["kd"].as<std::vector<double>>().data(),
                                    config_["control"]["pd_controller"]["kd"].as<std::vector<double>>().size());
  pos_fb_kp_ = Eigen::VectorXd::Map(config_["control"]["pos_fb_controller"]["kp"].as<std::vector<double>>().data(),
                                    config_["control"]["pos_fb_controller"]["kp"].as<std::vector<double>>().size());
  pos_fb_kd_ = Eigen::VectorXd::Map(config_["control"]["pos_fb_controller"]["kd"].as<std::vector<double>>().data(),
                                    config_["control"]["pos_fb_controller"]["kd"].as<std::vector<double>>().size());
  // 访问数据
  obs_scales_lin_vel_ = config_["obs_scales"]["lin_vel"].as<double>();
  obs_scales_lin_vel_y_ = config_["obs_scales"]["lin_vel_y"].as<double>();
  obs_scales_ang_vel_ = config_["obs_scales"]["ang_vel"].as<double>();
  obs_scales_dof_pos_ = config_["obs_scales"]["dof_pos"].as<double>();
  obs_scales_dof_vel_ = config_["obs_scales"]["dof_vel"].as<double>();
  obs_scales_quat_ = config_["obs_scales"]["quat"].as<double>();
  obs_scales_height_ = config_["obs_scales"]["height_measurements"].as<double>();
  action_scales_pos_ = config_["control"]["action_scales"]["pos"].as<double>();
  action_scales_vel_ = config_["control"]["action_scales"]["vel"].as<double>();
  edamp_kd_hip_ = config_["control"]["EDamping"]["edamp_hip"].as<double>();
  edamp_kd_knee_ = config_["control"]["EDamping"]["edamp_knee"].as<double>();
  edamp_kd_wheel_ = config_["control"]["EDamping"]["edamp_wheel"].as<double>();
  clip_obs_ = config_["clip_obs"].as<double>();
  clip_actions_ = config_["clip_actions"].as<double>();
  default_pos_ = Eigen::VectorXd::Map(config_["default_pos"].as<std::vector<double>>().data(), config_["default_pos"].as<std::vector<double>>().size());
  decimation_ = config_["control"]["decimation"].as<int>();
  model_est_ = config_["model"]["estimator"].as<std::string>();
  model_ctrl_ = config_["model"]["controller"].as<std::string>();
  model_scan_encoder_ = config_["model"]["scan_encoder"].as<std::string>();
  lin_vel_x_com_ = config_["lin_vel_x_com"].as<double>();
  lin_vel_y_com_ = config_["lin_vel_y_com"].as<double>();
  omega_com_ = config_["omega_com"].as<double>();
}

void RL::WarmUpModels() {
  controller_.eval();
  controller_.to(torch::kCPU);
  estimator_.eval();
  estimator_.to(torch::kCPU);
  // 预热模型
  {
    torch::NoGradGuard no_grad;
    torch::Tensor dummy_est_input = torch::rand({1, (num_obs_)*hist_len_});
    torch::Tensor dummy_ctrl_input = torch::rand({1, num_ctrl_});
    // 封装为IValue向量
    std::vector<torch::jit::IValue> est_inputs{dummy_est_input};
    std::vector<torch::jit::IValue> ctrl_inputs{dummy_ctrl_input};
    estimator_.forward(est_inputs);
    controller_.forward(ctrl_inputs);
  }
}

}  // namespace l5a
