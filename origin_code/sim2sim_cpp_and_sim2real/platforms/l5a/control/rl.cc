#include "rl.h"

namespace l5a {
RL::RL(RobotModel& robot_model) {
  // FLAGS_logtostderr = 1;
  // FLAGS_colorlogtostderr = 1;

  num_obs_ = 32;                                   // 观测值数目,输入到actor网络
  num_actions_ = robot_model.pino_model().nv - 6;  // 自由度数目,输出到电机执行
  hist_len_ = 10;                                   // 历史观测长度,输入到critic和actor网络
  num_est_ = 3;                                    // 估计状态数目,输入到critic和actor网络
  num_ctrl_ = num_obs_ + num_est_;                 // 控制输入数目

  obs_ = Eigen::VectorXd::Zero(num_obs_);                   // 观测值
  obs_hist_ = Eigen::VectorXd::Zero(num_obs_ * hist_len_);  // 观测历史
  est_latent_ = Eigen::VectorXd::Zero(num_est_);            // 估计值
  est_lin_vel_ = Eigen::VectorXd::Zero(num_est_);           // 估计线速度
  actions_ = Eigen::VectorXd::Zero(num_actions_);           // 动作
  vel_last_ = Eigen::VectorXd::Zero(num_actions_);          // 上一帧速度
  vel_filted_ = Eigen::VectorXd::Zero(num_actions_);        // 滤波后的速度
  vel_filter_init_ = false;

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

  // 初始化共享数据
  shared_data_.obs = Eigen::VectorXd::Zero(num_obs_);
  shared_data_.est_latent = Eigen::VectorXd::Zero(num_est_);
  shared_data_.est_lin_vel = Eigen::VectorXd::Zero(num_est_);
  shared_data_.actions = Eigen::VectorXd::Zero(num_actions_);
  shared_data_.base_vel = Eigen::Vector3d::Zero();
  shared_data_.inference_ready = false;
  shared_data_.has_new_result = false;
  shared_data_.obs_hist = Eigen::VectorXd::Zero(num_obs_ * hist_len_);

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
  // 启动推理线程
  inference_thread_ = std::thread(&RL::InferenceLoop, this);
}

RL::~RL() {
  // 停止推理线程
  should_stop_.store(true, std::memory_order_release);
  shared_data_.cv.notify_all();
  if (inference_thread_.joinable()) {
    inference_thread_.join();
  }
}

void RL::InferenceLoop() {
  while (!should_stop_) {
    // 等待推理请求
    Eigen::VectorXd obs, obs_hist, est_latent_hist, cmd;
    Eigen::Vector3d base_vel;

    {
      std::unique_lock<std::mutex> lock(shared_data_.mutex);  // 锁定共享数据shared_data_
      shared_data_.cv.wait(
          lock,
          [this]() {  // 调用cv.wait时,会自动解锁shared_data_.mutex,让其他线程有机会修改数据,当其他线程调用cv.notify_one或cv.notify_all,该线程会被唤醒,被唤醒后会重新加锁
            return shared_data_.inference_ready || should_stop_;  // false-阻塞等待,解锁中,其他线程可以修改共享数据;true-阻塞等待结束,加锁,向下执行数据拷贝
          });

      if (should_stop_) {
        break;
      }

      // 获取输入数据的副本
      obs = shared_data_.obs;
      obs_hist = shared_data_.obs_hist;
      base_vel = shared_data_.base_vel;
    }

    // 在进行任何推理相关操作之前开始计时
    auto start = std::chrono::high_resolution_clock::now();

    try {
      torch::NoGradGuard no_grad;  // 创建一个作用域,在该作用域内禁用自动梯度计算,能够显著减少内存占用并提高推理速度,因为在模型推理阶段不需要计算梯度.

      // estimator inference
      Eigen::VectorXd est_input = CreateEstimatorInput(obs_hist, num_obs_, num_actions_, hist_len_);

      torch::Tensor est_input_tensor =
          torch::from_blob(est_input.data(), {1, (num_obs_)*hist_len_}, torch::TensorOptions().dtype(torch::kDouble).requires_grad(false)).clone();
      est_input_tensor = est_input_tensor.toType(torch::kFloat);
      std::vector<torch::jit::IValue> est_inputs;
      est_inputs.push_back(est_input_tensor);
      at::Tensor est_outputs = estimator_.forward(est_inputs).toTensor();
      est_outputs = est_outputs.contiguous().cpu().toType(torch::kDouble);
      const double* est_outputs_ptr = est_outputs.data_ptr<double>();
      Eigen::Map<const Eigen::VectorXd> est_outputs_map(est_outputs_ptr, num_est_);
      Eigen::Vector3d est_lin_vel = est_outputs_map / obs_scales_lin_vel_;
      Eigen::VectorXd est_latent = est_lin_vel * obs_scales_lin_vel_;
      // Eigen::VectorXd est_latent = base_vel * obs_scales_lin_vel_;  // 数据缩放,直接使用仿真器真值
      // std::cout << "est:" << est_lin_vel << ", act:" << base_vel << std::endl;

      // controller inference
      Eigen::VectorXd ctrl_input(num_ctrl_);
      ctrl_input << obs, est_latent;
      torch::Tensor ctrl_inputs_tensor =
          torch::from_blob(ctrl_input.data(), {1, num_ctrl_}, torch::TensorOptions().dtype(torch::kDouble).requires_grad(false)).clone();
      ctrl_inputs_tensor = ctrl_inputs_tensor.toType(torch::kFloat);
      std::vector<torch::jit::IValue> ctrl_inputs;
      ctrl_inputs.push_back(ctrl_inputs_tensor);
      at::Tensor ctrl_outputs = controller_.forward(ctrl_inputs).toTensor();
      ctrl_outputs = ctrl_outputs.contiguous().cpu().toType(torch::kDouble);
      const double* ctrl_outputs_ptr = ctrl_outputs.data_ptr<double>();
      Eigen::Map<const Eigen::VectorXd> actions_map(ctrl_outputs_ptr, num_actions_);
      Eigen::VectorXd actions = actions_map;  // 推理得到动作
      // std::cout << "actions: " << actions.transpose() << std::endl;

      {
        std::lock_guard<std::mutex> lock(shared_data_.mutex);  // 再次加锁
        // 更新结果
        shared_data_.est_latent = est_latent;  // 将推理结果写入共享变量
        shared_data_.actions = actions;
        shared_data_.inference_ready = false;  // 重置变量,继续阻塞线程,等待下一次推理
        shared_data_.has_new_result = true;    // 推理得到新结果

        // 在完成所有操作后、释放锁之前结束计时
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double time_ns = duration.count();
        double time_ms = time_ns / 1000000;
        shared_data_.inference_time_ms = time_ms;
      }
    } catch (const c10::Error& e) {
      std::cerr << "Torch error in inference thread: " << e.what() << std::endl;
    }
  }
  std::cout << "InferenceLoop exited\n";
}

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

  Eigen::Vector3d base_euler = robot_model.ori_base_world_.euler;

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
  obs_[9] = 0.6447 * 5.0;
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
  // obs_[32] = robot_model.gait_enable_;
  // obs_[33] = sin(2 * M_PI * robot_model.phase_) * robot_model.gait_enable_;
  // obs_[34] = cos(2 * M_PI * robot_model.phase_) * robot_model.gait_enable_;
  // obs_[35] = 1;
  obs_ = obs_.cwiseMin(clip_obs_).cwiseMax(-clip_obs_);

  robot_model.observed_value[1] = actions_[0];
  robot_model.observed_value[2] = actions_[1];
  robot_model.observed_value[3] = actions_[2];
  robot_model.observed_value[4] = actions_[3];
  robot_model.observed_value[5] = actions_[4];
  robot_model.observed_value[6] = actions_[5];
  robot_model.observed_value[7] = actions_[6];
  robot_model.observed_value[8] = actions_[7];
  robot_model.observed_value[9] = obs_[8];
  robot_model.observed_value[10] = obs_[9];

  robot_model.observed_value[11] = obs_[10];
  robot_model.observed_value[12] = obs_[11];
  robot_model.observed_value[13] = obs_[12];
  robot_model.observed_value[14] = obs_[13];
  robot_model.observed_value[15] = obs_[14];
  robot_model.observed_value[16] = obs_[15];
  robot_model.observed_value[17] = obs_[16];
  robot_model.observed_value[18] = obs_[17];
  robot_model.observed_value[19] = obs_[18];
  robot_model.observed_value[20] = obs_[19];
  robot_model.observed_value[21] = obs_[20];
  robot_model.observed_value[22] = obs_[21];
  robot_model.observed_value[23] = obs_[22];
  robot_model.observed_value[24] = obs_[23];

  // robot_model.observed_value[22] = shared_data_.inference_time_ms;
  // robot_model.observed_value[23] = robot_model.omega_des_;




  if (iter >= decimation_) {
    // 更新共享数据以触发新的推理
    {
      std::lock_guard<std::mutex> lock(shared_data_.mutex);
      // 更新历史缓冲区
      if (!isInit_hist) {
        isInit_hist = true;
        obs_hist_ = obs_.replicate(hist_len_, 1);
        est_latent_hist_ = est_latent_.replicate(hist_len_, 1);
      } else {
        UpdateHistoryBuffer(obs_hist_, obs_, num_obs_);
        UpdateHistoryBuffer(est_latent_hist_, est_latent_, num_est_);
      }
      // 只有当前一次推理已完成时才触发新的推理
      if (!shared_data_.inference_ready) {
        shared_data_.obs = obs_;
        shared_data_.base_vel = base_vel;
        shared_data_.obs_hist = obs_hist_;
        shared_data_.inference_ready = true;
        shared_data_.cv.notify_one();
      }
    }
    iter = 0;
  }

  // 获取最新的推理结果（如果有）
  {
    std::lock_guard<std::mutex> lock(shared_data_.mutex);
    if (shared_data_.has_new_result) {
      actions_ = shared_data_.actions;
      est_latent_ = shared_data_.est_latent;
      est_lin_vel_ = shared_data_.est_lin_vel;
      time_ms_ = shared_data_.inference_time_ms;
      shared_data_.has_new_result = false;
    }
  }
  actions_ = actions_.cwiseMin(clip_actions_).cwiseMax(-clip_actions_);
  Eigen::VectorXd pos_ref = actions_ * action_scales_pos_;
  Eigen::VectorXd vel_ref = actions_ * action_scales_vel_;

// #if SIM_ENABLE
//   // 在仿真中添加延迟,部署时无须添加
//   // 缓存队列，长度为20，先进先出
//   const int kDelayLen = int(0.020 / robot_model.control_dt);
//   int kNumJoints = robot_model.pino_model().nv - 6;
//   using ActVec = Eigen::VectorXd;
//   static std::deque<ActVec> actions_fifo(kDelayLen, ActVec::Zero(kNumJoints));
//   // push新actions_到队尾
//   actions_fifo.push_back(actions_);
//   // 若队列超长，弹出队首
//   if (actions_fifo.size() > kDelayLen) {
//     actions_fifo.pop_front();
//   }
//   // 使用队首（最旧）actions_
//   const ActVec& delayed_actions = actions_fifo.front();

//   pos_ref = delayed_actions * action_scales_pos_;
//   vel_ref = delayed_actions * action_scales_vel_;
// #endif

  pos_ref.segment(3, 1).setZero();  // wheel pos ref set to zero
  pos_ref.segment(7, 1).setZero();

  vel_ref.segment(0, 3).setZero();  // joint vel ref set to zero
  vel_ref.segment(4, 3).setZero();

  // if(!vel_filter_init_){
  //   vel_filted_ = vel_ref;
  //   vel_filter_init_ = true;
  // }else{
  //   vel_filted_ = 0.95*vel_filted_ + 0.05*vel_ref;
  // }

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
