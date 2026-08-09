#ifndef RL_T2S_H
#define RL_T2S_H

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <torch/script.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "pd_controller.h"
#include "robot_model.h"

#include <glog/logging.h>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace l5a {

class RL {
 public:
  RL(RobotModel& robot_model);
  RL() = default;
  ~RL();

  void Run(RobotModel& robot_model);
  void RunEDamp(RobotModel& robot_model);

  Eigen::VectorXd tau() { return tau_; }
  Eigen::VectorXd pos_target() { return pos_target_; }
  Eigen::VectorXd vel_target() { return vel_target_; }

  Eigen::VectorXd pos_fb_kp_, pos_fb_kd_;

 private:
  void LoadParameters();
  void WarmUpModels();
  void UpdateHistoryBuffer(Eigen::VectorXd& history, const Eigen::VectorXd& observation, int observation_dim);
  void InferenceLoop();

  torch::jit::Module controller_, estimator_;
  std::string model_est_, model_ctrl_;

  int num_obs_;
  int num_actions_;
  int hist_len_;
  int num_est_;
  int num_cmd_;
  int decimation_;
  int iter;

  bool is_init_hist_{false};
  double time_ms_{0.0};

  struct SharedData {
    // 输入快照：控制线程只在推理线程空闲时写入。
    Eigen::VectorXd obs;
    Eigen::VectorXd obs_hist;
    Eigen::VectorXd cmd;

    // 最新推理结果：控制线程非阻塞读取并保持上一动作。
    Eigen::VectorXd actions;
    Eigen::VectorXd est_lin_vel;
    double inference_time_ms{0.0};

    std::mutex mutex;
    std::condition_variable cv;
    bool inference_ready{false};
    bool has_new_result{false};
  };

  SharedData shared_data_;
  std::thread inference_thread_;
  std::atomic<bool> should_stop_{false};

  Eigen::VectorXd default_pos_;
  Eigen::VectorXd obs_;
  Eigen::VectorXd obs_hist_;
  Eigen::VectorXd cmd_;
  Eigen::VectorXd est_lin_vel_;
  Eigen::VectorXd actions_;
  Eigen::VectorXd tau_;
  Eigen::VectorXd pos_target_;
  Eigen::VectorXd vel_target_;

  double obs_scales_ang_vel_;
  double obs_scales_gravity_;
  double obs_scales_dof_pos_;
  double obs_scales_dof_vel_;
  double obs_scales_last_action_;
  double lin_vel_x_com_;
  double lin_vel_y_com_;
  double omega_com_;

  double action_scales_pos_;
  double action_scales_vel_;

  PdController<Eigen::VectorXd> pd_controller_joints_;
  Eigen::VectorXd kp_joints_, kd_joints_;

  double edamp_kd_hip_;
  double edamp_kd_knee_;
  double edamp_kd_wheel_;

  double clip_obs_;
  double clip_actions_;

  YAML::Node config_;
};

}  // namespace l5a

#endif
