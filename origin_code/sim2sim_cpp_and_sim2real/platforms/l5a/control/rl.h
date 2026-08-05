#ifndef RL_T2S_H
#define RL_T2S_H

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <torch/script.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "math/gac_math.h"
#include "pd_controller.h"
#include "robot_model.h"

#include <glog/logging.h>
#include <atomic>
#include <condition_variable>
#include <ctime>
#include <fstream>
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
  Eigen::VectorXd tau() { return tau_; };
  Eigen::VectorXd pos_target() { return pos_target_; };
  Eigen::VectorXd vel_target() { return vel_target_; };
  Eigen::VectorXd pos_fb_kp_, pos_fb_kd_;
 private:
  void LoadParameters();
  void WarmUpModels();
  void UpdateHistoryBuffer(Eigen::VectorXd& obs_hist, const Eigen::VectorXd& obs, int num_obs);
  Eigen::VectorXd CreateEstimatorInput(const Eigen::VectorXd& obs_hist_, int num_obs_, int num_actions_, int hist_len_);

  torch::jit::Module controller_, estimator_, scan_encoder_;

  std::string model_est_, model_ctrl_, model_scan_encoder_;

  Eigen::VectorXd pos_target_, vel_target_;

  int num_obs_;
  int num_actions_;
  int hist_len_;
  int num_est_;
  int num_priv_;
  int num_scan_;
  int num_encoder_latent_;
  int num_cmd_;
  int num_ctrl_;
  int iter;

  int decimation_;

  bool isInit_hist{false};
  double time_ms_{0.0};

  // 线程相关成员
  struct SharedData {
    // 输入数据
    Eigen::VectorXd obs;
    Eigen::VectorXd obs_hist;
    Eigen::VectorXd est_latent;
    Eigen::VectorXd est_latent_hist;
    Eigen::VectorXd cmd;
    Eigen::Vector3d base_vel;

    // 输出数据
    Eigen::VectorXd actions;
    Eigen::VectorXd est_lin_vel;
    double inference_time_ms{0.0};  // 推理时间

    // 线程同步
    std::mutex mutex;
    std::condition_variable cv;
    bool inference_ready{false};
    bool has_new_result{false};  // 标识是否有新的推理结果
  };

  SharedData shared_data_;
  std::thread inference_thread_;
  std::atomic<bool> should_stop_{false};
  void InferenceLoop();

  //
  Eigen::VectorXd default_pos_;
  Eigen::VectorXd obs_;
  Eigen::VectorXd obs_hist_;
  Eigen::VectorXd cmd_;
  Eigen::VectorXd est_latent_;
  Eigen::VectorXd est_lin_vel_;
  Eigen::VectorXd est_latent_hist_;
  Eigen::VectorXd actions_;
  Eigen::VectorXd tau_;
  Eigen::VectorXd vel_last_;
  Eigen::VectorXd vel_filted_;
  bool vel_filter_init_;

  // rl obs scale
  double obs_scales_lin_vel_;
  double obs_scales_lin_vel_y_;
  double obs_scales_ang_vel_;
  double obs_scales_dof_pos_;
  double obs_scales_dof_vel_;
  double obs_scales_quat_;
  double obs_scales_height_;
  double lin_vel_x_com_;
  double lin_vel_y_com_;
  double omega_com_;

  // rl actions scale
  double action_scales_pos_, action_scales_vel_;

  // pd cotroller
  PdController<Eigen::VectorXd> pd_controller_joints_;
  Eigen::VectorXd kp_joints_, kd_joints_;

  // EDP
  double edamp_kd_hip_, edamp_kd_knee_, edamp_kd_wheel_;

  //
  double clip_obs_, clip_actions_;
  double forward_back_error_, left_right_error_;
  bool last_forward_pad_, last_back_pad_;
  bool last_left_pad_, last_right_pad_;

  //
  YAML::Node config_;
  // std::shared_ptr<ROS::Publisher> ros_publisher_;
};

}  // namespace l5a

#endif