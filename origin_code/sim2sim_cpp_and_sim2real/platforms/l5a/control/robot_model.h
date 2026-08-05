#ifndef ROBOT_MODEL_H
#define ROBOT_MODEL_H

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#if SIM_ENABLE
#  include <mujoco/mujoco.h>
#elif PHYSICS_ENABLE
#  include "standMode_types.h"
#endif

#include <cmath>
#include <iostream>
#include <unordered_map>

#include "fsm_id.h"
#include "imu.h"
#include "joint_vel_est.h"
#include "joystick.h"
#include "math/gac_math.h"
#include "ros_publisher.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"

namespace l5a {

enum class PinoJoints {
  universe = 0,
  root_joint,
  left_hip_roll_joint,
  left_hip_pitch_joint,
  left_knee_joint,
  left_wheel_joint,
  right_hip_roll_joint,
  right_hip_pitch_joint,
  right_knee_joint,
  right_wheel_joint,
};

enum class Joints {
  left_hip_roll_joint = 0,
  left_hip_pitch_joint,
  left_knee_joint,
  left_wheel_joint,
  right_hip_roll_joint,
  right_hip_pitch_joint,
  right_knee_joint,
  right_wheel_joint,
};

enum class AllJoints {
  left_hip_roll_joint = 6,
  left_hip_pitch_joint,
  left_knee_joint,
  left_wheel_joint,
  right_hip_roll_joint,
  right_hip_pitch_joint,
  right_knee_joint,
  right_wheel_joint,
};

const std::unordered_map<AllJoints, std::string> jointToString = {
    {AllJoints::left_hip_roll_joint, "left_hip_roll_joint"},   {AllJoints::left_hip_pitch_joint, "left_hip_pitch_joint"},
    {AllJoints::left_knee_joint, "left_knee_joint"},           {AllJoints::left_wheel_joint, "left_wheel_joint"},
    {AllJoints::right_hip_roll_joint, "right_hip_roll_joint"}, {AllJoints::right_hip_pitch_joint, "right_hip_pitch_joint"},
    {AllJoints::right_knee_joint, "right_knee_joint"},         {AllJoints::right_wheel_joint, "right_wheel_joint"},
};

struct Limits {
  AllJoints joint;
  double q_min;
  double q_max;
  double qdot_min;
  double qdot_max;
  double tau_min;
  double tau_max;
};

extern const std::vector<PinoJoints> wheel_joints;
extern const std::vector<PinoJoints> front_wheel_joints;
extern const std::vector<AllJoints> all_joints;
extern const std::vector<AllJoints> redundant_joints;
class RobotModel {
 public:
  RobotModel(std::string urdf_path);
  ~RobotModel();
  void Initialize();
  void UpdateModel();
#if SIM_ENABLE
  void UpdateMujocoJointStates(const mjModel* m, mjData* d);
  const mjModel* mj_model() { return mj_model_; }
  const mjData* mj_data() { return mj_data_; }
  std::shared_ptr<JoyStick> joystick() { return joystick_; }
#endif
#if PHYSICS_ENABLE
  void UpdateRealJointStates(standmode_output_t* standmode_output, standmode_input_t* standmode_input);
  standmode_output_t* standmode_output() { return standmode_output_; }
  standmode_input_t* standmode_input() { return standmode_input_; }
#endif
  pinocchio::Model& pino_model() { return pino_model_; }
  pinocchio::Data& pino_data() { return pino_data_; }
  pinocchio::Model& pino_model_fixed() { return pino_model_fixed_; }
  pinocchio::Data& pino_data_fixed() { return pino_data_fixed_; }
  std::string GetJointString(int index);

  // Robot cmds
  double vel_x_des_;
  double vel_y_des_;
  double omega_des_;
  double phase_;
  bool gait_enable_;
  bool emergency_;
  // Robot states
  double control_dt;
  Eigen::VectorXd q_rpy;
  Eigen::VectorXd q_pino;
  Eigen::VectorXd qdot;
  Eigen::VectorXd joint_vel_;
  Eigen::VectorXd qddot;
  Eigen::VectorXd q_desired;
  Eigen::VectorXd qdot_desired;
  Eigen::VectorXd q_fixed;
  Eigen::VectorXd qdot_fixed;
  Eigen::Matrix3d R_HW;
  Eigen::Matrix3d R_WH;
  Eigen::Matrix3d R_WB;
  Eigen::Matrix3d R_BW;
  Eigen::Matrix3d R_HB;
  Eigen::Vector3d omega_b;

  // Dynamic
  Eigen::MatrixXd M;
  Eigen::MatrixXd M_inverse;
  Eigen::VectorXd C;
  Eigen::VectorXd G;
  Eigen::MatrixXd S;
  Eigen::MatrixXd S_transpose_pinv;
  double mass;

  // State estimation
  Orientation ori_base_local_, ori_base_world_;
  Eigen::VectorXd observed_value;

  Eigen::Vector3d pos_left_wheel;
  Eigen::Vector3d pos_right_wheel;

 private:
  void AddFrames();
  void UpdateKinematic();
  void UpdateDynamic();
  // void UpdateMisc();
  void PublishRobotStates();
#if SIM_ENABLE
  // Mujoco
  const mjModel* mj_model_;
  const mjData* mj_data_;
  std::shared_ptr<ROS::Publisher> ros_publisher_;
  std::shared_ptr<JoyStick> joystick_;
#endif
  pinocchio::Model pino_model_;
  pinocchio::Data pino_data_;
  pinocchio::Model pino_model_fixed_;
  pinocchio::Data pino_data_fixed_;
  std::unique_ptr<JointVelEstimator> joint_vel_est_;
  std::unique_ptr<IMU> imu_;

  // IMU
  double yaw_offset_;
  bool yaw_offset_initialized_ = false;

#if PHYSICS_ENABLE
  standmode_output_t* standmode_output_;
  standmode_input_t* standmode_input_;
#endif
  YAML::Node config_;
};

}  // namespace l5a
#endif