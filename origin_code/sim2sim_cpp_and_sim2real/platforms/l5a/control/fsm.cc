#include "fsm.h"

namespace l5a {

FSM::FSM(RobotModel& robot_model) {
  last_fsm_id_ = -1;
  state_changed_ = false;
  EDamp_signal_ = false;
  tau_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);
  pos_fb_kp_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);
  pos_fb_kd_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);
  rl_ = std::make_shared<RL>(robot_model);
  safe_mode_ = true;
#if SIM_ENABLE
  ros_publisher_ = ROS::Publisher::getInstance();
#endif
}

FSM::~FSM() {}

void FSM::Initialize(RobotModel& robot_model) {
  EDamp_signal_ = false;
  tau_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);
  pos_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);
  vel_ = Eigen::VectorXd::Zero(robot_model.pino_model().nv - 6);
}

void FSM::Run(RobotModel& robot_model) {
  auto start = std::chrono::high_resolution_clock::now();
  // fsm_id_ = static_cast<int>(FsmId::Rl);  // Safe state
  if (!CheckSafety(robot_model)) {
  // if (true) {
    fsm_id_ = static_cast<int>(FsmId::Rl);  // Safe state
  } else {
    fsm_id_ = static_cast<int>(FsmId::RlEDamp);
  }
  switch (fsm_id_) {
    case static_cast<int>(FsmId::Rl):
      rl_->Run(robot_model);
      tau_ = rl_->tau();
      pos_ = rl_->pos_target();
      vel_ = rl_->vel_target();
      pos_fb_kp_ = rl_->pos_fb_kp_;
      pos_fb_kd_ = rl_->pos_fb_kd_;
      break;

    case static_cast<int>(FsmId::RlEDamp):
      rl_->RunEDamp(robot_model);
      robot_model.emergency_ = true;
      // std::cout << "emer:" << robot_model.emergency_ << std::endl;
      tau_ = rl_->tau();
      break;

    default:
      break;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  double time_ns = duration.count();
  double time_ms = time_ns / 1000000;
  // robot_model.observed_value[24] = time_ms;
}

bool FSM::CheckSafety(RobotModel& robot_model) {
  std::vector<Limits> joint_limits = {
      {AllJoints::left_hip_roll_joint, -0.384 + 0.05, 1.396 - 0.05, -16, 16, -90, 90},  {AllJoints::left_hip_pitch_joint, -0.908 + 0.05, 1.396 - 0.05, -16, 16, -90, 90},
      {AllJoints::left_knee_joint, -1.309 + 0.05, 0.262 - 0.05, -14, 14, -130, 130},      {AllJoints::left_wheel_joint, -1000, 1000, -15, 15, -60, 60},
      {AllJoints::right_hip_roll_joint, -1.396 + 0.05, 0.384 - 0.05, -16, 16, -90, 90}, {AllJoints::right_hip_pitch_joint, -0.908 + 0.05, 1.396 - 0.05, -16, 16, -90, 90},
      {AllJoints::right_knee_joint, -1.309 + 0.05, 0.262 - 0.05, -14, 14, -130, 130},     {AllJoints::right_wheel_joint, -1000, 1000, -15, 15, -60, 60}};
  // std::vector<Limits> joint_limits = {
  //     {AllJoints::left_hip_roll_joint, -0.35, 0.5, -16, 16, -90, 90},  {AllJoints::left_hip_pitch_joint, -0.5, 1.2, -16, 16, -90, 90},
  //     {AllJoints::left_knee_joint, -1.3, 0.262, -14, 14, -130, 130},      {AllJoints::left_wheel_joint, -1000, 1000, -15, 15, -60, 60},
  //     {AllJoints::right_hip_roll_joint, -0.5, 0.35, -16, 16, -90, 90}, {AllJoints::right_hip_pitch_joint, -0.5, 1.2, -16, 16, -90, 90},
  //     {AllJoints::right_knee_joint, -1.3, 0.262, -14, 14, -130, 130},     {AllJoints::right_wheel_joint, -1000, 1000, -15, 15, -60, 60}};

  for (const auto& limit : joint_limits) {
    int index = static_cast<int>(limit.joint);
    // std::cout << "the index is:" << index << "the ang is:" << robot_model.q_rpy[index] << std::endl;
    if ((robot_model.q_rpy[index] < limit.q_min || robot_model.q_rpy[index] > limit.q_max) && (index != static_cast<int>(AllJoints::left_wheel_joint)) &&
        (index != static_cast<int>(AllJoints::right_wheel_joint))) {
      LOG(ERROR) << "q of Joint " << robot_model.GetJointString(index) << " is out of range!"
                 << " q is " << robot_model.q_rpy[index] << "\n";
      LOG(ERROR) << "Robot is in unsafe state!!!";
      EDamp_signal_ = true;
      return EDamp_signal_;
    }
    if (robot_model.qdot[index] < limit.qdot_min || robot_model.qdot[index] > limit.qdot_max) {
      LOG(ERROR) << "qdot of Joint " << robot_model.GetJointString(index) << " is out of range!"
                 << " qdot is " << robot_model.qdot[index] << "\n";
      LOG(ERROR) << "Robot is in unsafe state!!!";
      // EDamp_signal_ = true;
      return EDamp_signal_;
    }
    // if(index == static_cast<int>(AllJoints::left_wheel_joint) || index == static_cast<int>(AllJoints::right_wheel_joint)){
    //    std::cout << "q_dot of wheel:" << robot_model.qdot[index] << std::endl;
    // }

    if (tau_[index - 6] < limit.tau_min || tau_[index - 6] > limit.tau_max) {
      LOG(WARNING) << "tau of Joint " << robot_model.GetJointString(index) << " is out of range!"
                   << " tau is " << tau_[index - 6] << "\n";
      tau_[index - 6] = std::clamp(tau_[index - 6], limit.tau_min, limit.tau_max);
      LOG(ERROR) << "Robot is in unsafe state!!!";
      // EDamp_signal_ = true;
      return EDamp_signal_;
    }
  }
  return EDamp_signal_;
}

}  // namespace l5a