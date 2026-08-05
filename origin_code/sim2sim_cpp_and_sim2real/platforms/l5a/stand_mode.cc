#include <thread>

#include "fsm.h"
#include "robot_model.h"
#include "standMode_types.h"
#include "ros_publisher.h"

#include <rclcpp/rclcpp.hpp>

#define _USE_MATH_DEFINES
#include <cmath>
//加载urdf并初始化robot_model和fsm
std::string urdf_path = "/sim/model/l5a/urdf/l5aurdf20260521.urdf";
l5a::RobotModel robot_model(urdf_path);
l5a::FSM fsm(robot_model);

//控制周期观测
double call_time_ms = 0.0;  // Initialize time_ms to 0.0
auto last_time = std::chrono::high_resolution_clock::now();
//控制信号声明及初始化,期望速度/角速度/运行强化学习算法
struct CtlSigs {
  double vel_x_des;
  double vel_y_des;
  double omega_des;
  bool rl_run;          //运行强化学习算法
  bool gait_enable;     //步态使能
  double gait_phase;    //步态相位
  double gait_counter;  //步态相位计数器
}; 
CtlSigs ctlSigs{0.0, 0.0, 0.0, false, false, 0.0, 0.0};
//手柄信号处理,速度信号平滑,观测值处理函数声明
void handleSigs_processing(standmode_output_t* standmode_output, const standmode_input_t* standmode_input, CtlSigs* ctlSigs, l5a::FSM* fsm);
float updateWithSmartBrake(float current, float target, float a_up, float a_down, float dt);
void displayObservedValue(standmode_output_t* standmode_output, const Eigen::VectorXd& observed_value);
//重启算法初始化函数
void standMode_initialize(standmode_output_t* standmode_output, standmode_input_t* standmode_input) {
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  robot_model.Initialize();
  fsm.Initialize(robot_model);
  ctlSigs.rl_run = false;
  ctlSigs.gait_enable = false;
  return;
}
//设置电机参数
void setMotorParameters(standmode_output_t* standmode_output, standmode_input_t* standmode_input) {
  // 全部电机使能
  standmode_output->joints_cmd.joint_h_roll_l.enable = 1;
  standmode_output->joints_cmd.joint_h_pitch_l.enable = 1;
  standmode_output->joints_cmd.joint_k_pitch_l.enable = 1;
  standmode_output->joints_cmd.joint_w_pitch_l.enable = 1;
  standmode_output->joints_cmd.joint_h_roll_r.enable = 1;
  standmode_output->joints_cmd.joint_h_pitch_r.enable = 1;
  standmode_output->joints_cmd.joint_k_pitch_r.enable = 1;
  standmode_output->joints_cmd.joint_w_pitch_r.enable = 1;

  // standmode_output->joints_cmd.joint_h_roll_l.enable = 0;
  // standmode_output->joints_cmd.joint_h_pitch_l.enable = 0;
  // standmode_output->joints_cmd.joint_k_pitch_l.enable = 0;
  // standmode_output->joints_cmd.joint_w_pitch_l.enable = 0;
  // standmode_output->joints_cmd.joint_h_roll_r.enable = 0;
  // standmode_output->joints_cmd.joint_h_pitch_r.enable = 0;
  // standmode_output->joints_cmd.joint_k_pitch_r.enable = 0;
  // standmode_output->joints_cmd.joint_w_pitch_r.enable = 0;

  // 设置模式：操作模式为1是位置模式，3是速度模式，4是力矩模式，11是力位混合模式
  // standmode_output->joints_cmd.joint_h_roll_l.operation_mode = 1;
  // standmode_output->joints_cmd.joint_h_pitch_l.operation_mode = 1;
  // standmode_output->joints_cmd.joint_k_pitch_l.operation_mode = 1;
  // standmode_output->joints_cmd.joint_w_pitch_l.operation_mode = 3;
  // standmode_output->joints_cmd.joint_h_roll_r.operation_mode = 1;
  // standmode_output->joints_cmd.joint_h_pitch_r.operation_mode = 1;
  // standmode_output->joints_cmd.joint_k_pitch_r.operation_mode = 1;
  // standmode_output->joints_cmd.joint_w_pitch_r.operation_mode = 3;

  // 设置模式：操作模式为1是位置模式，3是速度模式，4是力矩模式，11是力位混合模式
  // standmode_output->joints_cmd.joint_h_roll_l.operation_mode = 4;
  // standmode_output->joints_cmd.joint_h_pitch_l.operation_mode = 4;
  // standmode_output->joints_cmd.joint_k_pitch_l.operation_mode = 4;
  // standmode_output->joints_cmd.joint_w_pitch_l.operation_mode = 4;
  // standmode_output->joints_cmd.joint_h_roll_r.operation_mode = 4;
  // standmode_output->joints_cmd.joint_h_pitch_r.operation_mode = 4;
  // standmode_output->joints_cmd.joint_k_pitch_r.operation_mode = 4;
  // standmode_output->joints_cmd.joint_w_pitch_r.operation_mode = 4;
  //轮子力矩模式,其他关节电机力位混合模式
  standmode_output->joints_cmd.joint_h_roll_l.operation_mode = 11;
  standmode_output->joints_cmd.joint_h_pitch_l.operation_mode = 11;
  standmode_output->joints_cmd.joint_k_pitch_l.operation_mode = 11;
  standmode_output->joints_cmd.joint_w_pitch_l.operation_mode = 4;
  standmode_output->joints_cmd.joint_h_roll_r.operation_mode = 11;
  standmode_output->joints_cmd.joint_h_pitch_r.operation_mode = 11;
  standmode_output->joints_cmd.joint_k_pitch_r.operation_mode = 11;
  standmode_output->joints_cmd.joint_w_pitch_r.operation_mode = 4;
}

void setEmergencyParameters(standmode_output_t* standmode_output, standmode_input_t* standmode_input) {
  // 全部电机使能
  standmode_output->joints_cmd.joint_h_roll_l.enable = 1;
  standmode_output->joints_cmd.joint_h_pitch_l.enable = 1;
  standmode_output->joints_cmd.joint_k_pitch_l.enable = 1;
  standmode_output->joints_cmd.joint_w_pitch_l.enable = 1;
  standmode_output->joints_cmd.joint_h_roll_r.enable = 1;
  standmode_output->joints_cmd.joint_h_pitch_r.enable = 1;
  standmode_output->joints_cmd.joint_k_pitch_r.enable = 1;
  standmode_output->joints_cmd.joint_w_pitch_r.enable = 1;

  //轮子力矩模式,其他关节电机力位混合模式
  standmode_output->joints_cmd.joint_h_roll_l.operation_mode = 3;
  standmode_output->joints_cmd.joint_h_pitch_l.operation_mode = 3;
  standmode_output->joints_cmd.joint_k_pitch_l.operation_mode = 3;
  standmode_output->joints_cmd.joint_w_pitch_l.operation_mode = 3;
  standmode_output->joints_cmd.joint_h_roll_r.operation_mode = 3;
  standmode_output->joints_cmd.joint_h_pitch_r.operation_mode = 3;
  standmode_output->joints_cmd.joint_k_pitch_r.operation_mode = 3;
  standmode_output->joints_cmd.joint_w_pitch_r.operation_mode = 3;
}

void standMode_step(standmode_output_t* standmode_output, standmode_input_t* standmode_input) {
  auto call_time = std::chrono::high_resolution_clock::now();
  auto call_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(call_time - last_time);
  double call_time_ns = call_duration.count();
  call_time_ms = call_time_ns / 1000000;
  last_time = call_time;
  robot_model.observed_value[30] = call_time_ms;

  setMotorParameters(standmode_output, standmode_input);                                                                //设置电机参数
  robot_model.UpdateRealJointStates(standmode_output, standmode_input);                                                 //更新反馈关节状态
  robot_model.UpdateModel();                                                                                            //更新模型
  handleSigs_processing(standmode_output, standmode_input, &ctlSigs, &fsm);                                             //处理手柄信号
  robot_model.vel_x_des_ = updateWithSmartBrake(robot_model.vel_x_des_, ctlSigs.vel_x_des, 0.5, 1.5, robot_model.control_dt); //获取期望速度与角速度
  robot_model.vel_y_des_ = ctlSigs.vel_y_des;
  // robot_model.vel_x_des_ = ctlSigs.vel_x_des;
  robot_model.omega_des_ = ctlSigs.omega_des;
  robot_model.gait_enable_ = ctlSigs.gait_enable;
  robot_model.phase_ = ctlSigs.gait_phase;
  Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);         //下发的期望力矩命令
  Eigen::VectorXd pos_cmd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);         //下发的期望位置命令
  Eigen::VectorXd vel_cmd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);         //下发的速度命令
  Eigen::VectorXd pos_fb_kp = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);       //力位混合模式下,各关节的位置误差kp参数
  Eigen::VectorXd pos_fb_kd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);       //力位混合模式下,各关节的位置误差kd参数

  fsm.Run(robot_model);                           //运行状态机,更新控制策略
  tau_cmd = fsm.tau();                            //获取下发期望力矩命令
  vel_cmd = fsm.vel();
  pos_cmd = fsm.pos();                            //获取下发期望位置命令
  pos_fb_kp = fsm.pos_fb_kp_;                     //获取各关节位置误差kp参数
  pos_fb_kd = fsm.pos_fb_kd_;                     //获取各关节位置误差kd参数
  // pos_cmd << 0, 0.261799, -0.508606, 0.0, 0, 0.261799, -0.508606, 0.0;

  // robot_model.observed_value[14] = tau_cmd[0];     //观测力矩值
  // robot_model.observed_value[15] = tau_cmd[1];
  // robot_model.observed_value[16] = tau_cmd[2];
  // robot_model.observed_value[17] = tau_cmd[3];
  // robot_model.observed_value[18] = tau_cmd[4];
  // robot_model.observed_value[19] = tau_cmd[5];

  standmode_output->joints_cmd.joint_h_roll_l.KP = pos_fb_kp[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_l.KP = pos_fb_kp[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_l.KP = pos_fb_kp[static_cast<int>(l5a::Joints::left_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_l.KP = pos_fb_kp[static_cast<int>(l5a::Joints::left_wheel_joint)];
  standmode_output->joints_cmd.joint_h_roll_r.KP = pos_fb_kp[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_r.KP = pos_fb_kp[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_r.KP = pos_fb_kp[static_cast<int>(l5a::Joints::right_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_r.KP = pos_fb_kp[static_cast<int>(l5a::Joints::right_wheel_joint)];

  standmode_output->joints_cmd.joint_h_roll_l.KD = pos_fb_kd[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_l.KD = pos_fb_kd[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_l.KD = pos_fb_kd[static_cast<int>(l5a::Joints::left_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_l.KD = pos_fb_kd[static_cast<int>(l5a::Joints::left_wheel_joint)];
  standmode_output->joints_cmd.joint_h_roll_r.KD = pos_fb_kd[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_r.KD = pos_fb_kd[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_r.KD = pos_fb_kd[static_cast<int>(l5a::Joints::right_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_r.KD = pos_fb_kd[static_cast<int>(l5a::Joints::right_wheel_joint)];;
  // position
  standmode_output->joints_cmd.joint_h_roll_l.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_l.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_l.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::left_knee_joint)];
  standmode_output->joints_cmd.joint_h_roll_r.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_r.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_r.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::right_knee_joint)];
  // standmode_output->joints_cmd.joint_h_roll_l.pos_cmd = 0.0523599;
  // standmode_output->joints_cmd.joint_h_pitch_l.pos_cmd = 0.261799;
  // standmode_output->joints_cmd.joint_k_pitch_l.pos_cmd = -0.563811;
  // standmode_output->joints_cmd.joint_h_roll_r.pos_cmd = -0.0523599;
  // standmode_output->joints_cmd.joint_h_pitch_r.pos_cmd = 0.261799;
  // standmode_output->joints_cmd.joint_k_pitch_r.pos_cmd = -0.563811;
  // velocity
  standmode_output->joints_cmd.joint_h_roll_l.vel_cmd = 0;
  standmode_output->joints_cmd.joint_h_pitch_l.vel_cmd = 0;
  standmode_output->joints_cmd.joint_k_pitch_l.vel_cmd = 0;
  standmode_output->joints_cmd.joint_w_pitch_l.vel_cmd = 0;
  // standmode_output->joints_cmd.joint_w_pitch_l.vel_cmd = vel_cmd[static_cast<int>(l5a::Joints::left_wheel_joint)];
  standmode_output->joints_cmd.joint_h_roll_r.vel_cmd = 0;
  standmode_output->joints_cmd.joint_h_pitch_r.vel_cmd = 0;
  standmode_output->joints_cmd.joint_k_pitch_r.vel_cmd = 0;
 standmode_output->joints_cmd.joint_w_pitch_r.vel_cmd = 0;
  // standmode_output->joints_cmd.joint_w_pitch_r.vel_cmd = vel_cmd[static_cast<int>(l5a::Joints::right_wheel_joint)];

  standmode_output->joints_cmd.joint_h_roll_l.torque_cmd = 0;
  standmode_output->joints_cmd.joint_h_pitch_l.torque_cmd = 0;
  standmode_output->joints_cmd.joint_k_pitch_l.torque_cmd = 0;
  // standmode_output->joints_cmd.joint_w_pitch_l.torque_cmd = 0;
  standmode_output->joints_cmd.joint_w_pitch_l.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::left_wheel_joint)];

  standmode_output->joints_cmd.joint_h_roll_r.torque_cmd = 0;
  standmode_output->joints_cmd.joint_h_pitch_r.torque_cmd = 0;
  standmode_output->joints_cmd.joint_k_pitch_r.torque_cmd = 0;
  // standmode_output->joints_cmd.joint_w_pitch_r.torque_cmd = 0;
  standmode_output->joints_cmd.joint_w_pitch_r.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::right_wheel_joint)];
  // torque
  // standmode_output->joints_cmd.joint_h_roll_l.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  // standmode_output->joints_cmd.joint_h_pitch_l.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  // standmode_output->joints_cmd.joint_k_pitch_l.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::left_knee_joint)];
  // standmode_output->joints_cmd.joint_w_pitch_l.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::left_wheel_joint)];

  // standmode_output->joints_cmd.joint_h_roll_r.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  // standmode_output->joints_cmd.joint_h_pitch_r.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  // standmode_output->joints_cmd.joint_k_pitch_r.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::right_knee_joint)];
  // standmode_output->joints_cmd.joint_w_pitch_r.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::right_wheel_joint)];

  if(robot_model.emergency_){
    setEmergencyParameters(standmode_output, standmode_input);
  }
  // std::cout << "pos_cmd" << pos_cmd.transpose() << std::endl;

  // robot_model.observed_value[1] = robot_model.q_rpy[3];//roll
  // robot_model.observed_value[2] = robot_model.q_rpy[4];//pitch
  // // std::cout << "pitch:" << robot_model.q_rpy[4] << std::endl;
  // robot_model.observed_value[3] = pos_cmd[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  // robot_model.observed_value[4] = standmode_input->joints_status.joint_h_roll_l.pos_fb;
  // robot_model.observed_value[5] = pos_cmd[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  // robot_model.observed_value[6] = standmode_input->joints_status.joint_h_roll_r.pos_fb;
  // robot_model.observed_value[7] = pos_cmd[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  // robot_model.observed_value[8] = standmode_input->joints_status.joint_h_pitch_l.pos_fb;
  // robot_model.observed_value[9] = pos_cmd[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  // robot_model.observed_value[10] = standmode_input->joints_status.joint_h_pitch_r.pos_fb;
  // robot_model.observed_value[11] = pos_cmd[static_cast<int>(l5a::Joints::left_knee_joint)];
  // robot_model.observed_value[12] = standmode_input->joints_status.joint_k_pitch_l.pos_fb;
  // robot_model.observed_value[13] = pos_cmd[static_cast<int>(l5a::Joints::right_knee_joint)];
  // robot_model.observed_value[14] = standmode_input->joints_status.joint_k_pitch_r.pos_fb;
  // robot_model.observed_value[15] = tau_cmd[static_cast<int>(l5a::Joints::left_wheel_joint)];
  // robot_model.observed_value[16] = standmode_input->joints_status.joint_w_pitch_l.pos_fb * 2.1;
  // robot_model.observed_value[17] = tau_cmd[static_cast<int>(l5a::Joints::right_wheel_joint)];
  // robot_model.observed_value[18] = standmode_input->joints_status.joint_w_pitch_r.pos_fb * 2.1;
  // robot_model.observed_value[3] = tau_cmd[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  // robot_model.observed_value[4] = standmode_input->joints_status.joint_h_roll_l.torque_fb * 2.35;
  // robot_model.observed_value[5] = tau_cmd[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  // robot_model.observed_value[6] = standmode_input->joints_status.joint_h_roll_r.torque_fb * 2.35;
  // robot_model.observed_value[7] = tau_cmd[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  // robot_model.observed_value[8] = standmode_input->joints_status.joint_h_pitch_l.torque_fb * 2.35;
  // robot_model.observed_value[9] = tau_cmd[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  // robot_model.observed_value[10] = standmode_input->joints_status.joint_h_pitch_r.torque_fb * 2.35;
  // robot_model.observed_value[11] = tau_cmd[static_cast<int>(l5a::Joints::left_knee_joint)];
  // robot_model.observed_value[12] = standmode_input->joints_status.joint_k_pitch_l.torque_fb * 2.35;
  // robot_model.observed_value[13] = tau_cmd[static_cast<int>(l5a::Joints::right_knee_joint)];
  // robot_model.observed_value[14] = standmode_input->joints_status.joint_k_pitch_r.torque_fb * 2.35;
  // robot_model.observed_value[15] = tau_cmd[static_cast<int>(l5a::Joints::left_wheel_joint)];
  // robot_model.observed_value[16] = standmode_input->joints_status.joint_w_pitch_l.torque_fb * 2.1;
  // robot_model.observed_value[17] = tau_cmd[static_cast<int>(l5a::Joints::right_wheel_joint)];
  // robot_model.observed_value[18] = standmode_input->joints_status.joint_w_pitch_r.torque_fb * 2.1;

  // robot_model.observed_value[19] = robot_model.q_rpy[11];
  // robot_model.observed_value[20] = robot_model.q_rpy[12];
  // robot_model.observed_value[21] = robot_model.q_rpy[13];
  // record data


  // record data
  displayObservedValue(standmode_output, robot_model.observed_value);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto step_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - call_time);
  double step_time_ns = step_duration.count();
  double step_time_ms = step_time_ns / 1000000;
  robot_model.observed_value[29] = step_time_ms;
}

void standMode_terminate(void) { return; }

void handleSigs_processing(standmode_output_t* standmode_output, const standmode_input_t* standmode_input, CtlSigs* ctlSigs, l5a::FSM* fsm) {
  // 对手柄信号进行处理
  double handle_A = 0;   // 手柄A键
  double handle_RT = 0;  // 手柄RT键
  double handle_B = 0;   // 手柄B键
  double handle_Y = 0;   // 手柄Y键
  double handle_RB = 0;  // 手柄RB键
  double handle_LB = 0;  // 手柄LB键
  double handle_LT = 0;  // 手柄LT键
  if (standmode_input->Handle_signals.a2 >= 0.5) {
    handle_LT = 1.0;
  } else {
    handle_LT = 0;
  }
  if (standmode_input->Handle_signals.a5 >= 0.5) {
    handle_RT = 1.0;
  } else {
    handle_RT = 0;
  }
  if (standmode_input->Handle_signals.b0 >= 0.5) {
    handle_A = 1.0;
  } else {
    handle_A = 0;
  }
  if (standmode_input->Handle_signals.b1 >= 0.5) {
    handle_B = 1.0;
  } else {
    handle_B = 0;
  }
  if (standmode_input->Handle_signals.b3 >= 0.5) {
    handle_Y = 1.0;
  } else {
    handle_Y = 0;
  }
  if (standmode_input->Handle_signals.b4 >= 0.5) {
    handle_LB = 1.0;
  } else {
    handle_LB = 0;
  }
  if (standmode_input->Handle_signals.b5 >= 0.5) {
    handle_RB = 1.0;
  } else {
    handle_RB = 0;
  }

if (std::abs(standmode_input->Handle_signals.a1 - 0.5) <= 0.1 || handle_LB) {
    ctlSigs->vel_x_des = 0.0;
  } else {
    ctlSigs->vel_x_des = (standmode_input->Handle_signals.a1 - 0.5) * (-1.0) * handle_LT;
  }

  if (std::abs(standmode_input->Handle_signals.a0 - 0.5) <= 0.1 || handle_LB) {
    ctlSigs->vel_y_des = 0.0;
  } else {
    ctlSigs->vel_y_des = (standmode_input->Handle_signals.a0 - 0.5) * (-1.0) * handle_LT;
  }

  if (std::abs(standmode_input->Handle_signals.a3 - 0.5) <= 0.1 || handle_RB) {
    ctlSigs->omega_des = 0.0;
  } else {
    ctlSigs->omega_des = (standmode_input->Handle_signals.a3 - 0.5) * (-1.0) * handle_LT;
  }

  if (handle_RT) {
    ctlSigs->gait_enable = true;
    ctlSigs->gait_phase = std::fmod(ctlSigs->gait_counter * 0.002, 0.8) / 0.8;
    ctlSigs->gait_counter += 1;
  } else {
    ctlSigs->gait_enable = false;
    ctlSigs->gait_counter = 0;
  }

  // 定义模态切换按钮
  if (handle_A && handle_B) {
    ctlSigs->rl_run = true;
  }

}

void displayObservedValue(standmode_output_t* standmode_output, const Eigen::VectorXd& observed_value) {
  standmode_output->observed_value.channel_1 = observed_value[1];
  standmode_output->observed_value.channel_2 = observed_value[2];
  standmode_output->observed_value.channel_3 = observed_value[3];
  standmode_output->observed_value.channel_4 = observed_value[4];
  standmode_output->observed_value.channel_5 = observed_value[5];
  standmode_output->observed_value.channel_6 = observed_value[6];
  standmode_output->observed_value.channel_7 = observed_value[7];
  standmode_output->observed_value.channel_8 = observed_value[8];
  standmode_output->observed_value.channel_9 = observed_value[9];
  standmode_output->observed_value.channel_10 = observed_value[10];
  standmode_output->observed_value.channel_11 = observed_value[11];
  standmode_output->observed_value.channel_12 = observed_value[12];
  standmode_output->observed_value.channel_13 = observed_value[13];
  standmode_output->observed_value.channel_14 = observed_value[14];
  standmode_output->observed_value.channel_15 = observed_value[15];
  standmode_output->observed_value.channel_16 = observed_value[16];
  standmode_output->observed_value.channel_17 = observed_value[17];
  standmode_output->observed_value.channel_18 = observed_value[18];
  standmode_output->observed_value.channel_19 = observed_value[19];
  standmode_output->observed_value.channel_20 = observed_value[20];

  standmode_output->observed_value.channel_21 = observed_value[21];
  standmode_output->observed_value.channel_22 = observed_value[22];
  standmode_output->observed_value.channel_23 = observed_value[23];
  standmode_output->observed_value.channel_24 = observed_value[24];
  standmode_output->observed_value.channel_25 = observed_value[25];
  standmode_output->observed_value.channel_26 = observed_value[26];
  standmode_output->observed_value.channel_27 = observed_value[27];
  standmode_output->observed_value.channel_28 = observed_value[28];
  standmode_output->observed_value.channel_29 = observed_value[29];
  standmode_output->observed_value.channel_30 = observed_value[30];
  standmode_output->observed_value.channel_31 = observed_value[31];
  standmode_output->observed_value.channel_32 = observed_value[32];
  standmode_output->observed_value.channel_33 = observed_value[33];
  standmode_output->observed_value.channel_34 = observed_value[34];
  standmode_output->observed_value.channel_35 = observed_value[35];
  standmode_output->observed_value.channel_36 = observed_value[36];
  standmode_output->observed_value.channel_37 = observed_value[37];
  standmode_output->observed_value.channel_38 = observed_value[38];
  standmode_output->observed_value.channel_39 = observed_value[39];
  standmode_output->observed_value.channel_40 = observed_value[40];

  standmode_output->observed_value.channel_41 = observed_value[41];
  standmode_output->observed_value.channel_42 = observed_value[42];
  standmode_output->observed_value.channel_43 = observed_value[43];
  standmode_output->observed_value.channel_44 = observed_value[44];
  standmode_output->observed_value.channel_45 = observed_value[45];
  standmode_output->observed_value.channel_46 = observed_value[46];
  standmode_output->observed_value.channel_47 = observed_value[47];
  standmode_output->observed_value.channel_48 = observed_value[48];
  standmode_output->observed_value.channel_49 = observed_value[49];
  standmode_output->observed_value.channel_50 = observed_value[50];
  standmode_output->observed_value.channel_51 = observed_value[51];
  standmode_output->observed_value.channel_52 = observed_value[52];
  standmode_output->observed_value.channel_53 = observed_value[53];
  standmode_output->observed_value.channel_54 = observed_value[54];
  standmode_output->observed_value.channel_55 = observed_value[55];
  standmode_output->observed_value.channel_56 = observed_value[56];
  standmode_output->observed_value.channel_57 = observed_value[57];
  standmode_output->observed_value.channel_58 = observed_value[58];
  standmode_output->observed_value.channel_59 = observed_value[59];
  standmode_output->observed_value.channel_60 = observed_value[60];

  standmode_output->observed_value.channel_61 = observed_value[61];
  standmode_output->observed_value.channel_62 = observed_value[62];
  standmode_output->observed_value.channel_63 = observed_value[63];
  standmode_output->observed_value.channel_64 = observed_value[64];
  standmode_output->observed_value.channel_65 = observed_value[65];
  standmode_output->observed_value.channel_66 = observed_value[66];
  standmode_output->observed_value.channel_67 = observed_value[67];
  standmode_output->observed_value.channel_68 = observed_value[68];
  standmode_output->observed_value.channel_69 = observed_value[69];
  standmode_output->observed_value.channel_70 = observed_value[70];
  standmode_output->observed_value.channel_71 = observed_value[71];
  standmode_output->observed_value.channel_72 = observed_value[72];
  standmode_output->observed_value.channel_73 = observed_value[73];
  standmode_output->observed_value.channel_74 = observed_value[74];
  standmode_output->observed_value.channel_75 = observed_value[75];
  standmode_output->observed_value.channel_76 = observed_value[76];
  standmode_output->observed_value.channel_77 = observed_value[77];
  // standmode_output->observed_value.channel_78 = observed_value[78];//78通道控制支撑架
  standmode_output->observed_value.channel_79 = observed_value[79];
  standmode_output->observed_value.channel_80 = observed_value[80];
}

float updateWithSmartBrake(float current, float target, float a_up, float a_down, float dt) {
  float dv = target - current;
  float step;

  // 情况 1：符号不同（异号），说明要先刹到 0
  if (current * target < 0) {
    // 强制把目标临时设为 0，用大减速度逼近
    float dv0 = 0 - current;
    step = a_down * dt;

    if (fabs(dv0) <= step) {
      return 0.0f;  // 到零
    } else {
      return current + step * (dv0 > 0 ? 1 : -1);
    }
  }

  // 情况 2：符号相同
  if (fabs(target) > fabs(current)) {
    // 加速
    step = a_up * dt;
  } else if (fabs(target) < fabs(current)) {
    // 减速
    step = a_down * dt;
  } else {
    return target;
  }

  if (fabs(dv) <= step) {
    return target;
  } else {
    return current + step * (dv > 0 ? 1 : -1);
  }
}