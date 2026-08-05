#include "stand_mode.cc"
#include "ros_publisher.h"

#include <unistd.h>
#include <chrono>
#include <thread>

// std::shared_ptr<ROS::Publisher> ROS::SimPublisher::shared_instance_ =
// nullptr; std::mutex ROS::SimPublisher::mtx_;

void initialize_robot_staus(standmode_input_t& standmode_input) {
  standmode_input.IMU_signals.IMU_roll = 0;
  standmode_input.IMU_signals.IMU_pitch = 0;

  standmode_input.joints_status.joint_arm_l1.pos_fb = 0;
  standmode_input.joints_status.joint_arm_l2.pos_fb = 0;
  standmode_input.joints_status.joint_arm_l3.pos_fb = 0;
  standmode_input.joints_status.joint_arm_l4.pos_fb = 0;
  standmode_input.joints_status.joint_arm_l5.pos_fb = 0;
  standmode_input.joints_status.joint_arm_l6.pos_fb = 0;
  standmode_input.joints_status.joint_arm_l7.pos_fb = 0;

  standmode_input.joints_status.joint_h_pitch_l.pos_fb = 0;
  standmode_input.joints_status.joint_h_roll_l.pos_fb = 0;
  standmode_input.joints_status.joint_h_yaw_l.pos_fb = 0;
  standmode_input.joints_status.joint_k_pitch_l.pos_fb = 0;
  standmode_input.joints_status.joint_w_pitch_l.pos_fb = 0;

  standmode_input.joints_status.joint_arm_r1.pos_fb = 0;
  standmode_input.joints_status.joint_arm_r2.pos_fb = 0;
  standmode_input.joints_status.joint_arm_r3.pos_fb = 0;
  standmode_input.joints_status.joint_arm_r4.pos_fb = 0;
  standmode_input.joints_status.joint_arm_r5.pos_fb = 0;
  standmode_input.joints_status.joint_arm_r6.pos_fb = 0;
  standmode_input.joints_status.joint_arm_r7.pos_fb = 0;

  standmode_input.joints_status.joint_h_pitch_r.pos_fb = 0;
  standmode_input.joints_status.joint_h_roll_r.pos_fb = 0;
  standmode_input.joints_status.joint_h_yaw_r.pos_fb = 0;
  standmode_input.joints_status.joint_k_pitch_r.pos_fb = 0;
  standmode_input.joints_status.joint_w_pitch_r.pos_fb = 0;

  standmode_input.IMU_signals.IMU_wx_body = 0;
  standmode_input.IMU_signals.IMU_wy_body = 0;
  standmode_input.IMU_signals.IMU_wz_body = 0;

  standmode_input.joints_status.joint_arm_l1.vel_fb = 0;
  standmode_input.joints_status.joint_arm_l2.vel_fb = 0;
  standmode_input.joints_status.joint_arm_l3.vel_fb = 0;
  standmode_input.joints_status.joint_arm_l4.vel_fb = 0;
  standmode_input.joints_status.joint_arm_l5.vel_fb = 0;
  standmode_input.joints_status.joint_arm_l6.vel_fb = 0;
  standmode_input.joints_status.joint_arm_l7.vel_fb = 0;

  standmode_input.joints_status.joint_h_pitch_l.vel_fb = 0;
  standmode_input.joints_status.joint_h_roll_l.vel_fb = 0;
  standmode_input.joints_status.joint_h_yaw_l.vel_fb = 0;
  standmode_input.joints_status.joint_k_pitch_l.vel_fb = 0;
  standmode_input.joints_status.joint_w_pitch_l.vel_fb = 0;

  standmode_input.joints_status.joint_arm_r1.vel_fb = 0;
  standmode_input.joints_status.joint_arm_r2.vel_fb = 0;
  standmode_input.joints_status.joint_arm_r3.vel_fb = 0;
  standmode_input.joints_status.joint_arm_r4.vel_fb = 0;
  standmode_input.joints_status.joint_arm_r5.vel_fb = 0;
  standmode_input.joints_status.joint_arm_r6.vel_fb = 0;
  standmode_input.joints_status.joint_arm_r7.vel_fb = 0;

  standmode_input.joints_status.joint_h_pitch_r.vel_fb = 0;
  standmode_input.joints_status.joint_h_roll_r.vel_fb = 0;
  standmode_input.joints_status.joint_h_yaw_r.vel_fb = 0;
  standmode_input.joints_status.joint_k_pitch_r.vel_fb = 0;
  standmode_input.joints_status.joint_w_pitch_r.vel_fb = 0;
}

void exit_handler(int sig) {
  // std::cout << "Shutting down ROS..." << std::endl;
  std::cout << "exit_handler..." << std::endl;
  // rclcpp::shutdown();

  exit(0);
  // return;
}

int main(int argc, char* argv[]) {
  signal(SIGINT, exit_handler);
  signal(SIGTERM, exit_handler);

  // rclcpp::init(argc, argv);
  // standMode_initialize();

  standmode_output_t standmode_output;
  standmode_input_t standmode_input;
  initialize_robot_staus(standmode_input);

  const auto loop_interval = std::chrono::microseconds(2000);  // 2ms = 2000us

  // static L2C::RobotModel l2c_model;
  // static L2C::L2C_FSM fsm(l2c_model);

  // auto message_handle = ROS::Publisher::getInstance();
  // auto message_handle = std::make_shared<ROS::SimPublisher>("stand_mode");
  // auto spin_func = [](std::shared_ptr<ROS::Publisher> node_ptr) {
  // rclcpp::spin(node_ptr); }; auto spin_thread = std::thread(spin_func,
  // message_handle);

  while (true) {
    auto start_time = std::chrono::steady_clock::now();

    standMode_step(&standmode_output, &standmode_input);
    // l2c_model.UpdateModel(&standmode_output, &standmode_input);
    // fsm.Run(l2c_model);

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    // Ensure the loop runs at 500Hz
    if (elapsed_time < loop_interval) {
      std::this_thread::sleep_for(loop_interval - elapsed_time);
    }
  }

  standMode_terminate();
  // rclcpp::shutdown();
  return 0;
}
