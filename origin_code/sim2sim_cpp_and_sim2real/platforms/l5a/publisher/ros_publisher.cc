#include "ros_publisher.h"
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
namespace ROS {

Publisher::Publisher() : Node("SimPublisher_l4a") {
  // Glog
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  msg_queue_size_ = 10;

  RCLCPP_INFO(this->get_logger(), "Start SimPublisher ...");
}

rclcpp::QoS Publisher::GetBestEffortQoS(int queue_size) {
  rclcpp::QoS qos(queue_size);
  qos.best_effort();
  return qos;
}

void Publisher::Publish(const std::string& topic_name,
                        Eigen::VectorXd joint_vector) {
  switch (joint_vector.size()) {
    case 1:
      if (vector1d_publisher_.find(topic_name) != vector1d_publisher_.end()) {
        vector_callback(topic_name, joint_vector);
        return;
      }
      vector1d_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::Vector1d>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      vector_callback(topic_name, joint_vector);
      break;
    case 2:
      if (vector2d_publisher_.find(topic_name) != vector2d_publisher_.end()) {
        vector_callback(topic_name, joint_vector);
        return;
      }
      vector2d_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::Vector2d>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      vector_callback(topic_name, joint_vector);
      break;
    case 3:
      if (vector3d_publisher_.find(topic_name) != vector3d_publisher_.end()) {
        vector_callback(topic_name, joint_vector);
        return;
      }
      vector3d_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::Vector3d>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      vector_callback(topic_name, joint_vector);
      break;
    case 4:
      if (vector4d_publisher_.find(topic_name) != vector4d_publisher_.end()) {
        vector_callback(topic_name, joint_vector);
        return;
      }
      vector4d_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::Vector4d>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      vector_callback(topic_name, joint_vector);
      break;

    case 6:
      if (vector6d_publisher_.find(topic_name) != vector6d_publisher_.end()) {
        vector_callback(topic_name, joint_vector);
        return;
      }
      vector6d_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::Vector6d>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      vector_callback(topic_name, joint_vector);
      break;

    case 7:
      if (vector7d_publisher_.find(topic_name) != vector7d_publisher_.end()) {
        vector_callback(topic_name, joint_vector);
        return;
      }
      vector7d_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::Vector7d>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      vector_callback(topic_name, joint_vector);
      break;

    case 8:
      if (vector8d_publisher_.find(topic_name) != vector8d_publisher_.end()) {
        vector_callback(topic_name, joint_vector);
        return;
      }
      vector8d_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::Vector8d>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      vector_callback(topic_name, joint_vector);
      break;

    case 9:
      if (vector9d_publisher_.find(topic_name) != vector9d_publisher_.end()) {
        vector_callback(topic_name, joint_vector);
        return;
      }
      vector9d_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::Vector9d>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      vector_callback(topic_name, joint_vector);
      break;

    case 10:
      if (all_joint_publisher_.find(topic_name) != all_joint_publisher_.end()) {
        all_joints_callback(topic_name, joint_vector);
        return;
      }
      all_joint_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::AllJointStates>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      all_joints_callback(topic_name, joint_vector);
      break;

    case 12:
      if (state_estimator_publisher_.find(topic_name) !=
          state_estimator_publisher_.end()) {
        state_estimator_callback(topic_name, joint_vector);
        return;
      }
      state_estimator_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::StateEstimator>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      state_estimator_callback(topic_name, joint_vector);
      break;

    case 16:
      if (all_joint_with_base_publisher_.find(topic_name) !=
          all_joint_with_base_publisher_.end()) {
        all_joints_with_base_callback(topic_name, joint_vector);
        return;
      }
      all_joint_with_base_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::AllJointWithBaseStates>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      all_joints_with_base_callback(topic_name, joint_vector);
      break;

    case 22:
      if (vector22d_publisher_.find(topic_name) != vector22d_publisher_.end()) {
        vector_callback(topic_name, joint_vector);
        return;
      }
      vector22d_publisher_[topic_name] =
          (this->create_publisher<l5a_msgs::msg::Vector22d>(
              topic_name, GetBestEffortQoS(msg_queue_size_)));
      vector_callback(topic_name, joint_vector);
      break;

    default:
      LOG(WARNING) << "Unknown data! joint_vector size is "
                   << joint_vector.size() << " Topic name is: " << topic_name;
      break;
  }
}

void Publisher::Publish(const std::string& topic_name, double double_data) {
  if (double_publisher_.find(topic_name) != double_publisher_.end()) {
    double_callback(topic_name, double_data);
    return;
  }
  double_publisher_[topic_name] =
      (this->create_publisher<l5a_msgs::msg::Double>(
          topic_name, GetBestEffortQoS(msg_queue_size_)));
  double_callback(topic_name, double_data);
}

void Publisher::all_joints_callback(const std::string& topic_name,
                                    Eigen::VectorXd joint_vector) {
  l5a_msgs::msg::AllJointStates all_jointState;

  {
    all_jointState.left_hip_pitch_joint = joint_vector[0];
    all_jointState.left_hip_roll_joint = joint_vector[1];
    all_jointState.left_knee_joint = joint_vector[2];
    all_jointState.left_wheel_joint = joint_vector[3];
    all_jointState.right_hip_pitch_joint = joint_vector[4];
    all_jointState.right_hip_roll_joint = joint_vector[5];
    all_jointState.right_knee_joint = joint_vector[6];
    all_jointState.right_wheel_joint = joint_vector[7];
  }
  all_joint_publisher_[topic_name]->publish(all_jointState);
}

void Publisher::all_joints_with_base_callback(const std::string& topic_name,
                                              Eigen::VectorXd joint_vector) {
  l5a_msgs::msg::AllJointWithBaseStates all_joint_with_baseState;
  {
    all_joint_with_baseState.x = joint_vector[0];
    all_joint_with_baseState.y = joint_vector[1];
    all_joint_with_baseState.z = joint_vector[2];
    all_joint_with_baseState.roll = joint_vector[3];
    all_joint_with_baseState.pitch = joint_vector[4];
    all_joint_with_baseState.yaw = joint_vector[5];

    all_joint_with_baseState.left_hip_pitch_joint = joint_vector[6];
    all_joint_with_baseState.left_hip_roll_joint = joint_vector[7];
    all_joint_with_baseState.left_knee_joint = joint_vector[8];
    all_joint_with_baseState.left_wheel_joint = joint_vector[9];
    all_joint_with_baseState.right_hip_pitch_joint = joint_vector[10];
    all_joint_with_baseState.right_hip_roll_joint = joint_vector[11];
    all_joint_with_baseState.right_knee_joint = joint_vector[12];
    all_joint_with_baseState.right_wheel_joint = joint_vector[13];
  }
  all_joint_with_base_publisher_[topic_name]->publish(all_joint_with_baseState);
}

void Publisher::state_estimator_callback(const std::string& topic_name,
                                         Eigen::VectorXd joint_vector) {
  l5a_msgs::msg::StateEstimator state_estimator;

  {
    state_estimator.pos_x = joint_vector[0];
    state_estimator.pos_y = joint_vector[1];
    state_estimator.pos_z = joint_vector[2];
    state_estimator.roll = joint_vector[3];
    state_estimator.pitch = joint_vector[4];
    state_estimator.yaw = joint_vector[5];
    state_estimator.vel_x = joint_vector[6];
    state_estimator.vel_y = joint_vector[7];
    state_estimator.vel_z = joint_vector[8];
    state_estimator.omega_x = joint_vector[9];
    state_estimator.omega_y = joint_vector[10];
    state_estimator.omega_z = joint_vector[11];
  }
  state_estimator_publisher_[topic_name]->publish(state_estimator);
}

void Publisher::vector_callback(const std::string& topic_name,
                                Eigen::VectorXd joint_vector) {
  switch (joint_vector.size()) {
    case 1: {
      l5a_msgs::msg::Vector1d vector1d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector1d_msg.data[i] = joint_vector[i];
      }
      vector1d_publisher_[topic_name]->publish(vector1d_msg);
      break;
    }
    case 2: {
      l5a_msgs::msg::Vector2d vector2d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector2d_msg.data[i] = joint_vector[i];
      }
      vector2d_publisher_[topic_name]->publish(vector2d_msg);
      break;
    }
    case 3: {
      l5a_msgs::msg::Vector3d vector3d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector3d_msg.data[i] = joint_vector[i];
      }
      vector3d_publisher_[topic_name]->publish(vector3d_msg);
      break;
    }

    case 4: {
      l5a_msgs::msg::Vector4d vector4d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector4d_msg.data[i] = joint_vector[i];
      }
      vector4d_publisher_[topic_name]->publish(vector4d_msg);
      break;
    }
    case 6: {
      l5a_msgs::msg::Vector6d vector6d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector6d_msg.data[i] = joint_vector[i];
      }
      vector6d_publisher_[topic_name]->publish(vector6d_msg);
      break;
    }
    case 7: {
      l5a_msgs::msg::Vector7d vector7d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector7d_msg.data[i] = joint_vector[i];
      }
      vector7d_publisher_[topic_name]->publish(vector7d_msg);
      break;
    }
    case 8: {
      l5a_msgs::msg::Vector8d vector8d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector8d_msg.data[i] = joint_vector[i];
      }
      vector8d_publisher_[topic_name]->publish(vector8d_msg);
      break;
    }
    case 9: {
      l5a_msgs::msg::Vector9d vector9d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector9d_msg.data[i] = joint_vector[i];
      }
      vector9d_publisher_[topic_name]->publish(vector9d_msg);
      break;
    }
    case 14: {
      l5a_msgs::msg::Vector14d vector14d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector14d_msg.data[i] = joint_vector[i];
      }
      vector14d_publisher_[topic_name]->publish(vector14d_msg);
      break;
    }
    case 22: {
      l5a_msgs::msg::Vector22d vector22d_msg;
      for (int i = 0; i < joint_vector.size(); i++) {
        vector22d_msg.data[i] = joint_vector[i];
      }
      vector22d_publisher_[topic_name]->publish(vector22d_msg);
      break;
    }

    default:
      LOG(WARNING) << "Unknown joint_vector! joint_vector size is "
                   << joint_vector.size();
      break;
  }
}

void Publisher::double_callback(const std::string& topic_name,
                                double double_data) {
  l5a_msgs::msg::Double double_msg;
  double_msg.data = double_data;
  double_publisher_[topic_name]->publish(double_msg);
}

Publisher::~Publisher() {}
}  // namespace ROS
