#ifndef ros_publisher_H
#define ros_publisher_H

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

// reference: https://github.com/MindSpaceInc/Spot-MuJoCo-ROS2

// #include <rmw/types.h>
#include <rclcpp/rclcpp.hpp>
#include <unordered_map>

#include <glog/logging.h>
#include <Eigen/Dense>

#include <l5a_msgs/msg/all_joint_states.hpp>
#include <l5a_msgs/msg/all_joint_with_base_states.hpp>
#include <l5a_msgs/msg/double.hpp>
#include <l5a_msgs/msg/state_estimator.hpp>
#include <l5a_msgs/msg/vector14d.hpp>
#include <l5a_msgs/msg/vector1d.hpp>
#include <l5a_msgs/msg/vector22d.hpp>
#include <l5a_msgs/msg/vector2d.hpp>
#include <l5a_msgs/msg/vector3d.hpp>
#include <l5a_msgs/msg/vector4d.hpp>
#include <l5a_msgs/msg/vector6d.hpp>
#include <l5a_msgs/msg/vector7d.hpp>
#include <l5a_msgs/msg/vector8d.hpp>
#include <l5a_msgs/msg/vector9d.hpp>

using namespace rclcpp;
using namespace std::chrono_literals;

namespace ROS {

class Publisher : public rclcpp::Node {
 public:
  Publisher(const Publisher&) = delete;
  Publisher& operator=(const Publisher&) = delete;

  static std::shared_ptr<Publisher> getInstance() {
    if (shared_instance_ == nullptr) {
      std::lock_guard<std::mutex> lock(mtx_);  // 加锁保护，确保线程安全
      if (shared_instance_ == nullptr) {
        shared_instance_ = std::shared_ptr<Publisher>(new Publisher());
      }
    }
    return shared_instance_;  // 返回唯一的实例
  }
  void Publish(const std::string& topic_name, Eigen::VectorXd joint_vector);
  void Publish(const std::string& topic_name, double double_data);
  ~Publisher();

 private:
  static std::shared_ptr<Publisher> shared_instance_;
  static std::mutex mtx_;
  int msg_queue_size_;

  rclcpp::QoS GetBestEffortQoS(int queue_size);
  Publisher();
  void all_joints_callback(const std::string& topic_name,
                           Eigen::VectorXd joint_vector);
  void all_joints_with_base_callback(const std::string& topic_name,
                                     Eigen::VectorXd joint_vector);
  void state_estimator_callback(const std::string& topic_name,
                                Eigen::VectorXd joint_vector);
  void vector_callback(const std::string& topic_name,
                       Eigen::VectorXd joint_vector);
  void double_callback(const std::string& topic_name, double double_data);

  std::unordered_map<
      std::string, rclcpp::Publisher<l5a_msgs::msg::AllJointStates>::SharedPtr>
      all_joint_publisher_;
  std::unordered_map<
      std::string,
      rclcpp::Publisher<l5a_msgs::msg::AllJointWithBaseStates>::SharedPtr>
      all_joint_with_base_publisher_;
  std::unordered_map<
      std::string, rclcpp::Publisher<l5a_msgs::msg::StateEstimator>::SharedPtr>
      state_estimator_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Double>::SharedPtr>
      double_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector1d>::SharedPtr>
      vector1d_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector2d>::SharedPtr>
      vector2d_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector3d>::SharedPtr>
      vector3d_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector4d>::SharedPtr>
      vector4d_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector6d>::SharedPtr>
      vector6d_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector7d>::SharedPtr>
      vector7d_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector8d>::SharedPtr>
      vector8d_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector9d>::SharedPtr>
      vector9d_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector14d>::SharedPtr>
      vector14d_publisher_;
  std::unordered_map<std::string,
                     rclcpp::Publisher<l5a_msgs::msg::Vector22d>::SharedPtr>
      vector22d_publisher_;
};
inline std::shared_ptr<Publisher> Publisher::shared_instance_ = nullptr;
inline std::mutex Publisher::mtx_;

}  // namespace ROS

#endif  // ros_publisher_H