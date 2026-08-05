#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

// #include "ros2_msgs/msg/num.hpp"
// #include "ros2_msgs/msg/vector4d.hpp"

class MinimalSubscriber : public rclcpp::Node {
  // using Vector4d = ros2_msgs::msg::Vector4d;

 public:
  MinimalSubscriber() : Node("minimal_subscriber") {
    // auto topic_callback = [this](ros2_msgs::msg::Vector4d::UniquePtr msg) -> void {
    //   for (size_t i = 0; i < 4; i++) {
    //     /* code */
    //     RCLCPP_INFO(this->get_logger(), "I heard: '%lf'", msg->data[i]);
    //   }
    // };
    // subscription_ = this->create_subscription<ros2_msgs::msg::Vector4d>("topic", 10, topic_callback);
    auto topic_callback = [this](std_msgs::msg::String::UniquePtr msg) -> void { RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str()); };
    subscription_ = this->create_subscription<std_msgs::msg::String>("topic", 10, topic_callback);
  }

 private:
  // rclcpp::Subscription<ros2_msgs::msg::Vector4d>::SharedPtr subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}