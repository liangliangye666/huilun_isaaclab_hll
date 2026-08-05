#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <sys/wait.h>
#include <unistd.h>  // 包含 fork, execlp
#include <csignal>
#include <cstdio>
#include <iomanip>

using namespace std::chrono_literals;
pid_t child_pid = -1;  // 全局变量记录子进程PID

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class MinimalPublisher : public rclcpp::Node {
 public:
  MinimalPublisher() : Node("minimal_publisher"), count_(0) {
    start_time_ = std::chrono::system_clock::now();
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

 private:
  void timer_callback() {
    auto time_current = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(time_current - start_time_);
    double time_s = duration.count() / 1000000000;
    std::cout << "time_s: " << time_s << std::endl;
    std::cout << "hello" << std::endl;
    if (time_s > 3) {
      if (child_pid != -1) {
        if (kill(child_pid, SIGTERM) == 0) {
          std::cout << "正在终止ros2bag进程(PID: " << child_pid << ")..." << std::endl;
          int status;
          waitpid(child_pid, &status, 0);
          std::cout << "ros2bag进程已终止。" << std::endl;
        } else {
          perror("终止ros2bag进程失败");
        }
        child_pid = -1;
      }
    }

    auto message = std_msgs::msg::String();
    message.data = "Hello, world! " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;

  std::chrono::_V2::system_clock::time_point start_time_;
};

// 清理函数，终止子进程
void cleanup() {
  if (child_pid != -1) {
    if (kill(child_pid, SIGTERM) == 0) {
      std::cout << "正在终止ros2bag进程(PID: " << child_pid << ")..." << std::endl;
      int status;
      waitpid(child_pid, &status, 0);
      std::cout << "ros2bag进程已终止。" << std::endl;
    } else {
      perror("终止ros2bag进程失败");
    }
    child_pid = -1;
  }
}

void signal_handler(int sig) {
  std::cout << "\n接收到中断信号，正在退出..." << std::endl;
  exit(sig);  // 退出时会调用atexit注册的函数
}

std::string get_current_datetime() {
  // 获取当前系统时间
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm* tm_now = std::localtime(&now_time);
  std::ostringstream oss;
  oss << "-" << std::put_time(tm_now, "%Y-%m-%d %H:%M:%S");

  return oss.str();
}

int main(int argc, char* argv[]) {
  // 注册清理函数，程序正常退出时调用
  // atexit(cleanup);
  // signal(SIGINT, signal_handler);
  std::string project_root_path = std::getenv("PROJECT_ROOT_DIR");
  std::string date_time_str = get_current_datetime();
  std::string data_name = project_root_path + "/data/ros2_bag" + date_time_str;

  // 创建子进程
  child_pid = fork();
  if (child_pid == 0) {
    execlp("ros2", "ros2", "bag", "record", "-a", "-o", data_name.c_str(), nullptr);
    perror("执行ros2bag失败");
    exit(EXIT_FAILURE);
  } else if (child_pid == -1) {
    perror("创建子进程失败");
    return EXIT_FAILURE;
  }

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}