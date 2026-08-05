#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>

#include <Eigen/Dense>

int main() {
  std::string workspacePath = std::getenv("BUILD_WORKSPACE_DIRECTORY");
  YAML::Node config = YAML::LoadFile(workspacePath + "/platforms/Y3A/test/config.yaml");
  YAML::Node config_wbc = YAML::LoadFile(workspacePath + "/platforms/Y3A/control/wbc_parameters.yaml");

  if (config["Y3A"]) {
    // std::cout << "kp: " << config["Y3A"]["kp"].as<int>() << "\n";
    // Eigen::VectorXd x;
    // x = Eigen::VectorXd::Map(config_wbc["Y3A"]["kd"].as<std::vector<double>>().data(),
    //                          config_wbc["Y3A"]["kd"].as<std::vector<double>>().size());
    // std::cout << "kd: \n" << x << std::endl;
  }
  Eigen::VectorXd com_task_x_desired_;
  Eigen::VectorXd com_task_xdot_desired_;
  Eigen::VectorXd com_task_kp_;
  Eigen::VectorXd com_task_kd_;

  com_task_x_desired_ = Eigen::VectorXd::Map(config_wbc["com_task_x_desired"].as<std::vector<double>>().data(),
                                             config_wbc["com_task_x_desired"].as<std::vector<double>>().size());
  com_task_xdot_desired_ = Eigen::VectorXd::Map(config_wbc["com_task_xdot_desired"].as<std::vector<double>>().data(),
                                                config_wbc["com_task_xdot_desired"].as<std::vector<double>>().size());
  com_task_kp_ = Eigen::VectorXd::Map(config_wbc["com_task_kp"].as<std::vector<double>>().data(), config_wbc["com_task_kp"].as<std::vector<double>>().size());
  com_task_kd_ = Eigen::VectorXd::Map(config_wbc["com_task_kd"].as<std::vector<double>>().data(), config_wbc["com_task_kd"].as<std::vector<double>>().size());
  std::cout << "com_task_x_desired_: \n" << com_task_x_desired_ << std::endl;

  return 0;
}