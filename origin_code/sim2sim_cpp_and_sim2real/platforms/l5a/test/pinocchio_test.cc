#include <iostream>

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/fwd.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif
#include <glog/logging.h>
#include <chrono>

int main() {
  std::string workspacePath = std::getenv("PROJECT_ROOT_DIR");
  std::string urdf_filename =
      workspacePath + "/sim/model/l4a/urdf/l4aurdf20241119.urdf";
  pinocchio::Model model, model_fixed;
  pinocchio::JointModelFreeFlyer base_joint;
  pinocchio::urdf::buildModel(urdf_filename, base_joint, model);
  pinocchio::Data data = pinocchio::Data(model);
  pinocchio::urdf::buildModel(urdf_filename, model_fixed);
  pinocchio::Data data_fixed = pinocchio::Data(model_fixed);

  std::cout << "model name: " << model.name << std::endl;
  std::cout << "pino_model.nq: " << model.nq << std::endl;
  std::cout << "pino_model.nv: " << model.nv << std::endl;
  std::cout << "pino_model.njoints: " << model.njoints << std::endl;
  std::cout << "pino_model.nbodies: " << model.nbodies << std::endl;

  // Calculate the Mass Matrix
  Eigen::VectorXd q = pinocchio::neutral(model);
  Eigen::VectorXd q_fixed = pinocchio::neutral(model_fixed);

  auto start = std::chrono::high_resolution_clock::now();
  pinocchio::crba(model, data, q);
  pinocchio::crba(model_fixed, data_fixed, q_fixed);
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::forwardKinematics(model_fixed, data_fixed, q_fixed);
  pinocchio::centerOfMass(model, data, q);

  Eigen::Vector3d x1 = data.oMi[1].translation() - data.oMi[2].translation();
  Eigen::Vector3d x2 = data.oMi[1].translation() - data.oMi[7].translation();
  LOG(INFO) << "x1: " << x1.transpose();
  LOG(INFO) << "x2: " << x2.transpose();

  std::cout << "Total Mass: " << data.mass[0] << std::endl;
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "运行时间：" << duration.count() << "毫秒" << std::endl;

  // Print out the placement of each joint of the kinematic tree
  std::cout << "\nInitial q for floating base model: " << std::endl;
  for (int i = 0; i < model.nq; i++) {
    std::cout << i << "\t :\t" << q[i] << std::endl;
  }
  std::cout << "Initial q for fixed base model: " << std::endl;
  std::cout << std::endl;
  for (int index = 0; index < model_fixed.nq; index++) {
    std::cout << index << " :\t" << q_fixed[index] << std::endl;
  }

  for (pinocchio::JointIndex joint_id = 0;
       joint_id < (pinocchio::JointIndex)model.njoints; ++joint_id) {
    std::cout << std::setw(24) << std::left << joint_id << " "
              << model.names[joint_id] << " " << std::fixed
              << std::setprecision(3)
              << data.oMi[joint_id].translation().transpose() << std::endl;
  }

  for (pinocchio::JointIndex joint_id = 0;
       joint_id < (pinocchio::JointIndex)model.njoints; ++joint_id) {
    std::cout << model.names[joint_id] << "," << std::endl;
  }

  // fixed base model
  std::cout << std::endl;
  for (pinocchio::JointIndex joint_id = 0;
       joint_id < (pinocchio::JointIndex)model_fixed.njoints; ++joint_id) {
    std::cout << std::setw(24) << std::left << joint_id << " "
              << model_fixed.names[joint_id] << " " << std::fixed
              << std::setprecision(3)
              << data_fixed.oMi[joint_id].translation().transpose()
              << std::endl;
  }
  for (pinocchio::JointIndex joint_id = 0;
       joint_id < (pinocchio::JointIndex)model_fixed.njoints; ++joint_id) {
    std::cout << model_fixed.names[joint_id] << "," << std::endl;
  }

  return 0;
}