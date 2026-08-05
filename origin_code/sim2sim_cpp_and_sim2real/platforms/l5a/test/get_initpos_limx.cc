#include <cmath>
#include <iostream>

#include "gac_math.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/parsers/urdf.hpp"

int main(int argc, char** argv) {
  std::string workspacePath = std::getenv("PROJECT_ROOT_DIR");
  std::string urdf_filename = workspacePath + "/sim/model/l5a/urdf/limx.urdf";
  pinocchio::Model robot_model;
  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::urdf::buildModel(urdf_filename, root_joint, robot_model);
  pinocchio::Data robot_data = pinocchio::Data(robot_model);
  // 打印一下各关节id
  for (pinocchio::JointIndex joint_id = 1; joint_id < robot_model.njoints; ++joint_id) {
    const std::string& joint_name = robot_model.names[joint_id];
    const pinocchio::JointModel& joint = robot_model.joints[joint_id];
    std::cout << "Joint ID: " << joint_id << ", Name: " << joint_name << ", Type: " << joint.shortname() << std::endl;
  }

  for (pinocchio::FrameIndex frame_id = 0; frame_id < robot_model.frames.size(); ++frame_id) {
    const pinocchio::Frame& frame = robot_model.frames[frame_id];

    if (frame.type == pinocchio::FrameType::BODY) {
      std::cout << "Link ID: " << frame_id << ", Link name: " << frame.name << ", Parent joint: " << robot_model.names[frame.parentJoint] << std::endl;
    }
  }

  std::cout << "robot_name: " << robot_model.name << std::endl;
  std::cout << "robot_nq: " << robot_model.nq << std::endl;
  std::cout << "robot_nv: " << robot_model.nv << std::endl;
  std::cout << "robot_njoints: " << robot_model.njoints << std::endl;
  std::cout << "robot_nbodies: " << robot_model.nbodies << std::endl;

  // q order: x y z quaternion left:hip knee wheel right:hip knee wheel
  Eigen::VectorXd q = pinocchio::neutral(robot_model);
  std::cout << "the size of q is: " << q.size() << std::endl;
  pinocchio::JointIndex front_wheel_joint_l, front_wheel_joint_r;
  front_wheel_joint_l = robot_model.getJointId("wheel_L_Joint");
  front_wheel_joint_r = robot_model.getJointId("wheel_R_Joint");
  std::cout << "left_wheel_joint id is: " << front_wheel_joint_l << std::endl;

  Eigen::Vector3d whlposL, whlposR, com, whl2com;
  double angKnee = -M_PI / 2;
  for (int i = 0;; i++) {
    // angKnee += (0.001 / 180.0) * M_PI;
    angKnee = 0;
    // std::cout << "knee =: " << angKnee << std::endl;
    // double angRoll = (3.0 / 180.0) * M_PI;
  double angRoll = (0.0 / 180.0) * M_PI;
    double angHip = (0.0 / 180.0) * M_PI;
    q << 0, 0, 0, 0, 0, 0, 0, angRoll, angHip, angKnee, 0, angRoll, angHip, angKnee, 0;

    pinocchio::forwardKinematics(robot_model, robot_data, q);
    whlposL = robot_data.oMi[front_wheel_joint_l].translation();
    whlposR = robot_data.oMi[front_wheel_joint_r].translation();
    std::cout << "the L wheel position is :" << whlposL << std::endl;
    std::cout << "the R wheel position is :" << whlposR << std::endl;
    pinocchio::centerOfMass(robot_model, robot_data, q);
    com = robot_data.com[0];
    // std::cout << "the center of mass is :" << com << std::endl;
    whl2com = com - whlposL;
    // std::cout << "the vector from wheel to com is:" << whl2com << std::endl;
    double comAng = std::atan2(whl2com[0], whl2com[2]);
    // std::cout << "the comAng is :" << comAng << std::endl;
    // if (std::abs(comAng) <= (0.1 / 180.0) * M_PI) {
    //   std::cout << "angRoll is :" << angRoll << std::endl;
    //   std::cout << "angHip is :" << angHip << std::endl;
    //   std::cout << "angKnee is :" << angKnee << std::endl;
    //   std::cout << "angRoll degree is:" << angRoll / M_PI * 180.0 << std::endl;
    //   std::cout << "angHip degree is:" << angHip / M_PI * 180.0 << std::endl;
    //   std::cout << "angKnee degree is:" << angKnee / M_PI * 180.0 << std::endl;
    //   std::cout << "initPos is: " << whlposL.transpose() << std::endl;
    //   std::cout << "right wheel initPos is: " << whlposR.transpose() << std::endl;
    //   std::cout << "the center of mass is :" << com.transpose() << std::endl;
    //   break;
    // }
    // if (angKnee >= 2.1991) {
    //   std::cout << "can not find the balance point" << std::endl;
    //   break;
    // }
  }

  return 0;
}