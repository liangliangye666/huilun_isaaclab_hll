#include <cmath>
#include <iostream>

#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/parsers/urdf.hpp"

int main(int argc, char** argv) {
  std::string workspacePath = std::getenv("PROJECT_ROOT_DIR");
  std::string urdf_filename = workspacePath + "/sim/model/Y4A/urdf/l4aurdf20241119.urdf";
  pinocchio::Model robot_model;
  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::urdf::buildModel(urdf_filename, root_joint, robot_model);
  pinocchio::Data robot_data = pinocchio::Data(robot_model);

  std::cout << "robot_name: " << robot_model.name << std::endl;
  std::cout << "robot_nq: " << robot_model.nq << std::endl;
  std::cout << "robot_nv: " << robot_model.nv << std::endl;
  std::cout << "robot_njoints: " << robot_model.njoints << std::endl;
  std::cout << "robot_nbodies: " << robot_model.nbodies << std::endl;

  for (pinocchio::JointIndex joint_id = 1; joint_id < robot_model.njoints; ++joint_id) {
    const std::string& joint_name = robot_model.names[joint_id];
    const pinocchio::JointModel& joint = robot_model.joints[joint_id];
    std::cout << "Joint ID: " << joint_id << ", Name: " << joint_name << ", Type: " << joint.shortname() << std::endl;
  }
  //q order: x y z quaternion head_yaw/pitch left:hip knee front_wheel switch rear_wheel arm1-7; right:hip knee front_wheel switch rear_wheel arm1-7
  Eigen::VectorXd q = pinocchio::neutral(robot_model);
  // std::cout << "the size of q is: " << q.size() << std::endl;
  pinocchio::JointIndex front_wheel_joint_l, front_wheel_joint_r, handL_jointID, handR_jointID;
  front_wheel_joint_l = robot_model.getJointId("left_wheel_joint");
  std::cout << "left_wheel_joint id is: " << front_wheel_joint_l << std::endl;
  Eigen::Vector3d whlposL, com, whl2com, posHandL, posHandR;

  double angKnee = -M_PI/2;
  for(int i=0; ; i++)
  {
    angKnee += (0.01 / 180.0) * M_PI;
    // std::cout << "knee =: " << angKnee << std::endl;
    // q << 0,0,0,0,0,0,0,(20.0/180.0)*M_PI,angKnee,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    q << 0,0,0,0,0,0,0,(-20.0/180.0)*M_PI,0.0,0.0,angKnee,0.0,0.0,(-20.0/180.0)*M_PI,0.0,0.0,angKnee,0.0,0.0;
    pinocchio::forwardKinematics(robot_model,robot_data,q);
    whlposL = robot_data.oMi[front_wheel_joint_l].translation();
    std::cout << "the wheel position is :" << whlposL << std::endl;
    pinocchio::centerOfMass(robot_model, robot_data, q);
    com = robot_data.com[0];
    std::cout << "the center of mass is :" << com << std::endl;
    whl2com = com - whlposL;
    // std::cout << "the vector from wheel to com is:" << whl2com << std::endl;
    double comAng = std::atan2(whl2com[0],whl2com[2]);
    std::cout << "the comAng is :" << comAng << std::endl;
    if(std::abs(comAng) <= (0.01/180.0)*M_PI )
    {
      std::cout << "angKnee in rad is :" << angKnee << std::endl;
      std::cout << "initPos is: " << whlposL << std::endl;
      break;
    }
    if(angKnee >= M_PI/2)
    {
      std::cout << "can not find the balance point" << std::endl;
      break;
    }

  }

  return 0;
}