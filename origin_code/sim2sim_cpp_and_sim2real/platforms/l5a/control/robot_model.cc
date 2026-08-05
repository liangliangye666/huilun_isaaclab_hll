#include "robot_model.h"

namespace l5a {

const std::vector<PinoJoints> wheel_joints = {
    PinoJoints::left_wheel_joint,
    PinoJoints::right_wheel_joint,
};

const std::vector<AllJoints> all_joints = {
    AllJoints::left_hip_roll_joint,  AllJoints::left_hip_pitch_joint,  AllJoints::left_knee_joint,  AllJoints::left_wheel_joint,
    AllJoints::right_hip_roll_joint, AllJoints::right_hip_pitch_joint, AllJoints::right_knee_joint, AllJoints::right_wheel_joint,
};

RobotModel::RobotModel(std::string urdf_path) {
  // google::InitGoogleLogging("controller");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  control_dt = 0.002;  // default control dt
  std::string workspacePath = std::getenv("PROJECT_ROOT_DIR");
  std::string path = workspacePath + urdf_path;
  const char* urdf_filename = path.c_str();
  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::urdf::buildModel(urdf_filename, root_joint, pino_model_);
  pino_data_ = pinocchio::Data(pino_model_);
  pinocchio::urdf::buildModel(urdf_filename, pino_model_fixed_);
  pino_data_fixed_ = pinocchio::Data(pino_model_fixed_);
  joint_vel_est_ = std::make_unique<JointVelEstimator>(pino_model_.nv - 6, control_dt);

  observed_value = Eigen::VectorXd::Zero(81);
  Initialize();

#if SIM_ENABLE
  ros_publisher_ = ROS::Publisher::getInstance();
  joystick_ = std::make_shared<JoyStick>();
#endif
}

RobotModel::~RobotModel() {}

void RobotModel::Initialize() {
  // Robot states
  q_rpy = Eigen::VectorXd::Zero(pino_model_.nv);
  q_pino = pinocchio::neutral(pino_model_);
  qdot = Eigen::VectorXd::Zero(pino_model_.nv);
  joint_vel_ = Eigen::VectorXd::Zero(pino_model_.nv - 6);
  qddot = Eigen::VectorXd::Zero(pino_model_.nv);
  q_desired = Eigen::VectorXd::Zero(pino_model_.nv);
  qdot_desired = Eigen::VectorXd::Zero(pino_model_.nv);
  q_fixed = pinocchio::neutral(pino_model_fixed_);
  qdot_fixed = Eigen::VectorXd::Zero(pino_model_fixed_.nv);

  M = Eigen::MatrixXd::Zero(pino_model_.nv, pino_model_.nv);
  C = Eigen::VectorXd::Zero(pino_model_.nv);
  G = Eigen::VectorXd::Zero(pino_model_.nv);
  S = Eigen::MatrixXd::Zero(pino_model_.nv - 6, pino_model_.nv);
  S_transpose_pinv = Eigen::MatrixXd::Zero(pino_model_.nv - 6, pino_model_.nv);

  S.block(0, 6, pino_model_.nv - 6, pino_model_.nv - 6) = Eigen::MatrixXd::Identity(pino_model_.nv - 6, pino_model_.nv - 6);
  S_transpose_pinv = (S * S.transpose()).inverse() * S;
  vel_x_des_ = 0;
  vel_y_des_ = 0;
  omega_des_ = 0;
  phase_ = 0;
  gait_enable_ = false;
  emergency_ = false;

  R_HW = Eigen::Matrix3d::Identity();
  R_WH = Eigen::Matrix3d::Identity();
  R_WB = Eigen::Matrix3d::Identity();
  R_BW = Eigen::Matrix3d::Identity();
  R_HB = Eigen::Matrix3d::Identity();

  pos_left_wheel = Eigen::Vector3d::Zero(3);
  pos_right_wheel = Eigen::Vector3d::Zero(3);

#if SIM_ENABLE
  imu_ = std::make_unique<IMU>();
#else
  imu_ = std::make_unique<IMU>(Eigen::Vector3d(0.0, 0.0, M_PI));
#endif
}

#if SIM_ENABLE
void RobotModel::UpdateMujocoJointStates(const mjModel* m, mjData* d) {
  mj_model_ = m;
  mj_data_ = d;
  control_dt = mj_model_->opt.timestep;

  ori_base_world_.quat = Eigen::Quaterniond(mj_data_->qpos[3], mj_data_->qpos[4], mj_data_->qpos[5], mj_data_->qpos[6]);
  ori_base_world_.euler = GacMath::QuaternionToEuler(ori_base_world_.quat);  // 世界坐标系下base姿态,顺序yaw,pitch,roll
  Eigen::Quaterniond quat_base(mj_data_->qpos[3], mj_data_->qpos[4], mj_data_->qpos[5], mj_data_->qpos[6]);
  Eigen::Vector3d rpy_base = GacMath::QuaternionToRPY(quat_base);  // 世界坐标系下base姿态,顺序roll,pitch,yaw

  // Update q_rpy
  for (size_t i = 0; i < 3; i++) {
    q_rpy[i] = mj_data_->qpos[i];
    // q_rpy[i] = 0;
  }
  q_rpy[3] = rpy_base[0];
  q_rpy[4] = rpy_base[1];
  // q_rpy[5] = rpy_base[2];
  q_rpy[5] = 0;
  for (size_t i = 6; i < pino_model_.nv; i++) {
    q_rpy[i] = mj_data_->qpos[i + 1];
  }
  Eigen::Quaterniond quat_rpy = GacMath::RPYToQuaternion(q_rpy[3], q_rpy[4], q_rpy[5]);

  // q_pino = [global_base_position, global_base_quaternion, joint_positions]
  // Update q_pino
  for (size_t i = 0; i < 3; i++) {
    // q_pino[i] = q_rpy[i];
    q_pino[i] = 0;
  }
  q_pino[3] = quat_rpy.x();  // x
  q_pino[4] = quat_rpy.y();  // y
  q_pino[5] = quat_rpy.z();  // z
  q_pino[6] = quat_rpy.w();  // w
  for (size_t i = 7; i < mj_model_->nq; i++) {
    q_pino[i] = mj_data_->qpos[i];
  }
  // Update R_HW
  R_WB = GacMath::EulerToRotationMatrix(q_rpy[5], q_rpy[4], q_rpy[3]);
  R_BW = R_WB.transpose();
  Eigen::Vector3d x_head = Eigen::Vector3d::Zero();
  Eigen::Vector3d y_head = Eigen::Vector3d::Zero();
  Eigen::Vector3d z_head{0, 0, 1};
  x_head[0] = R_WB(0, 0);
  x_head[1] = R_WB(1, 0);
  x_head.normalize();
  y_head = z_head.cross(x_head);
  R_WH.col(0) = x_head;
  R_WH.col(1) = y_head;
  R_WH.col(2) = z_head;
  R_HW = R_WH.transpose();
  R_HB = R_HW * R_WB;

  joint_vel_ = joint_vel_est_->update(q_rpy.tail(pino_model_.nv - 6));
  // Update qdot
  for (size_t i = 0; i < pino_model_.nv; i++) {  // 世界坐标系下的速度
    qdot[i] = mj_data_->qvel[i];
  }
  qdot.segment(0, 3) = R_BW * qdot.segment(0, 3);  // local坐标系下的速度
  qdot.segment(3, 3) = R_BW * qdot.segment(3, 3);  // local坐标系下的角速度
  // Update qddot
  for (size_t i = 0; i < pino_model_.nv; i++) {  // 世界坐标系下的加速度
    qddot[i] = mj_data_->qacc[i];
  }

  // Update q_fixed qdot_fixed
  q_fixed = q_pino.segment(7, pino_model_fixed_.nq);
  qdot_fixed = qdot.segment(6, pino_model_fixed_.nv);

  // joystick
  joystick_->Update();
}
#endif
#if PHYSICS_ENABLE  // 确保位置和速度的单位是rad,rad/s
void RobotModel::UpdateRealJointStates(standmode_output_t* standmode_output, standmode_input_t* standmode_input) {
  standmode_output_ = standmode_output;
  standmode_input_ = standmode_input;
  control_dt = 0.002;

  ori_base_world_.euler << standmode_input_->IMU_signals.IMU_yaw, standmode_input_->IMU_signals.IMU_pitch, standmode_input_->IMU_signals.IMU_roll;
  ori_base_local_.omega << standmode_input_->IMU_signals.IMU_wx_body, standmode_input_->IMU_signals.IMU_wy_body, standmode_input_->IMU_signals.IMU_wz_body;
  ori_base_local_.acc << standmode_input_->IMU_signals.IMU_accx, standmode_input_->IMU_signals.IMU_accy, standmode_input_->IMU_signals.IMU_accz;
  imu_->Update(ori_base_local_, ori_base_world_);
  // Update q_rpy
  q_rpy.segment(0, 3) = Eigen::Vector3d::Zero();
  q_rpy[3] = ori_base_local_.euler[2];
  q_rpy[4] = ori_base_local_.euler[1];
  // q_rpy[5] = ori_base_local_.euler[0];
  q_rpy[5] = 0;

  q_rpy[6] = standmode_input->joints_status.joint_h_roll_l.pos_fb;
  q_rpy[7] = standmode_input->joints_status.joint_h_pitch_l.pos_fb;
  q_rpy[8] = standmode_input->joints_status.joint_k_pitch_l.pos_fb;
  q_rpy[9] = standmode_input->joints_status.joint_w_pitch_l.pos_fb;

  q_rpy[10] = standmode_input->joints_status.joint_h_roll_r.pos_fb;
  q_rpy[11] = standmode_input->joints_status.joint_h_pitch_r.pos_fb;
  q_rpy[12] = standmode_input->joints_status.joint_k_pitch_r.pos_fb;
  q_rpy[13] = standmode_input->joints_status.joint_w_pitch_r.pos_fb;

  // Update q_pino  --- q = [global_base_position, global_base_quaternion,
  // joint_positions]
  Eigen::Quaterniond quat_rpy = GacMath::RPYToQuaternion(q_rpy[3], q_rpy[4], q_rpy[5]);
  for (size_t i = 0; i < 3; i++) {
    // q_pino[i] = q_rpy[i];
    q_pino[i] = 0;
  }
  q_pino[3] = quat_rpy.x();  // x
  q_pino[4] = quat_rpy.y();  // y
  q_pino[5] = quat_rpy.z();  // z
  q_pino[6] = quat_rpy.w();  // w

  q_pino[7] = standmode_input->joints_status.joint_h_roll_l.pos_fb;
  q_pino[8] = standmode_input->joints_status.joint_h_pitch_l.pos_fb;
  q_pino[9] = standmode_input->joints_status.joint_k_pitch_l.pos_fb;
  q_pino[10] = standmode_input->joints_status.joint_w_pitch_l.pos_fb;
  q_pino[11] = standmode_input->joints_status.joint_h_roll_r.pos_fb;
  q_pino[12] = standmode_input->joints_status.joint_h_pitch_r.pos_fb;
  q_pino[13] = standmode_input->joints_status.joint_k_pitch_r.pos_fb;
  q_pino[14] = standmode_input->joints_status.joint_w_pitch_r.pos_fb;

  // Update R_HW
  R_WB = GacMath::EulerToRotationMatrix(q_rpy[5], q_rpy[4], q_rpy[3]);
  R_BW = R_WB.transpose();
  Eigen::Vector3d x_head = Eigen::Vector3d::Zero();
  Eigen::Vector3d y_head = Eigen::Vector3d::Zero();
  Eigen::Vector3d z_head{0, 0, 1};
  x_head[0] = R_WB(0, 0);
  x_head[1] = R_WB(1, 0);
  x_head.normalize();
  y_head = z_head.cross(x_head);
  R_WH.col(0) = x_head;
  R_WH.col(1) = y_head;
  R_WH.col(2) = z_head;
  R_HW = R_WH.transpose();
  R_HB = R_HW * R_WB;

  // Update qdot
  joint_vel_ = joint_vel_est_->update(q_rpy.tail(pino_model_.nv - 6));
  qdot.segment(0, 3) = Eigen::Vector3d::Zero();
  qdot.segment(3, 3) = ori_base_local_.omega;

  qdot[6] = standmode_input->joints_status.joint_h_roll_l.vel_fb;
  qdot[7] = standmode_input->joints_status.joint_h_pitch_l.vel_fb;
  qdot[8] = standmode_input->joints_status.joint_k_pitch_l.vel_fb;
  qdot[9] = standmode_input->joints_status.joint_w_pitch_l.vel_fb;

  qdot[10] = standmode_input->joints_status.joint_h_roll_r.vel_fb;
  qdot[11] = standmode_input->joints_status.joint_h_pitch_r.vel_fb;
  qdot[12] = standmode_input->joints_status.joint_k_pitch_r.vel_fb;
  qdot[13] = standmode_input->joints_status.joint_w_pitch_r.vel_fb;

  // Update qddot
  qddot.segment(0, 3) = ori_base_local_.acc;
  for (size_t i = 3; i < pino_model_.nv - 3; i++) {
    qddot[i] = 0;
  }

  q_fixed = q_pino.segment(7, pino_model_fixed_.nq);
  qdot_fixed = qdot.segment(6, pino_model_fixed_.nv);
  // joystick
  // joystick_->Update(standmode_input);
}

#endif

void RobotModel::UpdateModel() {
  AddFrames();  // 添加两个坐标系,左轮在左膝坐标系,右轮在右膝关节坐标系
  UpdateKinematic();
  UpdateDynamic();
  // UpdateMisc();  // 更新一些杂项
  PublishRobotStates();
}

void RobotModel::AddFrames() {
  Eigen::Vector3d vec_knee_wheel(-0.0984652, 0.022, -0.283378);
  pinocchio::SE3 T_knee_wheel(Eigen::Matrix3d::Identity(), vec_knee_wheel);
  pinocchio::Frame left_wheel_frame("left_wheel_frame", pino_model_.getJointId("left_knee_joint"), pino_model_.getFrameId("left_knee_joint"), T_knee_wheel,
                                    pinocchio::OP_FRAME);
  pinocchio::Frame right_wheel_frame("right_wheel_frame", pino_model_.getJointId("right_knee_joint"), pino_model_.getFrameId("right_knee_joint"), T_knee_wheel,
                                     pinocchio::OP_FRAME);
  pino_model_.addFrame(left_wheel_frame);
  pino_model_.addFrame(right_wheel_frame);

  pino_data_ = pinocchio::Data(pino_model_);
}

void RobotModel::UpdateKinematic() {
  pinocchio::forwardKinematics(pino_model_, pino_data_, q_pino);
  pinocchio::updateGlobalPlacements(pino_model_, pino_data_);
  pinocchio::updateFramePlacements(pino_model_, pino_data_);
  pinocchio::computeJointJacobians(pino_model_, pino_data_, q_pino);
  pinocchio::computeJointJacobiansTimeVariation(pino_model_, pino_data_, q_pino, qdot);
  pinocchio::centerOfMass(pino_model_, pino_data_, q_pino);


  int joint_id_left_wheel = pino_model_.getJointId("left_wheel_joint");
  int joint_id_right_wheel = pino_model_.getJointId("right_wheel_joint");
  pos_left_wheel = pino_data_.oMi[joint_id_left_wheel].translation();
  pos_right_wheel = pino_data_.oMi[joint_id_right_wheel].translation();
  // std::cout << "left_wheel_pos: " << pos_left_wheel << ", right_wheel_pos:" << pos_right_wheel << std::endl;


  // fixed model
  pinocchio::forwardKinematics(pino_model_fixed_, pino_data_fixed_, q_fixed);
  pinocchio::computeJointJacobians(pino_model_fixed_, pino_data_fixed_, q_fixed);
}

void RobotModel::UpdateDynamic() {
  // Calculate mass matrix and coriolis and gravitational forces used pinocchio
  pinocchio::crba(pino_model_, pino_data_, q_pino);
  pinocchio::computeCoriolisMatrix(pino_model_, pino_data_, q_pino, qdot);
  pinocchio::computeGeneralizedGravity(pino_model_, pino_data_, q_pino);

  pino_data_.M.triangularView<Eigen::Lower>() = pino_data_.M.transpose().triangularView<Eigen::Lower>();
  M = pino_data_.M;
  M_inverse = M.inverse();
  C = pino_data_.C * qdot;
  G = pino_data_.g;
  S.block(0, 6, pino_model_.nv - 6, pino_model_.nv - 6) = Eigen::MatrixXd::Identity(pino_model_.nv - 6, pino_model_.nv - 6);
  S_transpose_pinv = (S * S.transpose()).inverse() * S;
  mass = pino_data_.mass[0];
}

// void RobotModel::UpdateMisc() { fsm_id_ = joystick_->fsm_id(); }

void RobotModel::PublishRobotStates() {
  // ros_publisher_->Publish("l4a/model/q", q_rpy);
  // ros_publisher_->Publish("l4a/model/qdot", qdot);
  // ros_publisher_->Publish("l4a/model/qddot", qddot);
#if PHYSICS_ENABLE
  Eigen::VectorXd current_fb = Eigen::VectorXd::Zero(pino_model_.nv - 6);
  Eigen::VectorXd tau_fb = Eigen::VectorXd::Zero(pino_model_.nv - 6);
  current_fb[0] = standmode_input_->joints_status.joint_h_roll_l.current_fb;
  current_fb[1] = standmode_input_->joints_status.joint_h_pitch_l.current_fb;
  current_fb[2] = standmode_input_->joints_status.joint_k_pitch_l.current_fb;
  current_fb[3] = standmode_input_->joints_status.joint_w_pitch_l.current_fb;
  current_fb[4] = standmode_input_->joints_status.joint_h_roll_r.current_fb;
  current_fb[5] = standmode_input_->joints_status.joint_h_pitch_r.current_fb;
  current_fb[6] = standmode_input_->joints_status.joint_k_pitch_r.current_fb;
  current_fb[7] = standmode_input_->joints_status.joint_w_pitch_r.current_fb;

  tau_fb[0] = standmode_input_->joints_status.joint_h_roll_l.current_fb * 2.1;
  tau_fb[1] = standmode_input_->joints_status.joint_h_pitch_l.current_fb * 2.1;
  tau_fb[2] = standmode_input_->joints_status.joint_k_pitch_l.current_fb * 2.1;
  tau_fb[3] = standmode_input_->joints_status.joint_w_pitch_l.current_fb * 2.35;
  tau_fb[4] = standmode_input_->joints_status.joint_h_roll_r.current_fb * 2.1;
  tau_fb[5] = standmode_input_->joints_status.joint_h_pitch_r.current_fb * 2.1;
  tau_fb[6] = standmode_input_->joints_status.joint_k_pitch_r.current_fb * 2.1;
  tau_fb[7] = standmode_input_->joints_status.joint_w_pitch_r.current_fb * 2.35;
#endif
}

std::string RobotModel::GetJointString(int index) {
  auto joint = static_cast<AllJoints>(index);
  auto it = jointToString.find(joint);
  if (it != jointToString.end()) {
    return it->second;
  }
  return "Unknown";
}

}  // namespace l5a