#include "imu.h"

namespace l5a {

IMU::IMU(const Eigen::Vector3d& euler_install, const double acc_g) {
  // TODO: MadgwickFilter

  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  is_initialized_ = false;
  R_imu02init_.setIdentity();

  euler_install_ = euler_install;
  acc_g_ = acc_g;

//   ros_publisher_ = ROS::Publisher::getInstance();

  // Kalman Filter Initialization
  imu_filter_ = new KalmanFilter(9, 9);
  Eigen::MatrixXd R_imu = Eigen::MatrixXd::Identity(9, 9);
  Eigen::VectorXd R_diag_elements(9);
  R_diag_elements << 10 * Eigen::Vector3d::Ones(),
      100 * Eigen::Vector3d::Ones(), 100 * Eigen::Vector3d::Ones();
  R_imu.diagonal() = R_diag_elements;
  imu_filter_->SetInitialState(Eigen::VectorXd::Zero(9),
                               Eigen::MatrixXd::Identity(9, 9));
  imu_filter_->SetSystemModel(Eigen::MatrixXd::Identity(9, 9),
                              Eigen::VectorXd::Zero(9),
                              Eigen::MatrixXd::Identity(9, 9));
  imu_filter_->SetMeasurementModel(Eigen::MatrixXd::Identity(9, 9),
                                   Eigen::VectorXd::Zero(9), R_imu);
}

IMU::~IMU() {}

void IMU::Update(Orientation& ori_base_local, Orientation& ori_base_world) {
//   ros_publisher_->Publish("l5a/est/imu_raw_euler", ori_base_world.euler);
//   ros_publisher_->Publish("l5a/est/imu_raw_omega", ori_base_local.omega);
//   ros_publisher_->Publish("l5a/est/imu_raw_acc", ori_base_local.acc);

  Eigen::Vector<double, 9> imu_data =
      Eigen::Vector<double, 9>::Zero();  // Initialize with zeros
  imu_data << ori_base_world.euler, ori_base_local.omega, ori_base_local.acc;

#if PHYSICS_ENABLE
  imu_data[0] -= (imu_data[0] > M_PI) * 2.0 * M_PI;
  imu_filter_->Predict();
  imu_filter_->Update(imu_data);
  imu_data = imu_filter_->GetState();
#endif

  if (!is_initialized_) {
    R_imu02init_ = GacMath::EulerToRotationMatrix(imu_data[0], 0, 0);
    is_initialized_ = true;
  }

  // Imu0 是 imu 通电时形成的坐标系 imu输出的euler是相对于该坐标系的
  // Imu 定义为 imu 外壳标注的坐标系 imu输出的acc omega是相对于该坐标系的
  // Init 定义为进入算法时刻的 Imu 坐标系，与Imu0之间存在一个固定值的旋转
  // Base 是机器人基坐标系，固连于机器人基座，与之运动 与Local系之间有 roll
  // pitch Local 是机器人的朝向坐标系 x轴指向机器人的前方 与 World系之间只有 yaw
  // World 进入算法时，在Local系正下方地面上定义的固定坐标系 与初始Local同向

  Eigen::Vector3d euler_imu0(imu_data[0], imu_data[1], imu_data[2]);
  Eigen::Vector3d omega_imu(imu_data[3], imu_data[4], imu_data[5]);
  Eigen::Vector3d acc_imu(imu_data[6], imu_data[7], imu_data[8]);

  Eigen::Matrix3d R_imu02imu = GacMath::EulerToRotationMatrix(euler_imu0);
  Eigen::Matrix3d R_imu2base = GacMath::EulerToRotationMatrix(euler_install_);
  Eigen::Matrix3d R_init2world = GacMath::EulerToRotationMatrix(euler_install_);

  // R_Imu02Imu * R_Imu2Base(install) = R_Imu02Init * R_Init2World(install) *
  // R_World2Base = R_Imu02Base
  Eigen::Matrix3d R_world2base =
      (R_imu02init_ * R_init2world).transpose() * R_imu02imu * R_imu2base;
  Eigen::Quaterniond quat_world2base =
      GacMath::RotationMatrixToQuaternion(R_world2base);
  Eigen::Vector3d euler_world2base =
      GacMath::QuaternionToEuler(quat_world2base);

  Eigen::Vector3d euler_local2base =
      Eigen::Vector3d(0, euler_world2base[1], euler_world2base[2]);
  Eigen::Matrix3d R_local2base =
      GacMath::EulerToRotationMatrix(euler_local2base);
  Eigen::Quaterniond quat_local2base =
      GacMath::EulerToQuaternion(euler_local2base);

  Eigen::Vector3d omega_local =
      R_local2base * R_imu2base.transpose() * omega_imu;
  Eigen::Vector3d acc_local = R_local2base * R_imu2base.transpose() * acc_imu;
  acc_local[2] += acc_g_;

  Eigen::Matrix3d R_world2local =
      GacMath::EulerToRotationMatrix(euler_world2base[0], 0, 0);

  Eigen::Vector3d omega_world = R_world2local * omega_local;
  Eigen::Vector3d acc_world = R_world2local * acc_local;

  Eigen::Vector3d euler_dot_local =
      GacMath::EulerToRotationMatrix(euler_local2base[0], 0, 0).inverse() *
      omega_local;
  Eigen::Vector3d euler_dot_world =
      GacMath::EulerToRotationMatrix(euler_world2base[0], 0, 0).inverse() *
      omega_world;

  ori_base_local.euler = euler_local2base;
  ori_base_local.quat = quat_local2base;
  ori_base_local.acc = acc_local;
  ori_base_local.omega = omega_local;
  ori_base_local.euler_dot = euler_dot_local;

  ori_base_world.euler = euler_world2base;
  ori_base_world.quat = quat_world2base;
  ori_base_world.acc = acc_world;
  ori_base_world.omega = omega_world;
  ori_base_world.euler_dot = euler_dot_world;

//   ros_publisher_->Publish("l5a/est/imu_euler", ori_base_local.euler);
//   ros_publisher_->Publish("l5a/est/imu_omega", ori_base_local.omega);
//   ros_publisher_->Publish("l5a/est/imu_acc", ori_base_local.acc);
}

}  // namespace l5a
