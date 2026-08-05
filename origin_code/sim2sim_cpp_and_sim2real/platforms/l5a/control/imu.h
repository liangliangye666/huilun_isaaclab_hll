#ifndef IMU_H
#define IMU_H

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <glog/logging.h>
#include <Eigen/Dense>
#include <iostream>

#include "kalman_filter.h"
#include "math/gac_math.h"
#include "ros_publisher.h"

namespace l5a {

struct Orientation {
  Eigen::Vector3d euler;      // 欧拉角
  Eigen::Quaterniond quat;    // 四元数
  Eigen::Vector3d acc;        // 加速度
  Eigen::Vector3d omega;      // 角速度
  Eigen::Vector3d euler_dot;  // 欧拉角变化率
};

class IMU {
 public:
  IMU(const Eigen::Vector3d& euler_install = Eigen::Vector3d::Zero(),
      const double acc_g = -9.81);
  ~IMU();
  void Update(Orientation& ori_base_local, Orientation& ori_base_world);

 private:
  KalmanFilter* imu_filter_;
  Eigen::Vector3d euler_install_;
  double acc_g_;
  bool is_initialized_;
  Eigen::Matrix3d R_imu02init_;
  // std::shared_ptr<ROS::Publisher> ros_publisher_;
};
}  // namespace l5a
#endif