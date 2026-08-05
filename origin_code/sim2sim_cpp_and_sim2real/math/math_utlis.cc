#include "math_utlis.h"

namespace GacMath {
Eigen::Matrix3d EulerToRotationMatrix(const double& yaw, const double& pitch,
                                      const double& roll) {
  Eigen::AngleAxisd rot_z(yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd rot_y(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rot_x(roll, Eigen::Vector3d::UnitX());

  Eigen::Matrix3d R = (rot_z * rot_y * rot_x).toRotationMatrix();

  return R;
}

Eigen::Matrix3d EulerToRotationMatrix(const Eigen::Vector3d& euler) {
  Eigen::AngleAxisd rot_z(euler[0], Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd rot_y(euler[1], Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rot_x(euler[2], Eigen::Vector3d::UnitX());

  Eigen::Matrix3d R = (rot_z * rot_y * rot_x).toRotationMatrix();

  return R;
}

Eigen::Quaterniond EulerToQuaternion(const double& yaw, const double& pitch,
                                     const double& roll) {
  double cr = cos(roll / 2);
  double sr = sin(roll / 2);
  double cp = cos(pitch / 2);
  double sp = sin(pitch / 2);
  double cy = cos(yaw / 2);
  double sy = sin(yaw / 2);

  double w = cr * cp * cy + sr * sp * sy;
  double x = sr * cp * cy - cr * sp * sy;
  double y = cr * sp * cy + sr * cp * sy;
  double z = cr * cp * sy - sr * sp * cy;

  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Quaterniond EulerToQuaternion(const Eigen::Vector3d& eluer) {
  double roll = eluer[2];
  double pitch = eluer[1];
  double yaw = eluer[0];

  double cr = cos(roll / 2);
  double sr = sin(roll / 2);
  double cp = cos(pitch / 2);
  double sp = sin(pitch / 2);
  double cy = cos(yaw / 2);
  double sy = sin(yaw / 2);

  double w = cr * cp * cy + sr * sp * sy;
  double x = sr * cp * cy - cr * sp * sy;
  double y = cr * sp * cy + sr * cp * sy;
  double z = cr * cp * sy - sr * sp * cy;

  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Matrix3d RPYToRotationMatrix(const double& roll, const double& pitch,
                                    const double& yaw) {
  Eigen::AngleAxisd rot_z(yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd rot_y(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rot_x(roll, Eigen::Vector3d::UnitX());

  Eigen::Matrix3d R = (rot_z * rot_y * rot_x).toRotationMatrix();

  return R;
}

Eigen::Quaterniond RPYToQuaternion(const double& roll, const double& pitch,
                                   const double& yaw) {
  double cr = cos(roll / 2);
  double sr = sin(roll / 2);
  double cp = cos(pitch / 2);
  double sp = sin(pitch / 2);
  double cy = cos(yaw / 2);
  double sy = sin(yaw / 2);

  double w = cr * cp * cy + sr * sp * sy;
  double x = sr * cp * cy - cr * sp * sy;
  double y = cr * sp * cy + sr * cp * sy;
  double z = cr * cp * sy - sr * sp * cy;

  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Quaterniond RPYToQuaternion(Eigen::Vector3d& RPY) {
  double roll = RPY[0];
  double pitch = RPY[1];
  double yaw = RPY[2];

  double cr = cos(roll / 2);
  double sr = sin(roll / 2);
  double cp = cos(pitch / 2);
  double sp = sin(pitch / 2);
  double cy = cos(yaw / 2);
  double sy = sin(yaw / 2);

  double w = cr * cp * cy + sr * sp * sy;
  double x = sr * cp * cy - cr * sp * sy;
  double y = cr * sp * cy + sr * cp * sy;
  double z = cr * cp * sy - sr * sp * cy;

  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Vector3d RotationMatrixToEuler(Eigen::Matrix3d& R) {
  double pitch =
      std::atan2(-R(2, 0), std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0)));
  double yaw = std::atan2(R(1, 0) / cos(pitch), R(0, 0) / cos(pitch));
  double roll = std::atan2(R(2, 1) / cos(pitch), R(2, 2) / cos(pitch));
  return Eigen::Vector3d(yaw, pitch, roll);
}

Eigen::Vector3d RotationMatrixToRPY(Eigen::Matrix3d& R) {
  double pitch =
      std::atan2(-R(2, 0), std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0)));
  double yaw = std::atan2(R(1, 0) / cos(pitch), R(0, 0) / cos(pitch));
  double roll = std::atan2(R(2, 1) / cos(pitch), R(2, 2) / cos(pitch));
  return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Quaterniond RotationMatrixToQuaternion(Eigen::Matrix3d& R) {
  Eigen::Quaterniond quat(R);
  return quat;
}

Eigen::Vector3d QuaternionToEuler(Eigen::Quaterniond& q) {
  double yaw = std::atan2(2 * (q.w() * q.z() + q.x() * q.y()),
                          1 - 2 * (q.y() * q.y() + q.z() * q.z()));  // Z轴旋转
  double pitch = std::asin(-2 * (q.x() * q.z() - q.w() * q.y()));  // Y轴旋转
  double roll = std::atan2(2 * (q.w() * q.x() + q.y() * q.z()),
                           1 - 2 * (q.x() * q.x() + q.y() * q.y()));  // X轴旋转

  return Eigen::Vector3d(yaw, pitch, roll);
}

Eigen::Vector3d QuaternionToRPY(Eigen::Quaterniond& q) {
  double yaw = std::atan2(2 * (q.w() * q.z() + q.x() * q.y()),
                          1 - 2 * (q.y() * q.y() + q.z() * q.z()));  // Z轴旋转
  double pitch = std::asin(-2 * (q.x() * q.z() - q.w() * q.y()));  // Y轴旋转
  double roll = std::atan2(2 * (q.w() * q.x() + q.y() * q.z()),
                           1 - 2 * (q.x() * q.x() + q.y() * q.y()));  // X轴旋转

  return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Matrix3d QuaternionToRotationMatrix(Eigen::Quaterniond& q) {
  Eigen::Matrix3d R;
  R << 1 - 2 * (q.y() * q.y() + q.z() * q.z()),
      2 * (q.x() * q.y() - q.z() * q.w()), 2 * (q.x() * q.z() + q.y() * q.w()),
      2 * (q.x() * q.y() + q.z() * q.w()),
      1 - 2 * (q.x() * q.x() + q.z() * q.z()),
      2 * (q.y() * q.z() - q.x() * q.w()), 2 * (q.x() * q.z() - q.y() * q.w()),
      2 * (q.y() * q.z() + q.x() * q.w()),
      1 - 2 * (q.x() * q.x() + q.y() * q.y());

  return R;
}

Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& M) {
  double damp = 0;
  Eigen::MatrixXd Mres;
  Mres = M * M.transpose();
  //    Mres=*M.transpose()* pseudoInv_SVD(Mres);
  Mres.diagonal().array() += damp;
  Mres = M.transpose() * Mres.completeOrthogonalDecomposition().pseudoInverse();
  return Mres;
}

Eigen::MatrixXd PseudoInverseOptimized(const Eigen::MatrixXd& A,
                                       double epsilon) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A.rows(), A.cols());
  svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

  const auto& U = svd.matrixU();
  const auto& V = svd.matrixV();
  const auto& S = svd.singularValues();

  typename Eigen::MatrixXd::Index rank = S.size();
  Eigen::MatrixXd S_inv(S.size(), S.size());
  S_inv.setZero();
  for (Eigen::Index i = 0; i < S.size(); ++i) {
    if (S(i) > epsilon) {
      S_inv(i, i) = 1.0 / S(i);
      rank = i + 1;
    }
  }

  return V.leftCols(rank) * S_inv.topLeftCorner(rank, rank) *
         U.leftCols(rank).adjoint();
}

Eigen::MatrixXd PseudoInverseDynamic(const Eigen::MatrixXd& A,
                                     const Eigen::MatrixXd& M_inverse) {
  Eigen::MatrixXd J = M_inverse * A.transpose() *
                      PseudoInverseOptimized(A * M_inverse * A.transpose());
  return J;
}

Eigen::Matrix3d SkewMatrix(Eigen::Vector3d vec) {
  Eigen::Matrix3d skew{
      {0, -vec[2], vec[1]}, {vec[2], 0, -vec[0]}, {-vec[1], vec[0], 0}};
  return skew;
}

Eigen::Matrix<double, 6, 6> AdjointRepresentationMatrix(
    const Eigen::Matrix4d& T) {
  Eigen::Matrix3d R = T.block(0, 0, 3, 3);
  Eigen::Vector3d p = T.block(0, 3, 3, 1);
  Eigen::Matrix<double, 6, 6> Adj = Eigen::Matrix<double, 6, 6>::Zero();
  Adj.block<3, 3>(0, 0) = R;
  Adj.block<3, 3>(0, 3) = GacMath::SkewMatrix(p) * R;
  Adj.block<3, 3>(3, 3) = R;

  return Adj;
}

int Rank(Eigen::MatrixXd& A) {
  Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(A);

  int rank = qr.rank();
  return rank;
}

Eigen::Vector3d EulerDotToOmegaBody(const Eigen::Vector3d& rpy,
                                    const Eigen::Vector3d& eluer_dot) {
  double roll = rpy[0];
  double pitch = rpy[1];
  double yaw = rpy[2];
  Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
  R(0, 0) = 1;
  R(0, 2) = -sin(pitch);
  R(1, 1) = cos(roll);
  R(1, 2) = sin(roll) * cos(pitch);
  R(2, 1) = -sin(roll);
  R(2, 2) = cos(roll) * cos(pitch);

  return R * eluer_dot;
}

Eigen::Vector3d EulerDotToOmegaStationary(const Eigen::Vector3d& rpy,
                                          const Eigen::Vector3d& eluer_dot) {
  double roll = rpy[0];
  double pitch = rpy[1];
  double yaw = rpy[2];
  Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
  R(0, 0) = cos(yaw) * cos(pitch);
  R(0, 1) = -sin(yaw);
  R(1, 0) = sin(yaw) * cos(pitch);
  R(1, 1) = cos(yaw);
  R(2, 0) = -sin(pitch);
  R(2, 2) = 1;

  return R * eluer_dot;
}

double LinearInterpolation(double start_pos, double end_pos, double t) {
  return start_pos + (end_pos - start_pos) * t;
}

}  // namespace GacMath
