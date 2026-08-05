#ifndef MY_MATH_H
#define MY_MATH_H

#include <iostream>

#include <Eigen/Dense>
namespace GacMath {
Eigen::Matrix3d EulerToRotationMatrix(const double& yaw, const double& pitch,
                                      const double& roll);
Eigen::Matrix3d EulerToRotationMatrix(const Eigen::Vector3d& euler);
Eigen::Quaterniond EulerToQuaternion(const double& yaw, const double& pitch,
                                     const double& roll);
Eigen::Quaterniond EulerToQuaternion(const Eigen::Vector3d& eluer);
Eigen::Matrix3d RPYToRotationMatrix(const double& roll, const double& pitch,
                                    const double& yaw);
Eigen::Quaterniond RPYToQuaternion(const double& roll, const double& pitch,
                                   const double& yaw);
Eigen::Quaterniond RPYToQuaternion(Eigen::Vector3d& RPY);
Eigen::Vector3d RotationMatrixToEuler(Eigen::Matrix3d& R);
Eigen::Vector3d RotationMatrixToRPY(Eigen::Matrix3d& R);
Eigen::Quaterniond RotationMatrixToQuaternion(Eigen::Matrix3d& R);
Eigen::Vector3d QuaternionToEuler(Eigen::Quaterniond& q);
Eigen::Vector3d QuaternionToRPY(Eigen::Quaterniond& q);
Eigen::Matrix3d QuaternionToRotationMatrix(Eigen::Quaterniond& q);
Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& M);
Eigen::MatrixXd PseudoInverseOptimized(const Eigen::MatrixXd& A,
                                       double epsilon = 1e-9);
Eigen::MatrixXd PseudoInverseDynamic(const Eigen::MatrixXd& A,
                                     const Eigen::MatrixXd& M_inverse);
Eigen::Matrix3d SkewMatrix(Eigen::Vector3d vec);
Eigen::Matrix<double, 6, 6> AdjointRepresentationMatrix(
    const Eigen::Matrix4d& T);
Eigen::Vector3d EulerDotToOmegaBody(const Eigen::Vector3d& rpy,
                                    const Eigen::Vector3d& eluer_dot);
Eigen::Vector3d EulerDotToOmegaStationary(const Eigen::Vector3d& rpy,
                                          const Eigen::Vector3d& eluer_dot);
double LinearInterpolation(double start_pos, double end_pos, double t);
int Rank(Eigen::MatrixXd& A);
}  // namespace GacMath

#endif