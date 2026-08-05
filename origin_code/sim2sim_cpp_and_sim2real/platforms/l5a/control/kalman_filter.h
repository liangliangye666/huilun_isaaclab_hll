#ifndef KALMANFILTER_H_
#define KALMANFILTER_H_

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <Eigen/Dense>
#include <iostream>

#include <glog/logging.h>

namespace l5a {
class KalmanFilter {
 public:
  // 构造函数，初始化各个维度和相关矩阵
  KalmanFilter(int state_dim, int meas_dim, int ctrl_dim = 1);
  KalmanFilter() {};

  void SetInitialState(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0);
  void SetSystemModel(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                      const Eigen::MatrixXd& Q);
  void SetMeasurementModel(const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);
  void SetMeasurementModel(const Eigen::MatrixXd& H, const Eigen::MatrixXd& D,
                           const Eigen::MatrixXd& R);
  void Run(const Eigen::VectorXd& z, const Eigen::VectorXd& u);

  void Predict(const Eigen::VectorXd u = Eigen::VectorXd::Ones(1));
  void Update(const Eigen::VectorXd z);

  Eigen::VectorXd x() { return x_; };
  Eigen::VectorXd P() { return P_; };

 private:
  template <typename Derived>
  void CheckMatrixSize(const Eigen::MatrixBase<Derived>& mat, int expected_rows,
                       int expected_cols);

  int state_dim_;
  int meas_dim_;
  int input_dim_;
  Eigen::VectorXd x_;
  Eigen::VectorXd u_;
  Eigen::MatrixXd K_, P_;
  Eigen::MatrixXd A_, B_, H_, D_;
  Eigen::MatrixXd Q_, R_;
  Eigen::MatrixXd E_;
};
}  // namespace l5a

#endif  // KALMANFILTER_H_