#include "kalman_filter.h"

namespace l5a {
KalmanFilter::KalmanFilter(int state_dim, int meas_dim, int input_dim)
    : state_dim_(state_dim),
      meas_dim_(meas_dim),
      input_dim_(input_dim),
      x_(Eigen::VectorXd::Zero(state_dim)),
      u_(Eigen::VectorXd::Zero(input_dim)),
      P_(Eigen::MatrixXd::Identity(state_dim, state_dim)),
      A_(Eigen::MatrixXd::Identity(state_dim, state_dim)),
      B_(Eigen::MatrixXd::Zero(state_dim, input_dim)),
      Q_(Eigen::MatrixXd::Zero(state_dim, state_dim)),
      H_(Eigen::MatrixXd::Zero(meas_dim, state_dim)),
      D_(Eigen::MatrixXd::Zero(meas_dim, input_dim)),
      R_(Eigen::MatrixXd::Identity(meas_dim, meas_dim)),
      E_(Eigen::MatrixXd::Identity(state_dim, state_dim)),
      K_(Eigen::MatrixXd::Zero(state_dim, meas_dim))

{}

void KalmanFilter::SetInitialState(const Eigen::VectorXd& x0,
                                   const Eigen::MatrixXd& P0) {
  x_ = x0;
  P_ = P0;
}

void KalmanFilter::SetSystemModel(const Eigen::MatrixXd& A,
                                  const Eigen::MatrixXd& B,
                                  const Eigen::MatrixXd& Q) {
  CheckMatrixSize(A, state_dim_, state_dim_);
  CheckMatrixSize(B, state_dim_, input_dim_);
  CheckMatrixSize(Q, state_dim_, state_dim_);
  A_ = A;
  B_ = B;
  Q_ = Q;
}

void KalmanFilter::SetMeasurementModel(const Eigen::MatrixXd& H,
                                       const Eigen::MatrixXd& R) {
  CheckMatrixSize(H, meas_dim_, state_dim_);
  CheckMatrixSize(R, meas_dim_, meas_dim_);
  H_ = H;
  R_ = R;
}

void KalmanFilter::SetMeasurementModel(const Eigen::MatrixXd& H,
                                       const Eigen::MatrixXd& D,
                                       const Eigen::MatrixXd& R) {
  CheckMatrixSize(H, meas_dim_, state_dim_);
  CheckMatrixSize(D, meas_dim_, input_dim_);
  CheckMatrixSize(R, meas_dim_, meas_dim_);
  H_ = H;
  D_ = D;
  R_ = R;
}

void KalmanFilter::Run(const Eigen::VectorXd& z, const Eigen::VectorXd& u) {
  // Predict
  x_ = A_ * x_ + B_ * u;
  P_ = A_ * P_ * A_.transpose() + Q_;

  // Adjust
  K_ = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse();
  x_ = x_ + K_ * (z - H_ * x_);
  P_ = (E_ - K_ * H_) * P_;
}

void KalmanFilter::Predict(const Eigen::VectorXd u) {
  // 更新控制向量
  u_ = u;

  // 预测状态
  x_ = A_ * x_ + B_ * u_;

  // 计算预测协方差矩阵 P 的更新
  P_ = A_ * P_ * A_.transpose() + Q_;
}

void KalmanFilter::Update(const Eigen::VectorXd z) {
  // 计算观测残差
  Eigen::VectorXd y = z - (H_ * x_ + D_ * u_);

  // 计算S矩阵，即观测噪声协方差
  Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;

  // 求解Sy以计算卡尔曼增益的第一部分
  Eigen::VectorXd Sy = S.llt().solve(y);

  // 更新状态估计
  x_ = x_ + P_ * H_.transpose() * Sy;

  // 计算SH，用于更新协方差矩阵
  Eigen::MatrixXd SH = S.lu().solve(H_);

  // 更新协方差矩阵P_
  P_ = (E_ - P_ * H_.transpose() * SH) * P_;

  // 确保P_是对称的，防止数值误差
  P_ = 0.5 * (P_ + P_.transpose());
}

template <typename Derived>
void KalmanFilter::CheckMatrixSize(const Eigen::MatrixBase<Derived>& mat,
                                   int expected_rows, int expected_cols) {
  if (mat.rows() != expected_rows || mat.cols() != expected_cols) {
    throw std::runtime_error("Matrix size mismatch");
  }
}

}  // namespace l5a