#ifndef PD_CONTROLLER_H
#define PD_CONTROLLER_H

#include <Eigen/Dense>
#include <iostream>
#include <type_traits>

/**
 * @brief Constructor for a Proportional-Derivative (PD) controller.
 *
 * This constructor initializes a PD controller with the given proportional (kp)
 * and derivative (kd) gains.
 *
 * @param kp Proportional gain. A positive value that determines the response
 *           to the current error.
 * @param kd Derivative gain. A positive value that determines the response
 *           to the rate of change of the error.
 */

template <typename T>
class PdController {
 public:
  PdController() = default;
  PdController(const T& kp, const T& kd) : kp_(kp), kd_(kd) {
    if constexpr (std::is_arithmetic<T>::value) {
      x_actual_ = 0;
      x_desired_ = 0;
      xdot_desired_ = 0;
      xdot_actual_ = 0;
      xddot_desired_ = 0;
      x_error_ = 0;
      xdot_error_ = 0;
    } else {
      x_actual_ = Eigen::VectorXd::Zero(kp.size());
      x_desired_ = Eigen::VectorXd::Zero(kp.size());
      xdot_desired_ = Eigen::VectorXd::Zero(kp.size());
      xdot_actual_ = Eigen::VectorXd::Zero(kp.size());
      xddot_desired_ = Eigen::VectorXd::Zero(kp.size());
      x_error_ = Eigen::VectorXd::Ones(kp.size());
      xdot_error_ = Eigen::VectorXd::Zero(kp.size());
    }
  }
  PdController(const T& kp, const T& kd, const T& x_desired, const T& xdot_desired) : kp_(kp), kd_(kd), x_desired_(x_desired), xdot_desired_(xdot_desired) {
    if constexpr (std::is_arithmetic<T>::value) {
      x_actual_ = 0;
      xdot_actual_ = 0;
      xddot_desired_ = 0;
      x_error_ = 0;
      xdot_error_ = 0;
    } else {
      x_actual_ = Eigen::VectorXd::Zero(kp.size());
      xdot_actual_ = Eigen::VectorXd::Zero(kp.size());
      xddot_desired_ = Eigen::VectorXd::Zero(kp.size());
      x_error_ = Eigen::VectorXd::Ones(kp.size());
      xdot_error_ = Eigen::VectorXd::Zero(kp.size());
    }
  }

  ~PdController() {};
  void set_kp(T kp) { kp_ = kp; };
  void set_kd(T kd) { kd_ = kd; };
  void set_x_desired(T x_desired) { x_desired_ = x_desired; };
  void set_x_actual(T x_actual) { x_actual_ = x_actual; };
  void set_xdot_desired(T xdot_desired) { xdot_desired_ = xdot_desired; };
  void set_xdot_actual(T xdot_actual) { xdot_actual_ = xdot_actual; };
  void set_xddot_desired(T xddot_desired) { xddot_desired_ = xddot_desired; };
  void xddot_desired() { return xddot_desired_; };
  T x_error() { return x_error_; };
  T xdot_error() { return xdot_error_; };

  T Update() {
    if constexpr (std::is_arithmetic<T>::value) {
      // Scalar case
      x_error_ = x_desired_ - x_actual_;
      xdot_error_ = xdot_desired_ - xdot_actual_;
      xddot_desired_ = kp_ * x_error_ + kd_ * xdot_error_;
    } else {
      // Eigen vector case
      x_error_ = x_desired_ - x_actual_;
      xdot_error_ = xdot_desired_ - xdot_actual_;
      xddot_desired_ = kp_.cwiseProduct(x_error_) + kd_.cwiseProduct(xdot_error_);
    }

    return xddot_desired_;
  };

 private:
  T kp_;
  T kd_;
  T x_desired_;
  T x_actual_;
  T xdot_desired_;
  T xdot_actual_;
  T xddot_desired_;

  T x_error_;
  T xdot_error_;
};

#endif