#include "trajectory_interpolation.h"

namespace GacMath {

/*
  double x = -2 * t * t * t + 3 * t * t;
return start_pos + (end_pos - start_pos) * x;*/
double CubicPolynomial(double start_pos, double end_pos, double t, double T) {
  double k = t / T;
  k = std::clamp(k, 0.0, 1.0);
  double x = -2 * k * k * k + 3 * k * k;
  return start_pos + (end_pos - start_pos) * x;
}

Eigen::Vector3d CubicPolynomial(Eigen::Vector3d start_pos,
                                Eigen::Vector3d end_pos, double t, double T) {
  double k = t / T;
  k = std::clamp(k, 0.0, 1.0);
  double x = -2 * k * k * k + 3 * k * k;
  return start_pos + (end_pos - start_pos) * x;
}

Eigen::Vector3d CubicPolynomial(const Eigen::Vector3d& start_pos,
                                const Eigen::Vector3d& end_pos,
                                const Eigen::Vector3d& start_vel,
                                const Eigen::Vector3d& end_vel, double t,
                                double T) {
  double k = std::clamp(t / T, 0.0, 1.0);
  double k2 = k * k;
  double k3 = k2 * k;

  Eigen::Vector3d term1 = (2 * k3 - 3 * k2 + 1) * start_pos;
  Eigen::Vector3d term2 = (k3 - 2 * k2 + k) * start_vel * T;
  Eigen::Vector3d term3 = (-2 * k3 + 3 * k2) * end_pos;
  Eigen::Vector3d term4 = (k3 - k2) * end_vel * T;

  return term1 + term2 + term3 + term4;
}

Eigen::Vector3d Cycloid(Eigen::Vector3d start_pos, Eigen::Vector3d end_pos,
                        double height, double phase) {
  if (phase < 0 || phase > 1) {
    throw std::range_error(" Phase must be in the range[0, 1] ");
  }

  double theta = phase * 2 * M_PI;
  double x = start_pos[0] + (end_pos[0] - start_pos[0]) *
                                (theta - std::sin(theta)) / (2 * M_PI);
  double y = start_pos[1] + (end_pos[1] - start_pos[1]) *
                                (theta - std::sin(theta)) / (2 * M_PI);
  double z = start_pos[2] + (height) * (1 - std::cos(theta)) / 2;

  return Eigen::Vector3d(x, y, z);
}

Eigen::Vector<double, 6> Cycloid(Eigen::Vector3d start_pos,
                                 Eigen::Vector3d end_pos, double height,
                                 double phase, double phase_dot) {
  if (phase < 0 || phase > 1) {
    throw std::range_error(" Phase must be in the range[0, 1] ");
  }

  double theta = phase * 2 * M_PI;
  double theta_dot = phase_dot * 2 * M_PI;
  double x = start_pos[0] + (end_pos[0] - start_pos[0]) *
                                (theta - std::sin(theta)) / (2 * M_PI);
  double y = start_pos[1] + (end_pos[1] - start_pos[1]) *
                                (theta - std::sin(theta)) / (2 * M_PI);
  double z = start_pos[2] + (height) * (1 - std::cos(theta)) / 2;

  double x_dot = (end_pos[0] - start_pos[0]) * (1 - std::cos(theta)) /
                 (2 * M_PI) * theta_dot;
  double y_dot = (end_pos[1] - start_pos[1]) * (1 - std::cos(theta)) /
                 (2 * M_PI) * theta_dot;
  double z_dot = (height)*std::sin(theta) / 2 * theta_dot;

  Eigen::Vector<double, 6> res = Eigen::Vector<double, 6>::Zero();
  res << x, y, z, x_dot, y_dot, z_dot;
  return res;
}

double Ramp(double input, double pre_input, double max_rate, double dt) {
  if (input > 1.0) input = 1.0;
  if (input < -1.0) input = -1.0;

  double max_delta = max_rate * dt;
  double delta = input - pre_input;

  if (delta > max_delta) delta = max_delta;
  if (delta < -max_delta) delta = -max_delta;

  pre_input += delta;
  std::clamp(pre_input, -1.0, 1.0);
  return pre_input;
}

double Ramp(double u, double tgt, double inc) {
  double output;
  if (abs(u - tgt) < inc)
    output = tgt;
  else if (u < tgt - inc)
    output = u + inc;
  else if (u > tgt + inc)
    output = u - inc;
  else
    output = tgt;

  return output;
}

}  // namespace GacMath
