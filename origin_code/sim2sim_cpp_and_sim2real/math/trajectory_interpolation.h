#ifndef TRAJECTORY_INTERPOLATION_H
#define TRAJECTORY_INTERPOLATION_H

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif
#include <glog/logging.h>
#include <Eigen/Dense>

namespace GacMath {
double CubicPolynomial(double start_pos, double end_pos, double t, double T);
Eigen::Vector3d CubicPolynomial(Eigen::Vector3d start_pos,
                                Eigen::Vector3d end_pos, double t, double T);
Eigen::Vector3d CubicPolynomial(const Eigen::Vector3d& start_pos,
                                const Eigen::Vector3d& end_pos,
                                const Eigen::Vector3d& start_vel,
                                const Eigen::Vector3d& end_vel, double t,
                                double T);
Eigen::Vector3d Cycloid(Eigen::Vector3d start_pos, Eigen::Vector3d end_pos,
                        double height, double phase);
Eigen::Vector<double, 6> Cycloid(Eigen::Vector3d start_pos,
                                 Eigen::Vector3d end_pos, double height,
                                 double phase, double phase_dot);
double Ramp(double input, double pre_input, double max_rate, double dt);
double Ramp(double u, double tgt, double inc);
}  // namespace GacMath

#endif  // TRAJECTORY_INTERPOLATION_H