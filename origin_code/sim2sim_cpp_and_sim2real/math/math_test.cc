
#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include "gac_math.h"
#include "glog/logging.h"

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  Eigen::Vector3d start_pos(10.0, 0.0, 1.0);
  Eigen::Vector3d end_pos(20.0, 0, 3);
  double height = 1.0;
  for (double t = 0; t < 0.4; t += 0.05) {
    Eigen::Vector3d result = GacMath::CubicPolynomial(start_pos, end_pos, t, 0.4);
    LOG(INFO) << "result.z(): " << result.z();
  }

  return 0;
}