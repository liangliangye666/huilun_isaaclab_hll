
#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <glog/logging.h>
#include <Eigen/Dense>
#include <iostream>
#include <qpOASES.hpp>

int main() {
  google::InitGoogleLogging("qpOASES_test");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  // J = x1^2 + x2^2   st. 2 <= x1 <= 4 ;-4 <= x2 <= -2
  qpOASES::real_t H[4] = {2, 0, 0, 2};
  qpOASES::real_t g[2] = {0, 0};
  qpOASES::real_t A[4] = {1, 0, 0, 1};
  qpOASES::real_t lbA[2] = {2, -4};
  qpOASES::real_t ubA[2] = {4, -2};

  qpOASES::QProblem example(2, 2);
  qpOASES::Options options;
  // options.printLevel = qpOASES::PL_NONE;
  // options.terminationTolerance = 1e-6;
  options.setToMPC();
  example.setOptions(options);

  /* Solve first QP. */
  qpOASES::int_t nWSR = 100;
  qpOASES::returnValue qp_returnvalue;
  qp_returnvalue = example.init(H, g, A, nullptr, nullptr, lbA, ubA, nWSR);
  qpOASES::real_t xOpt[2];
  qpOASES::real_t yOpt[2];
  example.getPrimalSolution(xOpt);
  example.getDualSolution(yOpt);
  example.printOptions();
  if (example.isSolved()) {
    std::cout << "Is Solved!" << std::endl;
  }
  printf("\nxOpt = [ %e , %e ];  yOpt = [ %e ];  objVal = %e\n\n", xOpt[0], xOpt[1], yOpt[0],
         example.getObjVal());
  std::cout << "Iterations taken: " << nWSR << std::endl;
  std::cout << "Final termination tolerance (precision): " << options.terminationTolerance
            << std::endl;

  //  Check qp_solver status
  int qp_status = qpOASES::getSimpleStatus(qp_returnvalue, qpOASES::BT_TRUE);
  switch (qp_status) {
    case 0:
      LOG(INFO) << "QP problem solved";
      break;
    case 1:
      LOG(ERROR) << "QP could not be solved within given number of iterations";
      break;
    case -1:
      LOG(ERROR) << "QP could not be solved due to an internal error";
      break;
    case -2:
      LOG(ERROR) << "QP is infeasible (and thus could not be solved)";
      break;
    case -3:
      LOG(ERROR) << "QP is unbounded (and thus could not be solved)";
      break;

    default:
      break;
  }
  return 0;
}
