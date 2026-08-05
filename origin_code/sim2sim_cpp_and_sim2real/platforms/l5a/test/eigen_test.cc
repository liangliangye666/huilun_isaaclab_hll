#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <chrono>
#include <iostream>

#include "gac_math.h"

#include <glog/logging.h>
using namespace std;

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove) {
  unsigned int numRows = matrix.rows() - 1;
  unsigned int numCols = matrix.cols();

  if (rowToRemove < numRows)
    matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) =
        matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

  matrix.conservativeResize(numRows, numCols);
}

void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove) {
  unsigned int numRows = matrix.rows();
  unsigned int numCols = matrix.cols() - 1;

  if (colToRemove < numCols)
    matrix.block(0, colToRemove, numRows, numCols - colToRemove) =
        matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);

  matrix.conservativeResize(numRows, numCols);
}

void removeRows(Eigen::MatrixXd& matrix, unsigned int rowToRemove,
                int row_num) {
  unsigned int numRows = matrix.rows() - row_num;
  unsigned int numCols = matrix.cols();

  if (rowToRemove < numRows) {
    matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) =
        matrix.block(rowToRemove + row_num, 0, numRows - rowToRemove, numCols);
  }
  matrix.conservativeResize(numRows, numCols);
}

void removeColumns(Eigen::MatrixXd& matrix, unsigned int colToRemove,
                   int col_num) {
  unsigned int numRows = matrix.rows();
  unsigned int numCols = matrix.cols() - col_num;

  if (colToRemove < numCols)
    matrix.block(0, colToRemove, numRows, numCols - colToRemove) =
        matrix.block(0, colToRemove + col_num, numRows, numCols - colToRemove);

  matrix.conservativeResize(numRows, numCols);
}

Eigen::MatrixXd MatrixPower(const Eigen::MatrixXd& A, int power) {
  if (power == 0) {
    // 返回单位矩阵
    return Eigen::MatrixXd::Identity(A.rows(), A.cols());
  }

  Eigen::MatrixXd result = Eigen::MatrixXd::Identity(A.rows(), A.cols());
  Eigen::MatrixXd base = A;
  int current_power = power;

  while (current_power > 0) {
    if (current_power % 2 == 1) {
      result = result * base;
    }
    base = base * base;  // 平方
    current_power /= 2;
  }

  return result;
}

int main() {
  google::InitGoogleLogging("eigen_test");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  Eigen::MatrixXd A(3, 3);
  A << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  LOG(INFO) << "Original Matrix A: \n" << A;
  double* A_data = A.data();
  for (size_t i = 0; i < 9; i++) {
    std::cout << A_data[i] << " " << std::flush;
  }
  std::cout << std::endl;
  Eigen::MatrixXd B = A.transpose();
  LOG(INFO) << "Original Matrix B: \n" << B;

  double* B_data = B.data();
  for (size_t i = 0; i < 9; i++) {
    std::cout << B_data[i] << " " << std::flush;
  }

  return 0;
}