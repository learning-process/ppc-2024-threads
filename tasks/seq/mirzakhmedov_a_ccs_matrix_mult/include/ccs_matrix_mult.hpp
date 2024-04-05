// Copyright 2024 Mirzakhmedov Alexander
#pragma once

#include <complex>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

struct CCSSparseMatrix {
  int numRows, numCols;
  int numNonZeros;
  std::vector<int> columnPointers;
  std::vector<int> rowIndices;
  std::vector<std::complex<double>> nonzeroValues;

  CCSSparseMatrix(int numRows_ = 0, int numCols_ = 0, int numNonZeros_ = 0)
      : numRows(numRows_),
      numCols(numCols_),
      numNonZeros(numNonZeros_),
      columnPointers(numRows + 1),
      rowIndices(numNonZeros),
      nonzeroValues(numNonZeros) {}
};

class CSeq : public ppc::core::Task {
public:
  explicit CSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

private:
  CCSSparseMatrix* A, * B, * C;
};