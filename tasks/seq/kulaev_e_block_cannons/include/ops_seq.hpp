// Copyright 2024 Kulaev Zhenya
#pragma once

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace Kulaev_Seq {

class TestTaskSequentialKulaevCannon : public ppc::core::Task {
 public:
  explicit TestTaskSequentialKulaevCannon(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> result;
  int n = 0, m = 0;
};

std::vector<double> getRandomMatrix(int rows, int cols);

std::vector<double> multiplyMatrix(const std::vector<double>& A, const std::vector<double>& B, int rows_A, int col_B);

std::vector<double> cannonMatrixMultiplication(const std::vector<double>& A, const std::vector<double>& B, int n,
                                               int m);
}