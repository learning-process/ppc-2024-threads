#pragma once

#include <vector>
#include <complex>

#include "core/task/include/task.hpp"

class TestTaskSequential : public ppc::core::Task {
public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

private:
  std::vector<std::complex<double>> input_matrix_1_;
  std::vector<std::complex<double>> input_matrix_2_;
  std::vector<std::complex<double>> output_matrix_;
  int rows_, cols_;
};