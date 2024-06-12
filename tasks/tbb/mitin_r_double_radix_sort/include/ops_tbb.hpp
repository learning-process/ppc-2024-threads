// Copyright 2024 Mitin Roman
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace mitinr_radix_sort {

class SortRadixDoubleTaskTBB : public ppc::core::Task {
 public:
  explicit SortRadixDoubleTaskTBB(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double* data_ptr;
  size_t data_size;
  std::vector<double> bucket_arr[sizeof(double) * 256];
};

}  // namespace mitinr_radix_sort
