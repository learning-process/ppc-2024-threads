// Copyright 2024 Shipitsin Alex
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class ShipitsinTaskSequential : public ppc::core::Task {
 public:
  explicit ShipitsinTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double x_lim_l, x_lim_u, y_lim_l, y_lim_u, res, err, h_x, h_y;
  uint64_t n;
};
