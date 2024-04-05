// Copyright 2023 Sobol Liubov
#pragma once

#include <vector>
#include <string>
#include <cmath>

#include "core/task/include/task.hpp"

class Sobel_seq : public ppc::core::Task {
 public:
  explicit Sobel_seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int w{}, h{};
  std::vector<uint8_t> input_ = {};
  std::vector<uint8_t> res = {};
  
};

std::vector<uint8_t> getRandomPicture(int w, int h, uint8_t min, uint8_t max);