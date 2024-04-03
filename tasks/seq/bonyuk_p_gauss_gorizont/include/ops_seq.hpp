//  Copyright 2023 Bonyuk Peter
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

std::vector<int> getImage(int n, int m, uint8_t min, uint8_t max);

class LinearGaussianFiltering : public ppc::core::Task {
 public:
  explicit LinearGaussianFiltering(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool examination() override;
  bool pre_proc() override;
  bool post_proc() override;
  bool run() override;

  int getPixel(int x, int y) { return input[x * width + y]; }
  void setPixel(int x, int y, int value) { res[x * width + y] = value; }

 private:
  int min{}, max{};
  std::vector<int> input = {};
  std::vector<int> res = {};
  int height{}, width{};
};
