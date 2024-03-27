// Copyright 2024 Pozdnyakov Vasya
#pragma once

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Func = double (*)(double, double);

double flin(double x, double y);
double fxy(double x, double y);
double fysinx(double x, double y);
double fxexpy(double x, double y);

class PozdnyakovTaskSequential : public ppc::core::Task {
 public:
  explicit PozdnyakovTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Func f{};
  double x1{}, x2{}, y1{}, y2{};
  uint64_t n;
  double res;
};
