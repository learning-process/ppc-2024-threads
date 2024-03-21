// Copyright 2023 Korablev Nikita
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

using func = double (*)(double, double);

double flin(double x, double y);
double fsinxPsiny(double x, double y);
double fcosxPcosy(double x, double y);
double fxy(double x, double y);
double fxyy(double x, double y);

class MonteCarloSequential : public ppc::core::Task {
 public:
  explicit MonteCarloSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  func function{};
  double Int1[2]{}, Int2[2]{};

  int N{};
  double res{};
};