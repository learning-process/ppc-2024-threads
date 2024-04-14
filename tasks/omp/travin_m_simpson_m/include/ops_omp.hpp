// Copyright 2024 Travin Maksim
#pragma once

#include <string>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>
#include <utility>
#include <omp.h>

#include "core/task/include/task.hpp"

namespace travin_omp {

using function = double (*)(double, double);

class SimpsonTaskSeq : public ppc::core::Task {
 public:
  explicit SimpsonTaskSeq(std::shared_ptr<ppc::core::TaskData> taskData_, function fun_)
      : Task(std::move(taskData_)), func(std::move(fun_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  function func;
  double res{};
  int a{}, b{}, c{}, d{}, n{};
};

class SimpsonTaskOmp : public ppc::core::Task {
 public:
  explicit SimpsonTaskOmp(std::shared_ptr<ppc::core::TaskData> taskData_, function fun_)
      : Task(std::move(taskData_)), func(std::move(fun_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  function func;
  double res{};
  int a{}, b{}, c{}, d{}, n{};
};

double Simpson(function func, double a, double b, double y);
double SimpsonSeq(function func, int a, int b, int c, int d, int steps);
double SimpsonOmp(function func, int a, int b, int c, int d, int steps);
double LinFunc(double x, double y);
double TrigFunc(double x, double y);
double MulFunc(double x, double y);
double ExpFunc(double x, double y);
double RandFunc(double x, double y);

} // namespace travin_omp