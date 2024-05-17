// Copyright 2024 Musaev Ilgar
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

void ScaledIdentityMatrix(double* matrix, int n, double k = 1.0);
void IdentityMatrix(double* matrix, int n, double k = 1.0);
void GenerateRandomValue(double* matrix, int sz);

class MusaevTaskSequential : public ppc::core::Task {
 public:
  explicit MusaevTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double *A{}, *B{}, *C{};
  size_t n{};
};
