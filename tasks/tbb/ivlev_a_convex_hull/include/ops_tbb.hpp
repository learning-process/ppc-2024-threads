// Copyright 2024 Ivlev Alexander
#pragma once

#include <tbb/tbb.h>

#include <algorithm>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace ivlev_a_tbb {
class ConvexHullTBBTaskSequential : public ppc::core::Task {
 public:
  explicit ConvexHullTBBTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::pair<size_t, size_t>> sizes;
  std::vector<std::vector<std::pair<size_t, size_t>>> components;
  std::vector<std::vector<std::pair<size_t, size_t>>> results;

  static std::vector<std::pair<size_t, size_t>> Convex_Hull(const std::vector<std::pair<size_t, size_t>>& component_);
};

class ConvexHullTBBTaskParallel : public ppc::core::Task {
 public:
  explicit ConvexHullTBBTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::pair<size_t, size_t>> sizes;
  std::vector<std::vector<std::pair<size_t, size_t>>> components;
  std::vector<std::vector<std::pair<size_t, size_t>>> results;

  static std::vector<std::pair<size_t, size_t>> Convex_Hull_tmp(
      size_t p, size_t last, size_t n, const std::vector<std::pair<size_t, size_t>>& component_);
  static std::vector<std::pair<size_t, size_t>> Convex_Hull(const std::vector<std::pair<size_t, size_t>>& component_);
};
/*
class Convex_Hull_TMP: public oneapi::tbb::task {
 public:
  oneapi::tbb::task* execute();
}
*/
size_t rotation(const std::pair<ptrdiff_t, ptrdiff_t>& a, const std::pair<ptrdiff_t, ptrdiff_t>& b,
                const std::pair<ptrdiff_t, ptrdiff_t>& c);
}  // namespace ivlev_a_tbb