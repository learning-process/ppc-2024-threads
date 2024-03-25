// Copyright 2024 Shokurov Daniil
#pragma once

#include <string>
#include <vector>
#include <utility>
#include "core/task/include/task.hpp"
using namespace std;

class ConvexHullSequential : public ppc::core::Task {
 public:
  explicit ConvexHullSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  inline double normal(const pair<double, double>& v);
  inline double scalar_product(const pair<double, double>& v1, const pair<double, double>& v2);

  inline double cos(const pair<double, double>& v1, const pair<double, double>& v2) {
    const double n1 = normal(v1);
    const double n2 = normal(v2);
    return scalar_product(v1, v2) / (n1 * n2);
  }

  inline bool my_less(const pair<double, double>& v1, const pair<double, double>& v2, const pair<double, double>& v);

  size_t index_lowest_right_point(const vector<pair<double, double>>& v);

  inline pair<double, double> sub(const pair<double, double>& v1, const pair<double, double>& v2);

  inline bool comp(const pair<double, double>& a, const pair<double, double>& b);

  size_t solve(vector<pair<double, double>>& p);

 private:
  vector<pair<double, double>> points;
  size_t si = 0;
};
