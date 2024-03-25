// Copyright 2024 Shokurov Daniil
#include "seq/shokurov_d_convex_hull/include/convex_hull.hpp"

#include <cmath>
#include <thread>
#include <utility>
using namespace std::chrono_literals;

bool ConvexHullSequential::validation() {
  internal_order_test();
  try {
    if (taskData->inputs_count.size() != 1) throw 1;
    if (taskData->inputs_count[0] == 0) throw 2;
    if (taskData->outputs_count.size() != 1) throw 3;
    if (taskData->outputs_count[0] != taskData->inputs_count[0]) throw 4;
    return true;
  } catch (...) {
    return false;
  }
}

bool ConvexHullSequential::pre_processing() {
  internal_order_test();
  try {
    // Init value for input and output
    pair<double, double>* input_ = reinterpret_cast<pair<double, double>*>(taskData->inputs[0]);
    size_t n = taskData->inputs_count[0];

    for (size_t i = 0; i < n; ++i) {
      points.push_back(input_[i]);
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool ConvexHullSequential::run() {
  internal_order_test();
  try {
    si = solve(points);
    return true;
  } catch (...) {
    return false;
  }
}

bool ConvexHullSequential::post_processing() {
  internal_order_test();
  try {
    pair<double, double>* outputs_ = reinterpret_cast<pair<double, double>*>(taskData->outputs[0]);
    for (size_t i = 0; i < si; ++i) {
      outputs_[i] = points[i];
    }
    taskData->outputs_count[0] = si;
    return true;
  } catch (...) {
    return false;
  }
}

size_t ConvexHullSequential::solve(vector<pair<double, double>>& p) {
  const size_t n = p.size();
  size_t index = index_lowest_right_point(p);
  pair<double, double> p0 = p[index];
  pair<double, double> vec = {1.0, 0.0};
  pair<double, double> pk = p0;
  size_t k = 0;
  do {
    for (size_t i = k; i < n; ++i) {
      if (!comp(p[i], pk) && my_less(sub(p[i], pk), sub(p[k], pk), vec)) swap(p[k], p[i]);
    }
    vec = sub(p[k], pk);
    pk = p[k];
    ++k;
  } while (!comp(pk, p0));
  return k;
}

inline bool ConvexHullSequential::comp(const pair<double, double>& a, const pair<double, double>& b) {
  return normal(sub(a, b)) < 1e-6;
}

inline bool ConvexHullSequential::my_less(const pair<double, double>& v1, const pair<double, double>& v2,
                                          const pair<double, double>& v) {
  const double cosa = cos(v1, v), cosb = cos(v2, v);
  if (abs(cosa - cosb) > 1e-7)
    return cosa > cosb;
  else
    return normal(v1) > normal(v2);
}

inline pair<double, double> ConvexHullSequential::sub(const pair<double, double>& v1, const pair<double, double>& v2) {
  return pair<double, double>(v1.first - v2.first, v1.second - v2.second);
}

size_t ConvexHullSequential::index_lowest_right_point(const vector<pair<double, double>>& v) {
  auto less = [](const pair<double, double>& a, const pair<double, double>& b) {
    if (a.second != b.second)
      return a.second < b.second;
    else
      return a.first > b.first;
  };
  if (!v.empty()) {
    size_t index = 0;
    for (size_t i = 1; i < v.size(); ++i) {
      if (less(v[i], v[index])) {
        index = i;
      }
    }
    return index;
  } else {
    throw 11;
  }
}

inline double ConvexHullSequential::scalar_product(const pair<double, double>& v1, const pair<double, double>& v2) {
  return v1.first * v2.first + v1.second * v2.second;
}

inline double ConvexHullSequential::normal(const pair<double, double>& v) {
  return sqrt(v.first * v.first + v.second * v.second);
}

inline double ConvexHullSequential::cos(const pair<double, double>& v1, const pair<double, double>& v2) {
  const double n1 = normal(v1);
  const double n2 = normal(v2);
  return scalar_product(v1, v2) / (n1 * n2);
}