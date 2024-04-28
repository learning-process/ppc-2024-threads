// Copyright 2024 Ivlev Alexander
#include "tbb/ivlev_a_convex_hull/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <thread>

using namespace std::chrono_literals;
using namespace ivlev_a_tbb;

bool ConvexHullTBBTaskSequential::pre_processing() {
  internal_order_test();
  try {
    size_t n = taskData->inputs_count[0];
    components.resize(n);
    for (size_t i = 0; i < n; i++) {
      auto* input_ = reinterpret_cast<std::pair<size_t, size_t>*>(taskData->inputs[i]);
      size_t tmp_size = taskData->inputs_count[i + 1];
      components[i].assign(input_, input_ + tmp_size);
      size_t m_w = 0;

      for (size_t j = 0; j < tmp_size; j++) {
        if (components[i][j].second > m_w) {
          m_w = components[i][j].second;
        }
      }
      // std::sort(components[i].begin(), components[i].end());
      sizes.emplace_back(components[i].back().first + 1, m_w + 1);
    }
    results.resize(taskData->inputs_count[0]);
  } catch (...) {
    std::cout << "pre\n";
    return false;
  }
  return true;
}

bool ConvexHullTBBTaskSequential::validation() {
  internal_order_test();
  try {
    if (taskData->inputs_count.size() <= 1) return false;
    if (taskData->outputs_count.empty()) return false;
    if (taskData->inputs_count[0] < 1) return false;
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      if (taskData->inputs[i] == nullptr) return false;
      if (taskData->inputs_count[i] < 1) return false;
    }
    if (taskData->outputs[0] == nullptr) return false;

    // m.b. dop check
  } catch (...) {
    std::cout << "val\n";
    return false;
  }

  return true;
}

bool ConvexHullTBBTaskSequential::run() {
  internal_order_test();
  try {
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      results[i] = Convex_Hull(components[i]);
    }
  } catch (...) {
    std::cout << "run\n";
    return false;
  }
  return true;
}

bool ConvexHullTBBTaskSequential::post_processing() {
  internal_order_test();
  try {
    size_t n = taskData->inputs_count[0];
    auto* outputs_ = reinterpret_cast<std::vector<std::pair<size_t, size_t>>*>(taskData->outputs[0]);
    for (size_t i = 0; i < n; i++) {
      std::sort(results[i].begin(), results[i].end());
      outputs_[i].clear();
      for (size_t j = 0; j < results[i].size(); j++) {
        outputs_[i].push_back(results[i][j]);
      }
    }
  } catch (...) {
    std::cout << "post\n";
    return false;
  }
  return true;
}

bool ConvexHullTBBTaskParallel::pre_processing() {
  internal_order_test();
  try {
    size_t n = taskData->inputs_count[0];
    components.resize(n);
    for (size_t i = 0; i < n; i++) {
      auto* input_ = reinterpret_cast<std::pair<size_t, size_t>*>(taskData->inputs[i]);
      size_t tmp_size = taskData->inputs_count[i + 1];
      components[i].assign(input_, input_ + tmp_size);
      size_t m_w = 0;

      for (size_t j = 0; j < tmp_size; j++) {
        if (components[i][j].second > m_w) {
          m_w = components[i][j].second;
        }
      }
      // std::sort(components[i].begin(), components[i].end());
      sizes.emplace_back(components[i].back().first + 1, m_w + 1);
    }
    results.resize(taskData->inputs_count[0]);
  } catch (...) {
    std::cout << "pre\n";
    return false;
  }
  return true;
}

bool ConvexHullTBBTaskParallel::validation() {
  internal_order_test();
  try {
    if (taskData->inputs_count.size() <= 1) return false;
    if (taskData->outputs_count.empty()) return false;
    if (taskData->inputs_count[0] < 1) return false;
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      if (taskData->inputs[i] == nullptr) return false;
      if (taskData->inputs_count[i] < 1) return false;
    }
    if (taskData->outputs[0] == nullptr) return false;

    // m.b. dop check
  } catch (...) {
    std::cout << "val\n";
    return false;
  }

  return true;
}

bool ConvexHullTBBTaskParallel::run() {
  internal_order_test();
  try {
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      results[i] = Convex_Hull(components[i]);
    }
  } catch (...) {
    std::cout << "run\n";
    return false;
  }
  return true;
}

bool ConvexHullTBBTaskParallel::post_processing() {
  internal_order_test();
  try {
    size_t n = taskData->inputs_count[0];
    auto* outputs_ = reinterpret_cast<std::vector<std::pair<size_t, size_t>>*>(taskData->outputs[0]);
    for (size_t i = 0; i < n; i++) {
      std::sort(results[i].begin(), results[i].end());
      outputs_[i].clear();
      for (size_t j = 0; j < results[i].size(); j++) {
        outputs_[i].push_back(results[i][j]);
      }
    }
  } catch (...) {
    std::cout << "post\n";
    return false;
  }
  return true;
}

inline size_t ivlev_a_tbb::rotation(const std::pair<ptrdiff_t, ptrdiff_t>& a, const std::pair<ptrdiff_t, ptrdiff_t>& b,
                                    const std::pair<ptrdiff_t, ptrdiff_t>& c) {
  int tmp = (b.first - a.first) * (c.second - b.second) - (b.second - a.second) * (c.first - b.first);
  if (tmp == 0) return 0;
  return ((tmp > 0) ? 1 : 2);
}

std::vector<std::pair<size_t, size_t>> ConvexHullTBBTaskSequential::Convex_Hull(
    const std::vector<std::pair<size_t, size_t>>& component_) {
  size_t n = component_.size();
  if (n < 3) return component_;

  size_t left = 0;
  for (size_t i = 1; i < n; i++)
    if (component_[i].second < component_[left].second) left = i;

  std::vector<std::pair<size_t, size_t>> res = {};
  size_t p = left;
  do {
    size_t q = (p + 1) % n;
    for (size_t i = 0; i < n; i++) {
      if (rotation(component_[p], component_[i], component_[q]) == 2 ||
          (rotation(component_[p], component_[i], component_[q]) == 0 && i > p && i != q)) {
        q = i;
      }
    }
    res.push_back(component_[q]);
    p = q;
  } while (p != left);

  return res;
}
/*
oneapi::tbb::v1::task* Convex_Hull_TMP::execute() {
  std::printf("tbb is runningn");
  return NULL;
}*/

std::vector<std::pair<size_t, size_t>> ConvexHullTBBTaskParallel::Convex_Hull_tmp(
    size_t p, size_t last, size_t n, const std::vector<std::pair<size_t, size_t>>& component_) {
  std::vector<std::pair<size_t, size_t>> res = {};
  size_t q = 0;

  do {
    q = (p + 1) % n;

    for (size_t i = 0; i < n; i++) {
      size_t tmp_r = rotation(component_[p], component_[i], component_[q]);
      if (tmp_r == 2 || (tmp_r == 0 && i > p && i != q)) {
        q = i;
      }
    }

    res.push_back(component_[q]);
    p = q;
  } while (p != last);

  return res;
}

std::vector<std::pair<size_t, size_t>> ConvexHullTBBTaskParallel::Convex_Hull(
    const std::vector<std::pair<size_t, size_t>>& component_) {
  size_t n = component_.size();
  if (n < 3) return component_;

  size_t left = 0;
  size_t top = 0;
  size_t rigth = 0;
  size_t down = 0;

  if (n < 8) {
    for (int i = 1; i < (int)n; i++) {
      if (component_[i].second < component_[left].second) left = i;
    }
    return ConvexHullTBBTaskParallel::Convex_Hull_tmp(left, left, n, component_);
  }

  std::vector<std::pair<size_t, size_t>> res = {};

  for (int i = 1; i < (int)n; i++) {
    if (component_[i].second <= component_[left].second) left = i;
    if (component_[i].first < component_[top].first) top = i;
    if (component_[i].second > component_[rigth].second) rigth = i;
    if (component_[i].first >= component_[down].first) down = i;
  }

  oneapi::tbb::concurrent_vector<std::vector<std::pair<size_t, size_t>>> tmp_res(4);

  oneapi::tbb::parallel_invoke(
      [&] { tmp_res[0] = ConvexHullTBBTaskParallel::Convex_Hull_tmp(left, top, n, component_); },
      [&] { tmp_res[1] = ConvexHullTBBTaskParallel::Convex_Hull_tmp(top, rigth, n, component_); },
      [&] { tmp_res[2] = ConvexHullTBBTaskParallel::Convex_Hull_tmp(rigth, down, n, component_); },
      [&] { tmp_res[3] = ConvexHullTBBTaskParallel::Convex_Hull_tmp(down, left, n, component_); });

  for (size_t i = 0; i < 4; i++) res.insert(res.end(), tmp_res[i].begin(), tmp_res[i].end());

  for(size_t i = 0; i < 4; i++)
    res.insert(res.end(), tmp_res[i].begin(), tmp_res[i].end());

  return res;
}
