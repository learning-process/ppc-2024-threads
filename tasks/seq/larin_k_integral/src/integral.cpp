// Copyright 2024 Larin Konstantin

#include "seq/larin_k_integral/include/integral.hpp"

#include <cmath>

namespace larin {

num_t integral_impl(const std::vector<limit_t>& limits, const func_t& func, state_t& coords, size_t iter, num_t step) {
  size_t dim = limits.size();
  if (iter == dim) {
    return func(coords);
  }

  num_t result{};
  auto x_min = limits[iter].first(coords);
  auto x_max = limits[iter].second(coords);
  auto prev = x_min;

  coords.resize(iter + 1);
  size_t steps = (x_max - x_min) / step;
  for (size_t i = 1; i <= steps; i++) {
    auto x = x_min + step * i;
    auto m = (x + prev) / 2;
    prev = x;

    coords[iter] = m;

    result += integral_impl(limits, func, coords, iter + 1, step);
  }

  return result;
}

num_t integral(const std::vector<limit_t>& limits, const func_t& func, num_t step) {
  size_t dim = limits.size();

  num_t coef = std::pow(step, dim);

  state_t tmp;
  return coef * integral_impl(limits, func, tmp, 0, step);
}

bool useless_class_that_exists_because_of_disgusting_api::pre_processing() { return true; }

bool useless_class_that_exists_because_of_disgusting_api::validation() { return true; }

bool useless_class_that_exists_because_of_disgusting_api::run() {
  try {
    result_that_cannot_be_returned_from_run = integral(limits, func, step);
    return true;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }
  return false;
}

bool useless_class_that_exists_because_of_disgusting_api::post_processing() { return true; }
}  // namespace larin
