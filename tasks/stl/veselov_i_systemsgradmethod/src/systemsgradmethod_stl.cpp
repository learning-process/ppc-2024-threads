// Copyright 2024 Veselov Ilya

#include "stl/veselov_i_systemsgradmethod/include/systemsgradmethod_stl.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

double dotProduct(const std::vector<double> &aa, const std::vector<double> &bb) {
  double result = 0.0;
  for (size_t i = 0; i < aa.size(); ++i) {
    result += aa[i] * bb[i];
  }
  return result;
}

std::vector<double> matrixVectorProduct(const std::vector<double> &Aa, const std::vector<double> &xx, int n) {
  std::vector<double> result(n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result[i] += Aa[i * n + j] * xx[j];
    }
  }
  return result;
}

std::vector<double> SLEgradSolver(const std::vector<double> &Aa, const std::vector<double> &bb, int n,
                                  double tol = 1e-6) {
  std::vector<double> res(n, 0.0);
  std::vector<double> r = bb;
  std::vector<double> p = r;
  std::vector<double> r_old = bb;

  while (true) {
    std::vector<std::future<void>> futures;

    std::vector<double> Ap(n);
    futures.push_back(std::async(std::launch::async, [&]() { Ap = matrixVectorProduct(Aa, p, n); }));

    double alpha = 0.0;
    double r_dot_r = dotProduct(r, r);

    futures.push_back(std::async(std::launch::async, [&]() { alpha = r_dot_r / dotProduct(Ap, p); }));

    for (auto &f : futures) {
      f.wait();
    }

    for (size_t i = 0; i < res.size(); ++i) {
      res[i] += alpha * p[i];
    }

    for (size_t i = 0; i < r.size(); ++i) {
      r[i] = r_old[i] - alpha * Ap[i];
    }

    if (sqrt(dotProduct(r, r)) < tol) {
      break;
    }

    double beta = dotProduct(r, r) / r_dot_r;

    for (size_t i = 0; i < p.size(); ++i) {
      p[i] = r[i] + beta * p[i];
    }

    r_old = r;
  }

  return res;
}

bool SystemsGradMethodStl::pre_processing() {
  try {
    internal_order_test();
    A = std::vector<double>(taskData->inputs_count[0]);
    std::copy(reinterpret_cast<double *>(taskData->inputs[0]),
              reinterpret_cast<double *>(taskData->inputs[0]) + taskData->inputs_count[0], A.begin());
    b = std::vector<double>(taskData->inputs_count[1]);
    std::copy(reinterpret_cast<double *>(taskData->inputs[1]),
              reinterpret_cast<double *>(taskData->inputs[1]) + taskData->inputs_count[1], b.begin());
    rows = *reinterpret_cast<int *>(taskData->inputs[2]);
    x = std::vector<double>(rows, 0.0);
  } catch (...) {
    return false;
  }
  return true;
}

bool SystemsGradMethodStl::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] * taskData->inputs_count[1] &&
         taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool SystemsGradMethodStl::run() {
  try {
    internal_order_test();
    x = SLEgradSolver(A, b, rows);
  } catch (...) {
    return false;
  }
  return true;
}

bool SystemsGradMethodStl::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < x.size(); ++i) {
    reinterpret_cast<double *>(taskData->outputs[0])[i] = x[i];
  }
  return true;
}

bool checkSolution(const std::vector<double> &Aa, const std::vector<double> &bb, const std::vector<double> &xx,
                   double tol) {
  int n = bb.size();
  std::vector<double> Ax(n, 0.0);
  std::vector<std::future<void>> futures;
  for (int i = 0; i < n; ++i) {
    futures.push_back(std::async(std::launch::async, [&Aa, &bb, &xx, &Ax, n, i]() {
      for (int j = 0; j < n; ++j) {
        Ax[i] += Aa[i * n + j] * xx[j];
      }
    }));
  }
  for (auto &f : futures) {
    f.get();
  }
  for (int i = 0; i < n; ++i) {
    if (std::abs(Ax[i] - bb[i]) > tol) {
      return false;
    }
  }
  return true;
}

std::vector<double> genRandomVector(int size, int maxVal) {
  std::vector<double> res(size);
  std::mt19937 gen(4140);
  for (int i = 0; i < size; ++i) {
    res[i] = static_cast<double>(gen() % maxVal + 1);
  }
  return res;
}

std::vector<double> genRandomMatrix(int size, int maxVal) {
  std::vector<double> matrix(size * size);
  std::mt19937 gen(4041);
  for (int i = 0; i < size; ++i) {
    for (int j = i; j < size; ++j) {
      matrix[i * size + j] = static_cast<double>(gen() % maxVal + 1);
    }
  }
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < i; ++j) {
      matrix[i * size + j] = matrix[j * size + i];
    }
  }
  return matrix;
}  // namespace veselov_i_stl
