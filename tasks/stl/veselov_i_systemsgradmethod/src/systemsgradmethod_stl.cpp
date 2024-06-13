#include "seq/veselov_i_systemsgradmethod/include/systemsgradmethod_seq.hpp"

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
    return std::transform_reduce(std::execution::par, aa.begin(), aa.end(), bb.begin(), 0.0);
}

std::vector<double> matrixVectorProduct(const std::vector<double> &Aa, const std::vector<double> &xx, int n) {
    std::vector<double> result(n, 0.0);
    std::vector<std::future<void>> futures;
    for (int i = 0; i < n; ++i) {
        futures.push_back(std::async(std::launch::async, [&Aa, &xx, &result, n, i]() {
            for (int j = 0; j < n; ++j) {
                result[i] += Aa[i * n + j] * xx[j];
            }
        }));
    }
    for (auto &f : futures) {
        f.get();
    }
    return result;
}

std::vector<double> SLEgradSolver(const std::vector<double> &Aa, const std::vector<double> &bb, int n, double tol = 1e-6) {
    std::vector<double> res(n, 0.0);
    std::vector<double> r = bb;
    std::vector<double> p = r;
    std::vector<double> r_old = bb;

    while (true) {
        std::vector<double> Ap = matrixVectorProduct(Aa, p, n);
        double alpha = dotProduct(r, r) / dotProduct(Ap, p);

        std::transform(std::execution::par, res.begin(), res.end(), p.begin(), res.begin(),
                       [alpha](double res_i, double p_i) { return res_i + alpha * p_i; });

        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < r.size(); ++i) {
            futures.push_back(std::async(std::launch::async, [&r, &r_old, &Ap, alpha, i]() {
                r[i] = r_old[i] - alpha * Ap[i];
            }));
        }
        for (auto &f : futures) {
            f.get();
        }

        if (sqrt(dotProduct(r, r)) < tol) {
            break;
        }
        double beta = dotProduct(r, r) / dotProduct(r_old, r_old);

        std::transform(std::execution::par, r.begin(), r.end(), p.begin(), p.begin(),
                       [beta](double r_i, double p_i) { return r_i + beta * p_i; });

        r_old = r;
    }
    return res;
}

bool SystemsGradMethodSeq::pre_processing() {
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

bool SystemsGradMethodSeq::validation() {
    internal_order_test();
    return taskData->inputs_count[0] == taskData->inputs_count[1] * taskData->inputs_count[1] &&
           taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool SystemsGradMethodSeq::run() {
    try {
        internal_order_test();
        x = SLEgradSolver(A, b, rows);
    } catch (...) {
        return false;
    }
    return true;
}

bool SystemsGradMethodSeq::post_processing() {
    internal_order_test();
    for (size_t i = 0; i < x.size(); ++i) {
        reinterpret_cast<double *>(taskData->outputs[0])[i] = x[i];
    }
    return true;
}

bool checkSolution(const std::vector<double> &Aa, const std::vector<double> &bb, const std::vector<double> &xx, double tol) {
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
}
