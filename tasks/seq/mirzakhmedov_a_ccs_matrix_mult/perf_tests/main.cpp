// Copyright 2024 Mirzakhmedov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/mirzakhmedov_a_ccs_matrix_mult/include/ccs_matrix_mult.hpp"

const double PI = 3.14159265358979323846;

CCSSparseMatrix dft_matrix(int n) {
  auto N = (double)n;
  std::complex<double> exponent{0.0, -2.0 * PI / N};
  CCSSparseMatrix dft(n, n, n * n);
  for (int i = 1; i <= n; ++i) {
    dft.columnPointers[i] = i * n;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      dft.rowIndices[i * n + j] = j;
      dft.nonzeroValues[i * n + j] = std::exp(exponent * double(i * j));
    }
  }
  return dft;
}

CCSSparseMatrix dft_conj_matrix(int n) {
  auto N = (double)n;
  std::complex<double> exponent{0.0, 2.0 * PI / N};
  CCSSparseMatrix dft_conj(n, n, n * n);
  for (int i = 1; i <= n; ++i) {
    dft_conj.columnPointers[i] = i * n;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      dft_conj.rowIndices[i * n + j] = j;
      dft_conj.nonzeroValues[i * n + j] = std::exp(exponent * double(j * i));
    }
  }
  return dft_conj;
}

TEST(mirzakhmedov_a_ccs_matrix_mult, test_pipeline_run_dft384x384) {
  int n = 384;
  CCSSparseMatrix A = dft_matrix(n);
  CCSSparseMatrix B = dft_conj_matrix(n);
  CCSSparseMatrix C;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  auto testTaskSequential = std::make_shared<CSeq>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(mirzakhmedov_a_ccs_matrix_mult, test_task_run_dft384x384) {
  int n = 384;
  CCSSparseMatrix A = dft_matrix(n);
  CCSSparseMatrix B = dft_conj_matrix(n);
  CCSSparseMatrix C;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  auto testTaskSequential = std::make_shared<CSeq>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}