// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kuznetsov_a_cannon_matr_mult/include/ops_seq.hpp"

TEST(Kuznetsov_a_cannon_matr_mult_seq_perf_tests, test_500x500) {
  // Create data
  size_t size = 500;
  size_t block = 250;

  double minVal = -100.0;
  double maxVal = 100.0;

  auto inputMatrOne = getRandomSquareMatrix(size, minVal, maxVal);
  auto inputMatrTwo = getRandomSquareMatrix(size, minVal, maxVal);
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataSeq->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = multMatrSquare(inputMatrOne, inputMatrTwo, size);

  // Create Task
  auto testTaskSequential = std::make_shared<KuznetsovCannonMatrMultSeq>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(isEqual(resSeq[i], outputMatr[i]));
  }
}

TEST(Kuznetsov_a_cannon_matr_mult_seq_perf_tests, test_600x600) {
  // Create data
  size_t size = 600;
  size_t block = 300;

  double minVal = -100.0;
  double maxVal = 100.0;

  auto inputMatrOne = getRandomSquareMatrix(size, minVal, maxVal);
  auto inputMatrTwo = getRandomSquareMatrix(size, minVal, maxVal);
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataSeq->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = multMatrSquare(inputMatrOne, inputMatrTwo, size);

  // Create Task
  auto testTaskSequential = std::make_shared<KuznetsovCannonMatrMultSeq>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(isEqual(resSeq[i], outputMatr[i]));
  }
}
