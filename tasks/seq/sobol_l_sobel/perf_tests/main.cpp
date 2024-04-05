// Copyright 2023 Sobol Liubov
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sobol_l_sobel/include/sobel_seq.hpp"
#include "seq/sobol_l_sobel/src/sobel_seq.cpp"

TEST(Sequential_sobol_sobel_perf_test, test_pipeline_run) {
  const uint8_t min = 0;
  const uint8_t max = 255;
  const int w = 100;
  const int h = 100;

  // Create data
  std::vector<int> in(2);
  in[0] = w;
  in[1] = h;
  std::vector<uint8_t> picture = getRandomPicture(w, h, min, max);
  std::vector<uint8_t> out(w * h, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(picture.data());
  taskDataSeq->inputs_count.emplace_back(picture.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<Sobel_seq>(taskDataSeq);

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
  for (int i = 1; i < w - 1; i++) {
    for (int j = 1; j < h - 1; j++) {
      int x = -picture[(i - 1) * h + j - 1] + picture[(i - 1) * h + j + 1] - 2 * picture[i * h + j - 1] +
              2 * picture[i * h + j + 1] - picture[(i + 1) * h + j - 1] + picture[(i + 1) * h + j + 1];
      int y = picture[(i - 1) * h + j - 1] + 2 * picture[(i - 1) * h + j] + picture[(i - 1) * h + j + 1] -
              picture[(i + 1) * h + j - 1] - 2 * picture[(i + 1) * h + j] - picture[(i + 1) * h + j + 1];
      uint8_t expected = sqrt(x * x + y * y);
      ASSERT_EQ(expected, out[i * h + j]);
    }
  }
}

TEST(Sequential_sobol_sobel_perf_test, test_task_run) {
  const uint8_t min = 0;
  const uint8_t max = 255;
  const int w = 100;
  const int h = 100;

  // Create data
  std::vector<int> in(2);
  in[0] = w;
  in[1] = h;
  std::vector<uint8_t> picture = getRandomPicture(w, h, min, max);
  std::vector<uint8_t> out(w * h, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(picture.data());
  taskDataSeq->inputs_count.emplace_back(picture.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<Sobel_seq>(taskDataSeq);

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
  for (int i = 1; i < w - 1; i++) {
    for (int j = 1; j < h - 1; j++) {
      int x = -picture[(i - 1) * h + j - 1] + picture[(i - 1) * h + j + 1] - 2 * picture[i * h + j - 1] +
              2 * picture[i * h + j + 1] - picture[(i + 1) * h + j - 1] + picture[(i + 1) * h + j + 1];
      int y = picture[(i - 1) * h + j - 1] + 2 * picture[(i - 1) * h + j] + picture[(i - 1) * h + j + 1] -
              picture[(i + 1) * h + j - 1] - 2 * picture[(i + 1) * h + j] - picture[(i + 1) * h + j + 1];
      uint8_t expected = sqrt(x * x + y * y);
      ASSERT_EQ(expected, out[i * h + j]);
    }
  }
}
