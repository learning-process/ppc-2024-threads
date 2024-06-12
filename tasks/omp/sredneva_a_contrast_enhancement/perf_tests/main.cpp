// Copyright 2023 Sredneva Anastasiya
#include <gtest/gtest.h>
#include <omp.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/sredneva_a_contrast_enhancement/include/ops_omp.hpp"

TEST(sredneva_a_contrast_enhancement_omp, test_pipeline_run) {
  int n = 2000;
  int m = 1500;
  uint8_t min = 75;
  uint8_t max = 150;

  // Create data
  std::vector<uint8_t> in = getRandomPicture(n, m, min, max);

  std::vector<int> in2 = {n, m};
  std::vector<uint8_t> in3 = {min, max};
  std::vector<uint8_t> out(n * m);

  std::vector<uint8_t> input(in);
  std::vector<uint8_t> res(n * m, 0);
  for (int i = 0; i < n * m; i++) {
    res[i] = (input[i] - min) * 255 / (max - min);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataSeq->inputs_count.emplace_back(in3.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskOMP = std::make_shared<ContrastEnhancement_OMP_Parallel>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOMP);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(sredneva_a_contrast_enhancement_omp, test_task_run) {
  int n = 2000;
  int m = 1500;
  uint8_t min = 75;
  uint8_t max = 150;

  // Create data
  std::vector<uint8_t> in = getRandomPicture(n, m, min, max);

  std::vector<int> in2 = {n, m};
  std::vector<uint8_t> in3 = {min, max};
  std::vector<uint8_t> out(n * m);

  std::vector<uint8_t> input(in);
  std::vector<uint8_t> res(n * m, 0);
  for (int i = 0; i < n * m; i++) {
    res[i] = (input[i] - min) * 255 / (max - min);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataSeq->inputs_count.emplace_back(in3.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskOMP = std::make_shared<ContrastEnhancement_OMP_Parallel>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOMP);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}
