// Copyright 2024 Alexseev Danila
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/alexseev_d_hoare_sort_w_batcher_merge/include/hoare_sort_w_batcher_merge_omp.hpp"

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, test_pipeline_run) {
  std::vector<int> inputArray = generateRandomVector(2000, -1000, 1000);

  // Sequential
  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();

  // OMP
  std::vector<int> outputArrayOMP(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeOMP = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeOMP->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayOMP.data()));
  hoareSortWBatcherMergeOMP->outputs_count.emplace_back(inputArray.size());

  auto hoareSortWBatcherMergeOMPParallel = std::make_shared<HoareSortWBatcherMergeOMP>(hoareSortWBatcherMergeOMP);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(hoareSortWBatcherMergeOMPParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(outputArraySeq, outputArrayOMP);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, test_task_run) {
  std::vector<int> inputArray = generateRandomVector(2000, -1000, 1000);

  // Sequential
  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();

  // OMP
  std::vector<int> outputArrayOMP(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeOMP = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeOMP->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayOMP.data()));
  hoareSortWBatcherMergeOMP->outputs_count.emplace_back(inputArray.size());

  auto hoareSortWBatcherMergeOMPParallel = std::make_shared<HoareSortWBatcherMergeOMP>(hoareSortWBatcherMergeOMP);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(hoareSortWBatcherMergeOMPParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(outputArraySeq, outputArrayOMP);
}
