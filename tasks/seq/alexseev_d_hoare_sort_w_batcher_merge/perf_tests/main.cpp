// Copyright 2024 Alexseev Danila
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/alexseev_d_hoare_sort_w_batcher_merge/include/hoare_sort_w_batcher_merge_seq.hpp"

TEST(alexseev_d_hoare_sort_w_batcher_merge_seq, test_pipeline_run) {
  std::vector<int> inputArray = generateRandomVector(2000, -1000, 1000);
  std::vector<int> sortedInputArray = inputArray;
  std::sort(sortedInputArray.begin(), sortedInputArray.end());
  std::vector<int> outputArray(inputArray.size());
  std::vector<int> expectedArray = sortedInputArray;

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArray.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  auto hoareSortWBatcherMergeSequential = std::make_shared<HoareSortWBatcherMergeSequential>(hoareSortWBatcherMergeSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(hoareSortWBatcherMergeSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expectedArray, outputArray);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_seq, test_task_run) {
  std::vector<int> inputArray = generateRandomVector(2000, -1000, 1000);
  std::vector<int> sortedInputArray = inputArray;
  std::sort(sortedInputArray.begin(), sortedInputArray.end());
  std::vector<int> outputArray(inputArray.size());
  std::vector<int> expectedArray = sortedInputArray;

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArray.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  auto hoareSortWBatcherMergeSequential = std::make_shared<HoareSortWBatcherMergeSequential>(hoareSortWBatcherMergeSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(hoareSortWBatcherMergeSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expectedArray, outputArray);
}
