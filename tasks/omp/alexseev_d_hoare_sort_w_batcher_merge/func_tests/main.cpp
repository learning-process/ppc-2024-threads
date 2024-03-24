// Copyright 2024 Alexseev Danila
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "omp/alexseev_d_hoare_sort_w_batcher_merge/include/hoare_sort_w_batcher_merge_omp.hpp"

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Random_Constatnt_Array) {
  std::vector<int> inputArray = {3, 5, 4, 6, 9, 2, 8, 1, 7, 0};

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

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Random_Array_Even) {
  std::vector<int> inputArray = generateRandomVector(50, -1000, 1000);

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

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Random_Array_Odd) {
  std::vector<int> inputArray = generateRandomVector(51, -1000, 1000);

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

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Unit_Array) {
  std::vector<int> inputArray = generateRandomVector(1, -1000, 1000);

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

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Empty_Array) {
  std::vector<int> inputArray = {};

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

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}