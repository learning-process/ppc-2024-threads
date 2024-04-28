// Copyright 2024 Smirnov Leonid
#include <gtest/gtest.h>

#include <vector>

#include "omp/smirnov_l_radix_sort_omp/include/ops_omp.hpp"


TEST(Smirnov_L_Radix_Sort_Test_OMP, Validation_One_Test_OMP_Sm) {
  // Create data
  std::vector<int> in;
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq
  = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortOMPParallel radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test_OMP, Validation_Two_Test_OMP_Sm) {
  // Create data
  std::vector<int> in = {9, 6, 3};
  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq
  = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortOMPParallel radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}


TEST(Smirnov_L_Radix_Sort_Test_OMP, Validation_Three_Test_OMP_Sm) {
  // Create data
  std::vector<int> in1 = {1, 2, 3};
  std::vector<int> in2 = {3, 2, 1};
  std::vector<int> out(in1.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq
  = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortOMPParallel radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}


TEST(Smirnov_L_Radix_Sort_Test_OMP, Validation_Four_Test_OMP_Sm) {
  // Create data
  std::vector<int> in = {9, 6, 3};
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq
  = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(10);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortOMPParallel radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test_OMP, Validation_Five_Test_OMP_Sm) {
  // Create data
  std::vector<int> in = {9, 6, 3};
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq
  = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs_count.emplace_back(10);

  // Create Task
  RadixSortOMPParallel radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test_OMP, Validation_Six_Test) {
  // Create data
  std::vector<int> in = {1, 2, 3};
  std::vector<int> out1(in.size(), 0);
  std::vector<int> out2(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq
  = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
  taskDataSeq->outputs_count.emplace_back(out1.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataSeq->outputs_count.emplace_back(out2.size());

  // Create Task
  RadixSortOMPParallel radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test_OMP, Validation_Seven_Test) {
  // Create data
  std::vector<int> in = {1, 2, 3};
  std::vector<int> out(4, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq
  = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortOMPParallel radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test_OMP, Test_Cor_Work_Real_One_OMP_Sm) {
  std::vector<int> in = {6, 3, 1, 30, 50, 113, 5, 17, 76, 21};
  std::vector<int> expected = {1, 3, 5, 6, 17, 21, 30, 50, 76, 113};

  // Create data
  std::vector<int> out2(expected.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar
  = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataPar->outputs_count.emplace_back(out2.size());

  // Create Task
  RadixSortOMPParallel radixSortOMPParallel(taskDataPar);
  ASSERT_EQ(radixSortOMPParallel.validation(), true);
  ASSERT_TRUE(radixSortOMPParallel.pre_processing());
  ASSERT_TRUE(radixSortOMPParallel.run());
  ASSERT_TRUE(radixSortOMPParallel.post_processing());

  for (size_t i = 0; i < out2.size(); i++) {
    ASSERT_EQ(out2[i], expected[i]);
  }
}

TEST(Smirnov_L_Radix_Sort_Test_OMP, Test_Cor_Work_Real_Two_OMP_Sm) {
  std::vector<int> in = {7, 32, 1, -30, 49, 5, 17, 76, -5};
  std::vector<int> expected = {-30, -5, 1, 5, 7, 17, 32, 49, 76};

  // Create data
  std::vector<int> out2(expected.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar
  = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataPar->outputs_count.emplace_back(out2.size());

  // Create Task
  RadixSortOMPParallel radixSortOMPParallel(taskDataPar);
  ASSERT_EQ(radixSortOMPParallel.validation(), true);
  ASSERT_TRUE(radixSortOMPParallel.pre_processing());
  ASSERT_TRUE(radixSortOMPParallel.run());
  ASSERT_TRUE(radixSortOMPParallel.post_processing());

  for (size_t i = 0; i < out2.size(); i++) {
    ASSERT_EQ(out2[i], expected[i]);
  }
}

TEST(Smirnov_L_Radix_Sort_Test_OMP, Test_Cor_Work_Real_Three_OMP_Sm) {
  std::vector<int> in = {1, 5, 10, -1, -5, -10, 0};
  std::vector<int> expected = {-10, -5, -1, 0, 1, 5, 10};

  // Create data
  std::vector<int> out2(expected.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar
  = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataPar->outputs_count.emplace_back(out2.size());

  // Create Task
  RadixSortOMPParallel radixSortOMPParallel(taskDataPar);
  ASSERT_EQ(radixSortOMPParallel.validation(), true);
  ASSERT_TRUE(radixSortOMPParallel.pre_processing());
  ASSERT_TRUE(radixSortOMPParallel.run());
  ASSERT_TRUE(radixSortOMPParallel.post_processing());

  for (size_t i = 0; i < out2.size(); i++) {
    ASSERT_EQ(out2[i], expected[i]);
  }
}

TEST(Smirnov_L_Radix_Sort_Test_OMP, Test_Cor_Work_Real_Four_OMP_Sm) {
  std::vector<int> in = {9, 3, 6};
  std::vector<int> expected = {3, 6, 9};

  // Create data
  std::vector<int> out2(expected.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar
  = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataPar->outputs_count.emplace_back(out2.size());

  // Create Task
  RadixSortOMPParallel radixSortOMPParallel(taskDataPar);
  ASSERT_EQ(radixSortOMPParallel.validation(), true);
  ASSERT_TRUE(radixSortOMPParallel.pre_processing());
  ASSERT_TRUE(radixSortOMPParallel.run());
  ASSERT_TRUE(radixSortOMPParallel.post_processing());

  for (size_t i = 0; i < out2.size(); i++) {
    ASSERT_EQ(out2[i], expected[i]);
  }
}

TEST(Smirnov_L_Radix_Sort_Test_OMP, Test_Cor_Work_Real_Five_OMP_Sm) {
  std::vector<int> in = {-14, -10, -50, 50, 90, 54, 3, 14,
    19, 100, -54, 134, 363, 0, 79};
  std::vector<int> expected = {-54, -50, -14, -10, 0, 3, 14,
    19, 50, 54, 79, 90, 100, 134, 363};

  // Create data
  std::vector<int> out2(expected.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar
  = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataPar->outputs_count.emplace_back(out2.size());

  // Create Task
  RadixSortOMPParallel radixSortOMPParallel(taskDataPar);
  ASSERT_EQ(radixSortOMPParallel.validation(), true);
  ASSERT_TRUE(radixSortOMPParallel.pre_processing());
  ASSERT_TRUE(radixSortOMPParallel.run());
  ASSERT_TRUE(radixSortOMPParallel.post_processing());

  for (size_t i = 0; i < out2.size(); i++) {
    ASSERT_EQ(out2[i], expected[i]);
  }
}
