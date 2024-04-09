// Copyright 2024 Borovkov Sergey
#include <gtest/gtest.h>

#include <vector>

#include "seq/borovkov_s_cannon/include/ops_seq.hpp"

// Test case for matrix multiplication with size 2x2
TEST(borovkov_s_cannon_seq, MatrixMultiplication_2x2) {
  int rows = 2;
  int cols = 2;

  std::vector<double> matrix_A{1, 2, 3, 4};
  std::vector<double> matrix_B{6, 7, 8, 9};
  std::vector<double> result(rows * cols);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskData->inputs_count.emplace_back(matrix_A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskData->inputs_count.emplace_back(matrix_B.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());

  std::vector<double> reference_result = multiplyMatrix(matrix_A, matrix_B, rows, cols);

  TestTaskSequentialBorovkovCannon testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  for (size_t i = 0; i < reference_result.size(); ++i) {
    ASSERT_EQ(reference_result[i], result[i]);
  }
}

// Test case for matrix multiplication with size 3x3
TEST(borovkov_s_cannon_seq, MatrixMultiplication_3x3) {
  int rows = 3;
  int cols = 3;

  // Define input matrices
  std::vector<double> matrix_A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> matrix_B{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> result(rows * cols);

  // Prepare TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskData->inputs_count.emplace_back(matrix_A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskData->inputs_count.emplace_back(matrix_B.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());

  // Calculate reference result using original function
  std::vector<double> reference_result = multiplyMatrix(matrix_A, matrix_B, rows, cols);

  // Create Task and execute
  TestTaskSequentialBorovkovCannon testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  // Validate result
  for (size_t i = 0; i < reference_result.size(); ++i) {
    ASSERT_EQ(reference_result[i], result[i]);
  }
}

// Test case for matrix multiplication with size 5x5
TEST(borovkov_s_cannon_seq, MatrixMultiplication_5x5) {
  int rows = 5;
  int cols = 5;

  std::vector<double> matrix_A = getRandomMatrix(rows, cols);
  std::vector<double> matrix_B = getRandomMatrix(rows, cols);
  std::vector<double> result(rows * cols);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskData->inputs_count.emplace_back(matrix_A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskData->inputs_count.emplace_back(matrix_B.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());

  std::vector<double> reference_result = multiplyMatrix(matrix_A, matrix_B, rows, cols);

  TestTaskSequentialBorovkovCannon testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  for (size_t i = 0; i < reference_result.size(); ++i) {
    ASSERT_EQ(reference_result[i], result[i]);
  }
}

// Test case for matrix multiplication with size 0x0
TEST(borovkov_s_cannon_seq, MatrixMultiplication_0x0) {
  int rows = 0;
  int cols = 0;

  std::vector<double> matrix_A = getRandomMatrix(rows, cols);
  std::vector<double> matrix_B = getRandomMatrix(rows, cols);
  std::vector<double> result(rows * cols);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskData->inputs_count.emplace_back(matrix_A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskData->inputs_count.emplace_back(matrix_B.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());

  std::vector<double> reference_result = multiplyMatrix(matrix_A, matrix_B, rows, cols);

  TestTaskSequentialBorovkovCannon testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  for (size_t i = 0; i < reference_result.size(); ++i) {
    ASSERT_EQ(reference_result[i], result[i]);
  }
}

// Test case for matrix multiplication with size 100x100
TEST(borovkov_s_cannon_seq, MatrixMultiplication_100x100) {
  int rows = 100;
  int cols = 100;

  std::vector<double> matrix_A = getRandomMatrix(rows, cols);
  std::vector<double> matrix_B = getRandomMatrix(rows, cols);
  std::vector<double> result(rows * cols);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskData->inputs_count.emplace_back(matrix_A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskData->inputs_count.emplace_back(matrix_B.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());

  std::vector<double> reference_result = multiplyMatrix(matrix_A, matrix_B, rows, cols);

  TestTaskSequentialBorovkovCannon testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  for (size_t i = 0; i < reference_result.size(); ++i) {
    ASSERT_EQ(reference_result[i], result[i]);
  }
}
