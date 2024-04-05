
#include <gtest/gtest.h>
#include <memory>
#include <complex>
#include <vector>
#include "core/task/include/task.hpp"
#include "seq/savchuk_a_matrix_crs/include/ops_seq.hpp"

TEST(TestTaskSequential, Matrix_Multiplication) {
  // Create input matrices
  std::vector<std::complex<double>> input_matrix_1 = { {1, 2}, {3, 4}, {5, 6} };
  std::vector<std::complex<double>> input_matrix_2 = { {7, 8}, {9, 10}, {11, 12} };

  // Create output matrix
  std::vector<std::complex<double>> output_matrix(3 * 2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_1.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_2.data()));
  taskData->inputs_count = { 3, 2 };
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_matrix.data()));
  taskData->outputs_count = { 3, 2 };

  // Create Task
  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());

  // Run the task
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  // Check the result
  std::vector<std::complex<double>> expected_output_matrix = { {29, 32}, {67, 74}, {105, 116} };
  for (int i = 0; i < 3 * 2; i++) {
    ASSERT_EQ(output_matrix[i], expected_output_matrix[i]);
  }
}