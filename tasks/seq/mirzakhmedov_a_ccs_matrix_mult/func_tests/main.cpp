// Copyright 2024 Mirzakhmedov Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "seq/mirzakhmedov_a_ccs_matrix_mult/include/ccs_matrix_mult.hpp"

const double PI = 3.14159265358979323846;

CCSSparseMatrix DFTMatrix(int n, bool conjugate = false) {
  auto N = static_cast<double>(n);
  std::complex<double> exponent{0.0, (conjugate ? 2.0 : -2.0) * PI / N};
  CCSSparseMatrix dft(n, n, n * n);
  for (int i = 1; i <= n; ++i) {
    dft.columnPointers[i] = i * n;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      dft.rowIndices[i * n + j] = j;
      dft.nonzeroValues[i * n + j] = std::exp(exponent * static_cast<double>(i * j));
    }
  }
  return dft;
}

TEST(MirzakhmedovACCSMatrixMult, TestScalarMatrix) {
  CCSSparseMatrix A(1, 1, 1);
  CCSSparseMatrix B(1, 1, 1);
  CCSSparseMatrix C;
  A.columnPointers = {0, 1};
  A.rowIndices = {0};
  A.nonzeroValues = {std::complex<double>(0.0, 1.0)};
  B.columnPointers = {0, 1};
  B.rowIndices = {0};
  B.nonzeroValues = {std::complex<double>(0.0, -1.0)};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));
  CSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::complex<double> answer(1.0, 0.0);
  ASSERT_NEAR(std::abs(C.nonzeroValues[0] - answer), 0.0, 1e-6);
}

TEST(MirzakhmedovACCSMatrixMult, TestDFT) {
  struct TestConfig {
    int size;
    bool conjugate;
  };

  std::vector<TestConfig> testCases = {{2, false}, {16, true}, {64, false}, {256, false}, {257, false}};

  for (const auto& config : testCases) {
    CCSSparseMatrix A = DFTMatrix(config.size);
    CCSSparseMatrix B = DFTMatrix(config.size, config.conjugate);
    CCSSparseMatrix C;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

    CSeq testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    int n = config.size;
    double N = static_cast<double>(n);
    std::vector<std::complex<double>> expected_values(n * n, {0.0, 0.0});
    if (config.conjugate) {
      for (int i = 0; i < n; ++i) {
        expected_values[i * n + i] = {N, 0.0};
      }
    } else {
      for (int i = 0; i < n; ++i) {
        expected_values[i * n + i] = {N, 0.0};
      }
    }

    ASSERT_EQ(C.nonzeroValues.size(), expected_values.size());

    for (size_t i = 0; i < C.nonzeroValues.size(); ++i) {
      if (C.nonzeroValues[i] != std::complex<double>(0.0, 0.0)) {
        ASSERT_NEAR(std::abs(C.nonzeroValues[i] - expected_values[i]), 0.0, 1e-6);
      }
    }
  }
}

TEST(MirzakhmedovACCSMatrixMult, TestShiftingDiagonal) {
  int n = 256;
  CCSSparseMatrix A(n, n, n - 1);
  CCSSparseMatrix C;
  for (int i = 0; i < n - 1; ++i) {
    A.columnPointers[i + 1] = i;
    A.rowIndices[i] = i;
    A.nonzeroValues[i] = {1.0, 0.0};
  }

  A.columnPointers[n] = n - 1;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  CSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(C.columnPointers[n], n - 2);
  EXPECT_EQ(C.columnPointers[0], 0);
  EXPECT_EQ(C.columnPointers[1], 0);
  for (int i = 0; i < n - 2; ++i) {
    EXPECT_EQ(C.columnPointers[i + 2], i);
    EXPECT_EQ(C.rowIndices[i], i);
    EXPECT_NEAR(std::abs(C.nonzeroValues[i] - std::complex<double>(1.0, 0.0)), 0.0, 1e-6);
  }
}

TEST(MirzakhmedovACCSMatrixMult, TestPermutationMatrix) {
  int n = 257;
  CCSSparseMatrix A(n, n, n);
  CCSSparseMatrix B(n, n, n);
  CCSSparseMatrix C;
  int pos = 3;
  for (int i = 0; i < n; ++i) {
    A.columnPointers[i] = B.columnPointers[i] = i;
    A.nonzeroValues[i] = B.nonzeroValues[i] = {1.0, 0.0};
    A.rowIndices[i] = pos;
    pos = (pos * 3) % n;
  }
  A.rowIndices[n - 1] = 0;
  A.columnPointers[n] = B.columnPointers[n] = n;
  for (int i = 0; i < n; ++i) {
    B.rowIndices[A.rowIndices[i]] = i;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  CSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(C.columnPointers[i], i);
    EXPECT_EQ(C.rowIndices[i], i);
    EXPECT_NEAR(std::abs(C.nonzeroValues[i] - std::complex<double>(1.0, 0.0)), 0.0, 1e-6);
  }
  EXPECT_EQ(C.columnPointers[n], n);
}
