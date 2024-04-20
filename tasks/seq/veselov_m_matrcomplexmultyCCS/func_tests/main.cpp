// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/veselov_m_matrcomplexmultyCCS/include/ops_seq.hpp"

using namespace VeselovSeq;

TEST(veselov_m_matrcomplexmultyCCS, test_sizes_true) {
  size_t n1 = 4;
  size_t m1 = 6;
  size_t n2 = 6;
  size_t m2 = 2;

  // Create data
  std::vector<Complex> in1(n1 * m1);

  std::vector<Complex> in2(n2 * m2);

  std::vector<Complex> out(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SparseMatrixComplexMultiSequential sparseMatrixComplexMultiSequential(taskDataSeq);
  ASSERT_EQ(sparseMatrixComplexMultiSequential.validation(), true);
}

TEST(veselov_m_matrcomplexmultyCCS, test_sizes_false) {
  size_t n1 = 2;
  size_t m1 = 3;
  size_t n2 = 4;
  size_t m2 = 5;

  // Create data
  std::vector<Complex> in1(n1 * m1);

  std::vector<Complex> in2(n2 * m2);

  std::vector<Complex> out(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SparseMatrixComplexMultiSequential sparseMatrixComplexMultiSequential(taskDataSeq);
  ASSERT_FALSE(sparseMatrixComplexMultiSequential.validation());
}

TEST(veselov_m_matrcomplexmultyCCS, test_multy_correct) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;

  // Create data
  std::vector<Complex> in1{Complex{3, 2}, Complex{0, 0}, Complex{0, 0}, Complex{0, 0}, Complex{1, -3},
                           Complex{0, 0}, Complex{0, 0}, Complex{0, 0}, Complex{-4, 1}};
  std::vector<Complex> in2{Complex{0, 0},  Complex{2, -1}, Complex{0, 0}, Complex{0, 0}, Complex{0, 0},
                           Complex{-5, 2}, Complex{0, 0},  Complex{2, 1}, Complex{0, 0}};
  std::vector<Complex> out(n1 * m2);
  std::vector<Complex> test{Complex{0, 0},  Complex{8, 1}, Complex{0, 0},   Complex{0, 0}, Complex{0, 0},
                            Complex{1, 17}, Complex{0, 0}, Complex{-9, -2}, Complex{0, 0}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SparseMatrixComplexMultiSequential sparseMatrixComplexMultiSequential(taskDataSeq);
  sparseMatrixComplexMultiSequential.validation();
  sparseMatrixComplexMultiSequential.pre_processing();
  sparseMatrixComplexMultiSequential.run();
  sparseMatrixComplexMultiSequential.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].imag == test[i].imag && out[i].real == test[i].real) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);
}

TEST(veselov_m_matrcomplexmultyCCS, inverse_matrix) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;
  // Create data
  std::vector<Complex> in1{Complex{2, -1}, Complex{0, 0}, Complex{0, 0}, Complex{0, 0}, Complex{0, 0},
                           Complex{0, 2},  Complex{0, 0}, Complex{2, 1}, Complex{0, 0}};
  std::vector<Complex> in2{Complex{0.4, 0.2},  Complex{0, 0}, Complex{0, 0},    Complex{0, 0}, Complex{0, 0},
                           Complex{0.4, -0.2}, Complex{0, 0}, Complex{0, -0.5}, Complex{0, 0}};
  std::vector<Complex> out(n1 * m2);
  std::vector<Complex> identity{Complex{1, 0}, Complex{0, 0}, Complex{0, 0}, Complex{0, 0}, Complex{1, 0},
                                Complex{0, 0}, Complex{0, 0}, Complex{0, 0}, Complex{1, 0}};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);
  // Create Task
  SparseMatrixComplexMultiSequential sparseMatrixComplexMultiSequential(taskDataSeq);
  sparseMatrixComplexMultiSequential.validation();
  sparseMatrixComplexMultiSequential.pre_processing();
  sparseMatrixComplexMultiSequential.run();
  sparseMatrixComplexMultiSequential.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].real == identity[i].real && out[i].imag == identity[i].imag) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);
}

TEST(veselov_m_matrcomplexmultyCCS, zero_matrix) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;
  // Create data
  auto Null = Complex{0, 0};
  std::vector<Complex> in1{Complex{7, 1}, Complex{0, 0}, Complex{0, 0}, Complex{2, -6}, Complex{0, 5},
                           Complex{0, 0}, Complex{1, 0}, Complex{0, 0}, Complex{0, 0}};
  std::vector<Complex> in2{Null, Null, Null, Null, Null, Null, Null, Null, Null};
  std::vector<Complex> out(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SparseMatrixComplexMultiSequential sparseMatrixComplexMultiSequential(taskDataSeq);
  sparseMatrixComplexMultiSequential.validation();
  sparseMatrixComplexMultiSequential.pre_processing();
  sparseMatrixComplexMultiSequential.run();
  sparseMatrixComplexMultiSequential.post_processing();

  size_t m = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].imag == 0 && out[i].real == 0) {
      m++;
    }
  }

  ASSERT_EQ(m, n1 * m2);
}