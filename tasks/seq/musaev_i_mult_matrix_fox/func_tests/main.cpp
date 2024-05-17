// Copyright 2024 Musaev Ilgar
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "seq/musaev_i_mult_matrix_fox/include/ops_seq.hpp"

TEST(Musaev_I_Mult_Matrix_Fox, Validation_Test) {
  size_t n = 2;
  std::vector<double> vect_1{24.0, 45.0, -31.0, 10.0};
  std::vector<double> vect_2{1.0, 0.0, 0.0, 1.0, 2.0};
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_2.data()));
  taskDataSeq->inputs_count.emplace_back(vect_1.size());
  taskDataSeq->inputs_count.emplace_back(vect_2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MusaevTaskSequential musaevTaskSequential(taskDataSeq);
  ASSERT_EQ(musaevTaskSequential.validation(), false);
}

TEST(Musaev_I_Mult_Matrix_Fox, Identity_Mult_On_Another_Matrix) {
  size_t n = 2;
  std::vector<double> vect_1{0.0, 1.0, 1.0, 0.0};
  std::vector<double> vect_2{24.0, 45.0, -31.0, 10.0};
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_2.data()));
  taskDataSeq->inputs_count.emplace_back(vect_1.size());
  taskDataSeq->inputs_count.emplace_back(vect_2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MusaevTaskSequential musaevTaskSequential(taskDataSeq);
  ASSERT_EQ(musaevTaskSequential.validation(), true);
  musaevTaskSequential.pre_processing();
  musaevTaskSequential.run();
  musaevTaskSequential.post_processing();
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_DOUBLE_EQ(out[i * n + j], vect_2[((n - i - 1)) * n + j]);
    }
  }
}

TEST(Musaev_I_Mult_Matrix_Fox, Identity_Mult_On_Matrix) {
  size_t n = 2;
  std::vector<double> vect_1{1.0, 0.0, 0.0, 1.0};
  std::vector<double> vect_2{24.0, 45.0, -31.0, 10.0};
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_2.data()));
  taskDataSeq->inputs_count.emplace_back(vect_1.size());
  taskDataSeq->inputs_count.emplace_back(vect_2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MusaevTaskSequential musaevTaskSequential(taskDataSeq);
  ASSERT_EQ(musaevTaskSequential.validation(), true);
  musaevTaskSequential.pre_processing();
  musaevTaskSequential.run();
  musaevTaskSequential.post_processing();
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_DOUBLE_EQ(out[i], vect_2[i]);
  }
}

TEST(Musaev_I_Mult_Matrix_Fox, Matrix_Mult_On_Identity) {
  size_t n = 2;
  std::vector<double> vect_1{24.0, 45.0, -31.0, 10.0};
  std::vector<double> vect_2{1.0, 0.0, 0.0, 1.0};
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_2.data()));
  taskDataSeq->inputs_count.emplace_back(vect_1.size());
  taskDataSeq->inputs_count.emplace_back(vect_2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MusaevTaskSequential musaevTaskSequential(taskDataSeq);
  ASSERT_EQ(musaevTaskSequential.validation(), true);
  musaevTaskSequential.pre_processing();
  musaevTaskSequential.run();
  musaevTaskSequential.post_processing();
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_DOUBLE_EQ(out[i], vect_1[i]);
  }
}

TEST(Musaev_I_Mult_Matrix_Fox, Matrix_Mult_On_Another_Identity) {
  size_t n = 2;
  std::vector<double> vect_1{24.0, 45.0, -31.0, 10.0};
  std::vector<double> vect_2{0.0, 1.0, 1.0, 0.0};
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect_2.data()));
  taskDataSeq->inputs_count.emplace_back(vect_1.size());
  taskDataSeq->inputs_count.emplace_back(vect_2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MusaevTaskSequential musaevTaskSequential(taskDataSeq);
  ASSERT_EQ(musaevTaskSequential.validation(), true);
  musaevTaskSequential.pre_processing();
  musaevTaskSequential.run();
  musaevTaskSequential.post_processing();
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_DOUBLE_EQ(out[i * n + j], vect_1[i * n + (n - j - 1)]);
    }
  }
}
