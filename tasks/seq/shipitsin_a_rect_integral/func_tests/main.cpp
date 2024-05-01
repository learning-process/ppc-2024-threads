// Copyright 2024 Shipitsin Alex
#include <gtest/gtest.h>

#include <vector>

#include "seq/shipitsin_a_rect_integral/include/ops_seq.hpp"

TEST(shipitsin_a_rect_integral_seq, Test_x_sub_y_1) {
  double res = 0.0;
  Func f = shipitsin_flin;
  uint64_t n = 100;

  std::vector<double> in = {0, 3, 0, 3};
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  ShipitsinTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);
  ASSERT_LT(std::abs(res - out[0]), 0.5);
}

TEST(shipitsin_a_rect_integral_seq, Test_x_sub_y_2) {
  double res = -357.0;
  Func f = shipitsin_flin;
  uint64_t n = 100;

  std::vector<double> in = {-5, 2, 3, 10};
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  ShipitsinTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);
  ASSERT_LT(std::abs(res - out[0]), 0.5);
}

TEST(shipitsin_a_rect_integral_seq, Test_x_mul_y) {
  double res = 18.0;
  Func f = shipitsin_fxy;
  uint64_t n = 100;

  std::vector<double> in = {0, 2, 1, 2};
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  ShipitsinTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);
  ASSERT_LT(std::abs(res - out[0]), 0.5);
}

TEST(shipitsin_a_rect_integral_seq, Test_y_sin_x) {
  double res = 2.295;
  Func f = shipitsin_fysinx;
  uint64_t n = 100;

  std::vector<double> in = {-2, 2, 1, 2};
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  ShipitsinTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);
  ASSERT_LT(std::abs(res - out[0]), 0.5);
}

TEST(shipitsin_a_rect_integral_seq, Test_x_exp_y) {
  double res = 217.0907;
  Func f = shipitsin_fxexpy;
  uint64_t n = 1000;

  std::vector<double> in = {0, 5, 1, 3};
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  ShipitsinTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);
  ASSERT_LT(std::abs(res - out[0]), 0.5);
}
