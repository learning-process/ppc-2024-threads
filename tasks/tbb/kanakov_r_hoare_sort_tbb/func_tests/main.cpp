// Copyright 2024 Kanakov Roman
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "tbb/kanakov_r_hoare_sort_tbb/include/kanakov_hoare_sort_tbb.hpp"

using namespace kanakov_tbb;

TEST(kanakov_r_hoare_sort_tbb, Const_Array) {
  std::vector<int> inputArray = {6, 3, 2, 0, 5, 1, 9, 7, 4, 8};

  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortSequential hoareSortSequential(hoareSortSeq);
  ASSERT_TRUE(hoareSortSequential.validation());
  hoareSortSequential.pre_processing();
  hoareSortSequential.run();
  hoareSortSequential.post_processing();

  std::vector<int> outputArrayTBB(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortTBB = std::make_shared<ppc::core::TaskData>();
  hoareSortTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortTBB->inputs_count.emplace_back(inputArray.size());
  hoareSortTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayTBB.data()));
  hoareSortTBB->outputs_count.emplace_back(inputArray.size());

  HoareSortTBB hoareSortTBBParallel(hoareSortTBB);
  ASSERT_TRUE(hoareSortTBBParallel.validation());
  hoareSortTBBParallel.pre_processing();
  hoareSortTBBParallel.run();
  hoareSortTBBParallel.post_processing();

  ASSERT_EQ(outputArrayTBB, outputArraySeq);
}

TEST(kanakov_r_hoare_sort_tbb, Random_Array) {
  std::vector<int> inputArray = {634,  -434, -842, 883,  -799, -64,  574,  733,  -406, -823, 563,  -901, 807,
                                 292,  6,    690,  247,  138,  -573, -945, 275,  -887, 499,  -803, -449, 298,
                                 -172, -762, 679,  222,  360,  412,  -882, -652, 250,  230,  -111, 215,  846,
                                 -637, -780, 524,  -599, 402,  -103, -118, 73,   840,  -56,  -505};

  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortSequential hoareSortSequential(hoareSortSeq);
  ASSERT_TRUE(hoareSortSequential.validation());
  hoareSortSequential.pre_processing();
  hoareSortSequential.run();
  hoareSortSequential.post_processing();

  std::vector<int> outputArrayTBB(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortTBB = std::make_shared<ppc::core::TaskData>();
  hoareSortTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortTBB->inputs_count.emplace_back(inputArray.size());
  hoareSortTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayTBB.data()));
  hoareSortTBB->outputs_count.emplace_back(inputArray.size());

  HoareSortTBB hoareSortTBBParallel(hoareSortTBB);
  ASSERT_TRUE(hoareSortTBBParallel.validation());
  hoareSortTBBParallel.pre_processing();
  hoareSortTBBParallel.run();
  hoareSortTBBParallel.post_processing();

  ASSERT_EQ(outputArrayTBB, outputArraySeq);
}
TEST(kanakov_r_hoare_sort_tbb, Random_Negative_Array) {
  std::vector<int> inputArray = {-12,  -575, -13,  -898, -690, -625, -444, -699, -929, -907, -146, -333, -357,
                                 -265, -961, -258, -768, -225, -506, -262, -645, -366, -843, -46,  -197, -30,
                                 -728, -75,  -641, -982, -287, -725, -793, -882, -153, -264, -247, -43,  -416,
                                 -340, -324, -801, -971, -944, -185, -275, -780, -414, -34,  -887, -264, -199,
                                 -424, -974, -849, -172, -332, -41,  -413, -526, -635, -894, -989, -13};

  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortSequential hoareSortSequential(hoareSortSeq);
  ASSERT_TRUE(hoareSortSequential.validation());
  hoareSortSequential.pre_processing();
  hoareSortSequential.run();
  hoareSortSequential.post_processing();

  std::vector<int> outputArrayTBB(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortTBB = std::make_shared<ppc::core::TaskData>();
  hoareSortTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortTBB->inputs_count.emplace_back(inputArray.size());
  hoareSortTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayTBB.data()));
  hoareSortTBB->outputs_count.emplace_back(inputArray.size());

  HoareSortTBB hoareSortTBBParallel(hoareSortTBB);
  ASSERT_TRUE(hoareSortTBBParallel.validation());
  hoareSortTBBParallel.pre_processing();
  hoareSortTBBParallel.run();
  hoareSortTBBParallel.post_processing();

  ASSERT_EQ(outputArrayTBB, outputArraySeq);
}
TEST(kanakov_r_hoare_sort_tbb, Single_Array) {
  std::vector<int> inputArray = {634};

  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortSequential hoareSortSequential(hoareSortSeq);
  ASSERT_TRUE(hoareSortSequential.validation());
  hoareSortSequential.pre_processing();
  hoareSortSequential.run();
  hoareSortSequential.post_processing();

  std::vector<int> outputArrayTBB(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortTBB = std::make_shared<ppc::core::TaskData>();
  hoareSortTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortTBB->inputs_count.emplace_back(inputArray.size());
  hoareSortTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayTBB.data()));
  hoareSortTBB->outputs_count.emplace_back(inputArray.size());

  HoareSortTBB hoareSortTBBParallel(hoareSortTBB);
  ASSERT_TRUE(hoareSortTBBParallel.validation());
  hoareSortTBBParallel.pre_processing();
  hoareSortTBBParallel.run();
  hoareSortTBBParallel.post_processing();

  ASSERT_EQ(outputArrayTBB, outputArraySeq);
}
TEST(kanakov_r_hoare_sort_tbb, Empty_Array) {
  std::vector<int> inputArray = {};

  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortSequential hoareSortSequential(hoareSortSeq);
  ASSERT_TRUE(hoareSortSequential.validation());
  hoareSortSequential.pre_processing();
  hoareSortSequential.run();
  hoareSortSequential.post_processing();

  std::vector<int> outputArrayTBB(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortTBB = std::make_shared<ppc::core::TaskData>();
  hoareSortTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortTBB->inputs_count.emplace_back(inputArray.size());
  hoareSortTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayTBB.data()));
  hoareSortTBB->outputs_count.emplace_back(inputArray.size());

  HoareSortTBB hoareSortTBBParallel(hoareSortTBB);
  ASSERT_TRUE(hoareSortTBBParallel.validation());
  hoareSortTBBParallel.pre_processing();
  hoareSortTBBParallel.run();
  hoareSortTBBParallel.post_processing();

  ASSERT_EQ(outputArrayTBB, outputArraySeq);
}
