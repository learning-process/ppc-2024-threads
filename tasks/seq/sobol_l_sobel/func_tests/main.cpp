// Copyright 2023 Sobol Liubov
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/sobol_l_sobel/include/sobel_seq.hpp"

TEST(sobol_a_sobel_seq, Test_One_Pixel) {
  const int w = 1;
  const int h = 1;
  const uint8_t min = 0;
  const uint8_t max = 255;

  std::vector<int> in(2);
  in[0] = w;
  in[1] = h;
  std::vector<uint8_t> picture = getRandomPicture(w, h, min, max);
  std::vector<uint8_t> out(w * h, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(picture.data());
  taskDataSeq->inputs_count.emplace_back(picture.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  Sobel_seq Sobel_Sequential(taskDataSeq);
  ASSERT_EQ(Sobel_Sequential.validation(), true);
  Sobel_Sequential.pre_processing();
  Sobel_Sequential.run();
  Sobel_Sequential.post_processing();
  ASSERT_EQ(picture[0], out[0]);
}

TEST(sobol_a_sobel_seq, Test_Sobel) {
  const int w = 5;
  const int h = 5;
  const uint8_t min = 0;
  const uint8_t max = 255;

  std::vector<int> in(2);
  in[0] = w;
  in[1] = h;
  std::vector<uint8_t> picture = getRandomPicture(w, h, min, max);
  std::vector<uint8_t> out(w * h, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(picture.data());
  taskDataSeq->inputs_count.emplace_back(picture.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  Sobel_seq Sobel_Sequential(taskDataSeq);
  ASSERT_EQ(Sobel_Sequential.validation(), true);
  Sobel_Sequential.pre_processing();
  Sobel_Sequential.run();
  Sobel_Sequential.post_processing();

  for (int i = 1; i < w - 1; i++) {
    for (int j = 1; j < h - 1; j++) {
      int x = -picture[(i - 1) * h + j - 1] + picture[(i - 1) * h + j + 1] -
               2 * picture[i * h + j - 1] + 2 * picture[i * h + j + 1] - 
               picture[(i + 1) * h + j - 1] + picture[(i + 1) * h + j + 1];
      int y = picture[(i - 1) * h + j - 1] + 2 * picture[(i - 1) * h + j] +
               picture[(i - 1) * h + j + 1] - picture[(i + 1) * h + j - 1] - 
               2 * picture[(i + 1) * h + j] - picture[(i + 1) * h + j + 1];
      uint8_t expected = sqrt(x * x + y * y);
      ASSERT_EQ(expected, out[i * h + j]);
    }
  }
}

TEST(sobol_a_sobel_seq, Test_Picture) {
  const int w = 0;
  const int h = 0;

  std::vector<int> in(2);
  in[0] = w;
  in[1] = h;
  std::vector<uint8_t> picture;
  std::vector<uint8_t> out;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(picture.data());
  taskDataSeq->inputs_count.emplace_back(picture.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  Sobel_seq Sobel_Sequential(taskDataSeq);
  ASSERT_EQ(Sobel_Sequential.validation(), true);
  Sobel_Sequential.pre_processing();
  ASSERT_EQ(Sobel_Sequential.run(), false);
}

TEST(sobol_a_sobel_seq, Test_Big_Piclure) {
  const uint8_t min = 0;
  const uint8_t max = 255;
  const int w = 100;
  const int h = 100;

  std::vector<int> in(2);
  in[0] = w;
  in[1] = h;
  std::vector<uint8_t> picture = getRandomPicture(w, h, min, max);
  std::vector<uint8_t> out(w * h, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(picture.data());
  taskDataSeq->inputs_count.emplace_back(picture.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  Sobel_seq Sobel_Sequential(taskDataSeq);
  ASSERT_EQ(Sobel_Sequential.validation(), true);
  Sobel_Sequential.pre_processing();
  Sobel_Sequential.run();
  Sobel_Sequential.post_processing();

  for (int i = 1; i < w - 1; i++) {
    for (int j = 1; j < h - 1; j++) {
      int x = -picture[(i - 1) * h + j - 1] + picture[(i - 1) * h + j + 1] -
               2 * picture[i * h + j - 1] + 2 * picture[i * h + j + 1] - 
               picture[(i + 1) * h + j - 1] + picture[(i + 1) * h + j + 1];
      int y = picture[(i - 1) * h + j - 1] + 2 * picture[(i - 1) * h + j] +
               picture[(i - 1) * h + j + 1] - picture[(i + 1) * h + j - 1] - 
               2 * picture[(i + 1) * h + j] - picture[(i + 1) * h + j + 1];
      uint8_t expected = sqrt(x * x + y * y);
      ASSERT_EQ(expected, out[i * h + j]);
    }
  }
}

TEST(sobol_a_sobel_seq, Test_White_Picture) {
  const int w = 10;
  const int h = 10;
  const uint8_t white = 255;

  std::vector<int> in(2);
  in[0] = w;
  in[1] = h;
  std::vector<uint8_t> picture(w * h, white);
  std::vector<uint8_t> out(w * h, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(picture.data());
  taskDataSeq->inputs_count.emplace_back(picture.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  Sobel_seq Sobel_Sequential(taskDataSeq);
  ASSERT_EQ(Sobel_Sequential.validation(), true);
  Sobel_Sequential.pre_processing();
  Sobel_Sequential.run();
  Sobel_Sequential.post_processing();

  for (int i = 1; i < w - 1; i++) {
    for (int j = 1; j < h - 1; j++) {
      ASSERT_EQ(0, out[i * h + j]);
    }
  }
}