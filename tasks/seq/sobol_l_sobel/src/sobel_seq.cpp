// Copyright 2023 Sobol Liubov
#include "seq/sobol_l_sobel/include/sobel_seq.hpp"

#include <random>
#include <thread>

using namespace std::chrono_literals;

bool Sobel_seq::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool Sobel_seq::pre_processing() {
  internal_order_test();
  w = reinterpret_cast<int *>(taskData->inputs[0])[0];
  h = reinterpret_cast<int *>(taskData->inputs[0])[1];
  for (int i = 0; i < w * h; i++) {
    input_.push_back(reinterpret_cast<uint8_t *>(taskData->inputs[1])[i]);
    res.push_back(0);
  }
  return true;
}

bool Sobel_seq::run() {
  internal_order_test();
  int size = w * h;
  if (size == 0) {
    return false;
  }
  if (size == 1) {
    res[0] = input_[0];
    return true;
  }

  for (int i = 1; i < w - 1; i++) {
    for (int j = 1; j < h - 1; j++) {
      int x = -input_[(i - 1) * h + j - 1] + input_[(i - 1) * h + j + 1] - 2 * input_[i * h + j - 1] +
               2 * input_[i * h + j + 1] - input_[(i + 1) * h + j - 1] + input_[(i + 1) * h + j + 1];
      int y = input_[(i - 1) * h + j - 1] + 2 * input_[(i - 1) * h + j] + input_[(i - 1) * h + j + 1] -
               input_[(i + 1) * h + j - 1] - 2 * input_[(i + 1) * h + j] - input_[(i + 1) * h + j + 1];
      res[i * h + j] = sqrt(x * x + y * y);
    }
  }
  std::this_thread::sleep_for(30ms);
  return true;
}

bool Sobel_seq::post_processing() {
  internal_order_test();
  for (int i = 0; i < w * h; i++) {
    reinterpret_cast<uint8_t *>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

std::vector<uint8_t> getRandomPicture(int w, int h, uint8_t min, uint8_t max) {
  std::random_device device;
  std::mt19937 gen(device());
  std::uniform_int_distribution<int> dstr(min, max);
  std::vector<uint8_t> picture(w * h);
  int size = w * h;
  for (int i = 0; i < size; i++) {
    picture[i] = dstr(gen);
  }
  return picture;
}
