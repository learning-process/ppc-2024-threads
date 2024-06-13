// Copyright 2024 Veselov Ilya

#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <vector>

#include "stl/veselov_i_systemsgradmethod/include/systemsgradmethod_stl.hpp"

using namespace veselov_i_stl;

TEST(veselov_i_systems_grad_method_stl, Test_some_matrix) {
  int rows = 2;
  std::vector<double> matrix = {2.0, 1.0, 1.0, 2.0};
  std::vector<double> vec = {1.0, 3.0};
  std::vector<double> res(rows, 0.0);
  std::vector<double> excepted_res = {-0.3333333, 1.6666666};

  std::shared_ptr<ppc::core::TaskData> taskDataStl = std::make_shared<ppc::core::TaskData>();
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataStl->inputs_count.emplace_back(rows * rows);
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataStl->inputs_count.emplace_back(vec.size());
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskDataStl->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataStl->outputs_count.emplace_back(vec.size());

  SystemsGradMethodStl systemsGradMethodStl(taskDataStl);
  ASSERT_EQ(systemsGradMethodStl.validation(), true);
  ASSERT_TRUE(systemsGradMethodStl.pre_processing());
  ASSERT_TRUE(systemsGradMethodStl.run());
  ASSERT_TRUE(systemsGradMethodStl.post_processing());
  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_LE(abs(excepted_res[i] - res[i]), 1e-6);
  }
}

TEST(veselov_i_systems_grad_method_stl, Test_unit_matrix) {
  int rows = 4;
  std::vector<double> matrix = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<double> vec = {1, 2, 3, 4};
  std::vector<double> res(rows, 0.0);
  std::vector<double> excepted_res = {1, 2, 3, 4};

  std::shared_ptr<ppc::core::TaskData> taskDataStl = std::make_shared<ppc::core::TaskData>();
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataStl->inputs_count.emplace_back(rows * rows);
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataStl->inputs_count.emplace_back(vec.size());
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskDataStl->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataStl->outputs_count.emplace_back(vec.size());

  SystemsGradMethodStl systemsGradMethodStl(taskDataStl);
  ASSERT_EQ(systemsGradMethodStl.validation(), true);
  systemsGradMethodStl.pre_processing();
  systemsGradMethodStl.run();
  systemsGradMethodStl.post_processing();
  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_LE(abs(excepted_res[i] - res[i]), 1e-6);
  }
}

TEST(veselov_i_systems_grad_method_stl, Test_random_10) {
  int rows = 10;
  std::vector<double> matrix = genRandomMatrix(rows, 10);
  std::vector<double> vec = genRandomVector(rows, 10);
  std::vector<double> res(rows, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataStl = std::make_shared<ppc::core::TaskData>();
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataStl->inputs_count.emplace_back(matrix.size());
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataStl->inputs_count.emplace_back(vec.size());
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskDataStl->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataStl->outputs_count.emplace_back(res.size());

  SystemsGradMethodStl systemsGradMethodStl(taskDataStl);
  ASSERT_EQ(systemsGradMethodStl.validation(), true);
  systemsGradMethodStl.pre_processing();
  systemsGradMethodStl.run();
  systemsGradMethodStl.post_processing();
  ASSERT_TRUE(checkSolution(matrix, vec, res));
}

TEST(veselov_i_systems_grad_method_stl, Test_random_5) {
  int rows = 5;
  std::vector<double> matrix = genRandomMatrix(rows, 10);
  std::vector<double> vec = genRandomVector(rows, 10);
  std::vector<double> res(rows);

  std::shared_ptr<ppc::core::TaskData> taskDataStl = std::make_shared<ppc::core::TaskData>();
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataStl->inputs_count.emplace_back(matrix.size());
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataStl->inputs_count.emplace_back(vec.size());
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskDataStl->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataStl->outputs_count.emplace_back(res.size());

  SystemsGradMethodStl systemsGradMethodStl(taskDataStl);
  ASSERT_EQ(systemsGradMethodStl.validation(), true);
  systemsGradMethodStl.pre_processing();
  systemsGradMethodStl.run();
  systemsGradMethodStl.post_processing();
  ASSERT_TRUE(checkSolution(matrix, vec, res));
}

TEST(veselov_i_systems_grad_method_stl, Test_triple_diag_matrix) {
  int rows = 4;
  std::vector<double> matrix = {2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2};
  std::vector<double> vec = {1, 2, 3, 4};
  std::vector<double> res(rows);
  std::vector<double> excepted_res = {4, 7, 8, 6};

  std::shared_ptr<ppc::core::TaskData> taskDataStl = std::make_shared<ppc::core::TaskData>();
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataStl->inputs_count.emplace_back(matrix.size());
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataStl->inputs_count.emplace_back(vec.size());
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows));
  taskDataStl->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataStl->outputs_count.emplace_back(vec.size());

  SystemsGradMethodStl systemsGradMethodStl(taskDataStl);
  ASSERT_EQ(systemsGradMethodStl.validation(), true);
  systemsGradMethodStl.pre_processing();
  systemsGradMethodStl.run();
  systemsGradMethodStl.post_processing();
  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_LE(abs(excepted_res[i] - res[i]), 1e-6);
  }
}
