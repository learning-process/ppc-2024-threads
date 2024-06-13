// Copyright 2024 Veselov Ilya
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "stl/veselov_i_systemsgradmethod/include/systemsgradmethod_stl.hpp"

using namespace veselov_i_stl;

TEST(veselov_i_systems_grad_method_stl, test_pipeline) {
  int rows = 200;

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
  taskDataStl->outputs_count.emplace_back(vec.size());

  auto testTaskStl = std::make_shared<SystemsGradMethodStl>(taskDataStl);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskStl);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_TRUE(checkSolution(matrix, vec, res, 1e-6));
}

TEST(veselov_i_systems_grad_method_stl, test_task_run) {
  int rows = 200;

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
  taskDataStl->outputs_count.emplace_back(vec.size());

  auto testTaskStl = std::make_shared<SystemsGradMethodStl>(taskDataStl);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskStl);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_TRUE(checkSolution(matrix, vec, res, 1e-6));
}
