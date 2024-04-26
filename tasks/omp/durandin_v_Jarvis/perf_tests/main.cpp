// Copyright 2024 Durandin Vladimir
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "omp/durandin_v_Jarvis/include/ops_omp.hpp"

TEST(VladimirD_OpenMP_Perf_Test, test_pipeline_run) {
  auto generatePointsInCircle = [](const Jarvis::Point2d& center, double radius, uint32_t numPoints) {
    std::vector<Jarvis::Point2d> points;
    double angleIncrement = 2 * Const::MY_PI / numPoints;
    for (uint32_t i = 0; i < numPoints; ++i) {
      double angle = i * angleIncrement;
      double x = center.x + radius * std::cos(angle);
      double y = center.y + radius * std::sin(angle);
      points.push_back({x, y});
    }
    return points;
  };

  Jarvis::Point2d center{0, 0};
  double radius = 5.0;
  uint32_t numPoints = 10'000;  // Number of points for the circle

  std::vector<Jarvis::Point2d> points = generatePointsInCircle(center, radius, numPoints);
  // Expected result: the circle should be a convex hull
  std::vector<Jarvis::Point2d> expectedHull = points;  // Since the circle is already a convex hull

  std::vector<Jarvis::Point2d> out(points.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(points.size()));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataPar->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  auto testTaskParallel = std::make_shared<Jarvis::JarvisTestTaskParallel>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Such a strange check because my algorithm outputs points in a different order
  uint32_t tmp = numPoints >> 1;

  for (uint32_t i = 0; i < expectedHull.size(); ++i) {
    if (i < tmp) {
      EXPECT_EQ(expectedHull[i].x, out[i + tmp].x);
      EXPECT_EQ(expectedHull[i].y, out[i + tmp].y);
    } else {
      EXPECT_EQ(expectedHull[i].x, out[i - tmp].x);
      EXPECT_EQ(expectedHull[i].y, out[i - tmp].y);
    }
  }
}

TEST(VladimirD_OpenMP_Perf_Test, test_task_run) {
  auto generatePointsInCircle = [](const Jarvis::Point2d& center, double radius, uint32_t numPoints) {
    std::vector<Jarvis::Point2d> points;
    double angleIncrement = 2 * Const::MY_PI / numPoints;
    for (uint32_t i = 0; i < numPoints; ++i) {
      double angle = i * angleIncrement;
      double x = center.x + radius * std::cos(angle);
      double y = center.y + radius * std::sin(angle);
      points.push_back({x, y});
    }
    return points;
  };

  Jarvis::Point2d center{0, 0};
  double radius = 5.0;
  uint32_t numPoints = 10'000;  // Number of points for the circle

  std::vector<Jarvis::Point2d> points = generatePointsInCircle(center, radius, numPoints);
  // Expected result: the circle should be a convex hull
  std::vector<Jarvis::Point2d> expectedHull = points;  // Since the circle is already a convex hull

  std::vector<Jarvis::Point2d> out(points.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(points.size()));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataPar->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  auto testTaskParallel = std::make_shared<Jarvis::JarvisTestTaskParallel>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Such a strange check because my algorithm outputs points in a different order
  uint32_t tmp = numPoints >> 1;

  for (uint32_t i = 0; i < out.size(); ++i) {
    if (i < tmp) {
      EXPECT_EQ(expectedHull[i].x, out[i + tmp].x);
      EXPECT_EQ(expectedHull[i].y, out[i + tmp].y);
    } else {
      EXPECT_EQ(expectedHull[i].x, out[i - tmp].x);
      EXPECT_EQ(expectedHull[i].y, out[i - tmp].y);
    }
  }
}
