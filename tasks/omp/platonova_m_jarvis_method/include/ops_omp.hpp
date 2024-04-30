// Copyright 2024 Platonova Maria
#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

struct Point {
  int x, y;

  bool operator==(const Point& other) const { return x == other.x && y == other.y; }
  bool operator<(const Point& other) const {
    if (x != other.x) {
      return x < other.x;
    }
    return y < other.y;
  }
};

class TestOMPJarvisSeq : public ppc::core::Task {
 public:
  explicit TestTaskSequentialJarvis(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Point> points;
  std::vector<Point> resPoints;
};

class TestOMPJarvisParallel : public ppc::core::Task {
 public:
  explicit TestOMPTaskParallelMoiseevJarvis(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Point> points;
  std::vector<Point> resPoints;
};
std::vector<Point> Jarvis(const std::vector<Point>& points);
std::vector<Point> Jarvis_omp(const std::vector<Point>& points, int num_threads);