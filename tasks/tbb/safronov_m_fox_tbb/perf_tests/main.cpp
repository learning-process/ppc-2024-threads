// Copyright 2024 Safronov Mikhail

#include <gtest/gtest.h>
#include <tbb/tbb.h>
#include <chrono>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/safronov_m_fox_tbb/include/safronov_m_tbb.h"

double getCurrentTimeInSeconds() {
    static auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = current_time - start_time;
    return elapsed.count();
}

TEST(Safronov_M_Mult_Matrix_Fox, Test_Pipeline_Run) {
    size_t n = 300;
    double k = 50.0;
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> par(n * n);
    GetRandomValue(A.data(), A.size());
    identityMatrix(B.data(), n, k);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs_count.emplace_back(B.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(par.data()));
    taskDataSeq->outputs_count.emplace_back(par.size());

    auto safronovTaskTbb = std::make_shared<SafronovFoxAlgTaskTBB>(taskDataSeq);

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    perfAttr->current_timer = getCurrentTimeInSeconds;

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(safronovTaskTbb);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);

    for (size_t i = 0; i < n * n; i++) {
        EXPECT_DOUBLE_EQ(par[i], k * A[i]);
    }
}

TEST(Safronov_M_Mult_Matrix_Fox, Test_Task_Run) {
    size_t n = 300;
    double k = 50.0;
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> par(n * n);
    GetRandomValue(A.data(), A.size());
    identityMatrix(B.data(), n, k);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs_count.emplace_back(B.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(par.data()));
    taskDataSeq->outputs_count.emplace_back(par.size());

    auto safronovTaskTbb = std::make_shared<SafronovFoxAlgTaskTBB>(taskDataSeq);

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    perfAttr->current_timer = getCurrentTimeInSeconds;

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(safronovTaskTbb);
    perfAnalyzer->task_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);

    for (size_t i = 0; i < n * n; i++) {
        EXPECT_DOUBLE_EQ(par[i], k * A[i]);
    }
}
