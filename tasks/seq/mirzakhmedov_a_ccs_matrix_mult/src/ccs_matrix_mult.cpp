// Copyright 2024 Mirzakhmedov Alexander
#include "seq/mirzakhmedov_a_ccs_matrix_mult/include/ops_seq.hpp"

#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

bool CSeq::pre_processing() {
    internal_order_test();
    A = reinterpret_cast<CCSSparseMatrix*>(taskData->inputs[0]);
    B = reinterpret_cast<CCSSparseMatrix*>(taskData->inputs[1]);
    C = reinterpret_cast<CCSSparseMatrix*>(taskData->outputs[0]);
    return true;
}

bool CSeq::validation() {
    internal_order_test();
    int A_cols = reinterpret_cast<CCSSparseMatrix*>(taskData->inputs[0])->cols_num;
    int B_rows = reinterpret_cast<CCSSparseMatrix*>(taskData->inputs[1])->rows_num;
    return (A_cols == B_rows);
}

bool CSeq::run() {
    internal_order_test();

    C->numRows = A->numRows;
    C->numCols = B->numCols;
    C->columnPointers.resize(C->numCols + 1);
    C->columnPointers[0] = 0;
    std::vector<int> P_elem(C->numRows);
    for (int b_col = 0; b_col < C->numCols; ++b_col) {
        for (int c_row = 0; c_row < C->numRows; ++c_row) {
            P_elem[c_row] = 0;
        }
        for (int b_idx = B->columnPointers[b_col]; b_idx < B->columnPointers[b_col + 1]; ++b_idx) {
            int b_row = B->rowIndices[b_idx];
            for (int a_idx = A->columnPointers[b_row]; a_idx < A->columnPointers[b_row + 1]; ++a_idx) {
                P_elem[A->rowIndices[a_idx]] = 1;
            }
        }
        int col_nonumNonZeros_count = 0;
        for (int c_row = 0; c_row < C->numRows; ++c_row) {
            col_nonumNonZeros_count += P_elem[c_row];
        }
        C->columnPointers[b_col + 1] = col_nonumNonZeros_count + C->columnPointers[b_col];
    }

    int total_nonumNonZeross = C->columnPointers[C->numCols];
    C->numNonZeros = total_nonumNonZeross;
    C->rowIndices.resize(total_nonumNonZeross);
    C->nonzeroValues.resize(total_nonumNonZeross);

    std::complex<double> zero;
    std::complex<double> BValue;
    std::vector<std::complex<double>> acc(C->numRows);
    for (int b_col = 0; b_col < C->numCols; ++b_col) {
        for (int c_row = 0; c_row < C->numRows; ++c_row) {
            acc[c_row] = zero;
            P_elem[c_row] = 0;
        }
        for (int b_idx = B->columnPointers[b_col]; b_idx < B->columnPointers[b_col + 1]; ++b_idx) {
            int b_row = B->rowIndices[b_idx];
            BValue = B->nonzeroValues[b_idx];
            for (int a_idx = A->columnPointers[b_row]; a_idx < A->columnPointers[b_row + 1]; ++a_idx) {
                int a_row = A->rowIndices[a_idx];
                acc[a_row] += A->nonzeroValues[a_idx] * BValue;
                P_elem[a_row] = 1;
            }
        }
        int c_pos = C->columnPointers[b_col];
        for (int c_row = 0; c_row < C->numRows; ++c_row) {
            if (P_elem[c_row] != 0) {
                C->rowIndices[c_pos] = c_row;
                C->nonzeroValues[c_pos++] = acc[c_row];
            }
        }
    }

    return true;
}

bool CSeq::post_processing() {
    internal_order_test();
    return true;
}