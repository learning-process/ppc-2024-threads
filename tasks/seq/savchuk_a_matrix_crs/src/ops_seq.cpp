#include "seq/savchuk_a_matrix_crs/include/ops_seq.hpp"

#include <thread>
#include <chrono>

using namespace std::chrono_literals;

bool TestTaskSequential::pre_processing() {
  // Retrieve input matrix dimensions
  rows_ = static_cast<int>(taskData->inputs_count[0]);
  cols_ = static_cast<int>(taskData->inputs_count[1]);

  // Initialize input matrices
  input_matrix_1_.resize(rows_ * cols_);
  input_matrix_2_.resize(cols_ * rows_);
  std::copy(reinterpret_cast<std::complex<double>*>(taskData->inputs[0]),
            reinterpret_cast<std::complex<double>*>(taskData->inputs[0]) + rows_ * cols_,
            input_matrix_1_.begin());
  std::copy(reinterpret_cast<std::complex<double>*>(taskData->inputs[1]),
            reinterpret_cast<std::complex<double>*>(taskData->inputs[1]) + cols_ * rows_,
            input_matrix_2_.begin());

  // Initialize output matrix
  output_matrix_.resize(rows_ * rows_);

  return true;
}

bool TestTaskSequential::validation() {
  // Check count elements of input and output matrices
  return static_cast<int>(taskData->inputs_count[0]) == rows_ && static_cast<int>(taskData->inputs_count[1]) == cols_
         && static_cast<int>(taskData->outputs_count[0]) == rows_ && static_cast<int>(taskData->outputs_count[1]) == rows_;
}

bool TestTaskSequential::run() {
  // Perform matrix multiplication
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < rows_; j++) {
      std::complex<double> sum = 0;
      for (int k = 0; k < cols_; k++) {
        sum += input_matrix_1_[i * cols_ + k] * input_matrix_2_[k * rows_ + j];
      }
      output_matrix_[i * rows_ + j] = sum;
    }
  }

  std::this_thread::sleep_for(20ms);  // Artificial delay simulation

  return true;
}

bool TestTaskSequential::post_processing() {
  // Store the result in the output buffer
  std::copy(output_matrix_.begin(), output_matrix_.end(),
            reinterpret_cast<std::complex<double>*>(taskData->outputs[0]));

  return true;
}