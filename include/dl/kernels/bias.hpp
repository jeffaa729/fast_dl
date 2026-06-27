#pragma once

namespace dl::kernels {

void add_row_bias(float* output, const float* bias, int rows, int cols);
void sum_rows(const float* input, float* output, int rows, int cols);

}  // namespace dl::kernels
