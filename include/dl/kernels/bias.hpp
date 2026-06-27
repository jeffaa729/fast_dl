#pragma once

namespace dl::kernels {

void add_row_bias(float* output, const float* bias, int rows, int cols);

}  // namespace dl::kernels
