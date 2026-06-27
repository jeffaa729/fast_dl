#pragma once

namespace dl::kernels {

void layernorm(const float* input, float* output, int rows, int cols,
               float eps);

}  // namespace dl::kernels
