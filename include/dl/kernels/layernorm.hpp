#pragma once

namespace dl::kernels {

void layernorm(const float* input, float* output, int rows, int cols,
               float eps);

void layernorm_backward(const float* input, const float* grad_output,
                        float* grad_input, int rows, int cols, float eps);

}  // namespace dl::kernels
