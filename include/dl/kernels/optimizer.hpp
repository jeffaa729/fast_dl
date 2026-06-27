#pragma once

#include <cstdint>

namespace dl::kernels {

void sgd_step_float32(float* param, const float* grad, float learning_rate,
                      int64_t n);

}  // namespace dl::kernels
