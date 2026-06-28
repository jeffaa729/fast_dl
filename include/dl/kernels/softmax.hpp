#pragma once

#include <cstddef>

namespace dl::kernels {

void softmax(const float* input, float* output, std::size_t rows,
             std::size_t cols);

}  // namespace dl::kernels
