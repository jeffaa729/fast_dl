#pragma once

#include <cstddef>

namespace dl::kernels {

void transpose(const float* input, float* output, std::size_t size);

}  // namespace dl::kernels
