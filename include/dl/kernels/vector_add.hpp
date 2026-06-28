#pragma once

#include <cstddef>

namespace dl::kernels {

void vector_add(const float* a, const float* b, float* c, std::size_t size);

}  // namespace dl::kernels
