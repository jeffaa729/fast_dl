#pragma once

#include <cstdint>

namespace dl::kernels {

void uniform_float32(float* data, int64_t n, float low, float high, uint64_t seed);
void normal_float32(float* data, int64_t n, float mean, float stddev, uint64_t seed);

}  // namespace dl::kernels
