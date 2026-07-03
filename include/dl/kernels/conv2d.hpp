#pragma once

#include <cstdint>

namespace dl::kernels {

void conv2d(const float* input, const float* weight, const float* bias,
            float* output, int batch_size, int in_channels, int in_height,
            int in_width, int out_channels, int kernel_height,
            int kernel_width, int stride, int padding);
}  // namespace dl::kernels
