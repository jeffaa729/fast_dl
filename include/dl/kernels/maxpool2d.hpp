#pragma once

namespace dl::kernels {

void max_pool2d_forward(const float* input,
                        float* output,
                        int batch_size,
                        int channels,
                        int H,
                        int W,
                        int H_out,
                        int W_out,
                        int kernel_size,
                        int stride,
                        int padding);

void max_pool2d_backward(const float* input,
                         const float* grad_output,
                         float* grad_input,
                         int batch_size,
                         int channels,
                         int H,
                         int W,
                         int H_out,
                         int W_out,
                         int kernel_size,
                         int stride,
                         int padding);

}  // namespace dl::kernels
