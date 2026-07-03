#pragma once

#include <cstdint>

namespace dl::kernels {

void conv2d(const float* input, const float* weight, const float* bias,
            float* output, int batch_size, int C_in, int H,
            int W, int C_out, int K_h,
            int K_w, int H_out, int W_out, int stride, int padding);

void conv2d_backward_input(const float* grad_output,
                           const float* weight,
                           float* grad_input,
                           int batch_size,
                           int C_in,
                           int H,
                           int W,
                           int C_out,
                           int K_h,
                           int K_w,
                           int H_out,
                           int W_out,
                           int stride,
                           int padding);

void conv2d_backward_weight(const float* input,
                            const float* grad_output,
                            float* grad_weight,
                            int batch_size,
                            int C_in,
                            int H,
                            int W,
                            int C_out,
                            int K_h,
                            int K_w,
                            int H_out,
                            int W_out,
                            int stride,
                            int padding);

void conv2d_backward_bias(const float* grad_output,
                          float* grad_bias,
                          int batch_size,
                          int C_out,
                          int H_out,
                          int W_out);

}  // namespace dl::kernels
