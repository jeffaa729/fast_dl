#include <dl/kernels/conv2d.hpp>

namespace {
    __global__ void conv2d_kernel(const float* input, const float* weight, const float* bias,
                          float* output, int batch_size, int in_channels,
                          int in_height, int in_width, int out_channels,
                          int kernel_height, int kernel_width, int stride,
                          int padding) {
        const int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        const int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = batch_size * out_channels * out_height * out_width;
        if (idx >= total) {
            return;
        }

        const int n = idx / (out_channels * out_height * out_width);
        const int c_out = (idx / (out_height * out_width)) % out_channels;
        const int h_out = (idx / out_width) % out_height;
        const int w_out = idx % out_width;

        float sum = 0.0f;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    const int h_in = h_out * stride - padding + kh;
                    const int w_in = w_out * stride - padding + kw;
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        sum += input[n * in_channels * in_height * in_width +
                                     c_in * in_height * in_width +
                                     h_in * in_width + w_in] *
                               weight[c_out * in_channels * kernel_height * kernel_width +
                                      c_in * kernel_height * kernel_width +
                                      kh * kernel_width + kw];
                    }
                }
            }
        }
        sum += bias[c_out];
        output[idx] = sum;
    }
}