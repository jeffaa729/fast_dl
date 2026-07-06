#include <dl/kernels/maxpool2d.hpp>

#include <cuda_runtime.h>

#include <cfloat>

namespace {

__global__ void max_pool2d_forward_kernel(const float* input,
                                          float* output,
                                          int batch_size,
                                          int channels,
                                          int H,
                                          int W,
                                          int H_out,
                                          int W_out,
                                          int kernel_size,
                                          int stride,
                                          int padding) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * H_out * W_out;
    if (idx >= total) {
        return;
    }

    const int ow = idx % W_out;
    const int oh = (idx / W_out) % H_out;
    const int c = (idx / (H_out * W_out)) % channels;
    const int n = idx / (channels * H_out * W_out);

    float max_value = -FLT_MAX;
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int ih = oh * stride - padding + kh;
        if (ih < 0 || ih >= H) {
            continue;
        }
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int iw = ow * stride - padding + kw;
            if (iw < 0 || iw >= W) {
                continue;
            }

            const int input_idx = ((n * channels + c) * H + ih) * W + iw;
            const float value = input[input_idx];
            if (value > max_value) {
                max_value = value;
            }
        }
    }

    output[idx] = max_value;
}

__global__ void max_pool2d_backward_kernel(const float* input,
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
                                           int padding) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * H_out * W_out;
    if (idx >= total) {
        return;
    }

    const int ow = idx % W_out;
    const int oh = (idx / W_out) % H_out;
    const int c = (idx / (H_out * W_out)) % channels;
    const int n = idx / (channels * H_out * W_out);

    float max_value = -FLT_MAX;
    int max_input_idx = -1;
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int ih = oh * stride - padding + kh;
        if (ih < 0 || ih >= H) {
            continue;
        }
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int iw = ow * stride - padding + kw;
            if (iw < 0 || iw >= W) {
                continue;
            }

            const int input_idx = ((n * channels + c) * H + ih) * W + iw;
            const float value = input[input_idx];
            if (value > max_value) {
                max_value = value;
                max_input_idx = input_idx;
            }
        }
    }

    if (max_input_idx >= 0) {
        atomicAdd(&grad_input[max_input_idx], grad_output[idx]);
    }
}

}  // namespace

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
                        int padding) {
    const int total = batch_size * channels * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    max_pool2d_forward_kernel<<<num_blocks, block_size>>>(
        input,
        output,
        batch_size,
        channels,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        padding);
}

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
                         int padding) {
    const int total = batch_size * channels * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    max_pool2d_backward_kernel<<<num_blocks, block_size>>>(
        input,
        grad_output,
        grad_input,
        batch_size,
        channels,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        padding);
}

}  // namespace dl::kernels
