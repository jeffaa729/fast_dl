#include <dl/kernels/conv2d.hpp>

#include <cuda_runtime.h>

#include <cmath>

namespace {
    __global__ void conv2d_kernel(const float* input, const float* weight, const float* bias,
                       float* output, int batch_size, int C_in, int H,
                       int W, int C_out, int K_h,
                       int K_w, int H_out, int W_out, int stride, int padding) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = batch_size * C_out * H_out * W_out;
        if (idx >= total) {
            return;
        }

        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int c_out = (idx / (H_out * W_out)) % C_out;
        const int n = idx / (C_out * H_out * W_out);

        float sum = bias[c_out];
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K_h; ++kh) {
                for (int kw = 0; kw < K_w; ++kw) {
                    const int h_in = h_out * stride - padding + kh;
                    const int w_in = w_out * stride - padding + kw;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        sum += input[n * C_in * H * W + c_in * H * W + h_in * W + w_in] *
                               weight[c_out * C_in * K_h * K_w + c_in * K_h * K_w + kh * K_w + kw];
                    }
                }
            }
        }
        output[idx] = sum;
    }

    __global__ void conv2d_backward_input_kernel(const float* grad_output,
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
                                                 int padding) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = batch_size * C_in * H * W;
        if (idx >= total) {
            return;
        }

        const int iw = idx % W;
        const int ih = (idx / W) % H;
        const int ci = (idx / (H * W)) % C_in;
        const int n = idx / (C_in * H * W);

        float sum = 0.0f;  // dL/dinput[n, ci, ih, iw]
        for (int co = 0; co < C_out; ++co) {
            for (int kh = 0; kh < K_h; ++kh) {
                const int oh_unstrided = ih + padding - kh;
                if (oh_unstrided < 0 || oh_unstrided % stride != 0) {
                    continue;
                }
                const int oh = oh_unstrided / stride;
                if (oh < 0 || oh >= H_out) {
                    continue;
                }

                for (int kw = 0; kw < K_w; ++kw) {
                    const int ow_unstrided = iw + padding - kw;
                    if (ow_unstrided < 0 || ow_unstrided % stride != 0) {
                        continue;
                    }
                    const int ow = ow_unstrided / stride;
                    if (ow < 0 || ow >= W_out) {
                        continue;
                    }

                    const int grad_output_idx =
                        ((n * C_out + co) * H_out + oh) * W_out + ow;
                    const int weight_idx =
                        ((co * C_in + ci) * K_h + kh) * K_w + kw;
                    sum += grad_output[grad_output_idx] * weight[weight_idx];
                }
            }
        }

        grad_input[idx] = sum;
    }

    __global__ void conv2d_backward_weight_kernel(const float* input,
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
                                                  int padding) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = C_out * C_in * K_h * K_w;
        if (idx >= total) {
            return;
        }

        const int kw = idx % K_w;
        const int kh = (idx / K_w) % K_h;
        const int ci = (idx / (K_h * K_w)) % C_in;
        const int co = idx / (C_in * K_h * K_w);

        float sum = 0.0f;  // dL/dweight[co, ci, kh, kw]
        for (int n = 0; n < batch_size; ++n) {
            for (int oh = 0; oh < H_out; ++oh) {
                const int ih = oh * stride - padding + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }
                for (int ow = 0; ow < W_out; ++ow) {
                    const int iw = ow * stride - padding + kw;
                    if (iw < 0 || iw >= W) {
                        continue;
                    }

                    const int input_idx =
                        ((n * C_in + ci) * H + ih) * W + iw;
                    const int grad_output_idx =
                        ((n * C_out + co) * H_out + oh) * W_out + ow;
                    sum += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }

        grad_weight[idx] = sum;
    }

    __global__ void conv2d_backward_bias_kernel(const float* grad_output,
                                                float* grad_bias,
                                                int batch_size,
                                                int C_out,
                                                int H_out,
                                                int W_out) {
        const int co = blockIdx.x * blockDim.x + threadIdx.x;
        if (co >= C_out) {
            return;
        }

        float sum = 0.0f;  // dL/dbias[co]
        for (int n = 0; n < batch_size; ++n) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    const int grad_output_idx =
                        ((n * C_out + co) * H_out + oh) * W_out + ow;
                    sum += grad_output[grad_output_idx];
                }
            }
        }

        grad_bias[co] = sum;
    }
}

namespace dl::kernels {

void conv2d(const float* input, const float* weight, const float* bias,
            float* output, int batch_size, int C_in, int H,
            int W, int C_out, int K_h,
            int K_w, int H_out, int W_out, int stride, int padding) {
    const int total = batch_size * C_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    conv2d_kernel<<<num_blocks, block_size>>>(
        input,
        weight,
        bias,
        output,
        batch_size,
        C_in,
        H,
        W,
        C_out,
        K_h,
        K_w,
        H_out,
        W_out,
        stride,
        padding);
}

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
                           int padding) {
    const int total = batch_size * C_in * H * W;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    conv2d_backward_input_kernel<<<num_blocks, block_size>>>(
        grad_output,
        weight,
        grad_input,
        batch_size,
        C_in,
        H,
        W,
        C_out,
        K_h,
        K_w,
        H_out,
        W_out,
        stride,
        padding);
}

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
                            int padding) {
    const int total = C_out * C_in * K_h * K_w;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    conv2d_backward_weight_kernel<<<num_blocks, block_size>>>(
        input,
        grad_output,
        grad_weight,
        batch_size,
        C_in,
        H,
        W,
        C_out,
        K_h,
        K_w,
        H_out,
        W_out,
        stride,
        padding);
}

void conv2d_backward_bias(const float* grad_output,
                          float* grad_bias,
                          int batch_size,
                          int C_out,
                          int H_out,
                          int W_out) {
    const int block_size = 256;
    const int num_blocks = (C_out + block_size - 1) / block_size;
    conv2d_backward_bias_kernel<<<num_blocks, block_size>>>(
        grad_output,
        grad_bias,
        batch_size,
        C_out,
        H_out,
        W_out);
}

}  // namespace dl::kernels
