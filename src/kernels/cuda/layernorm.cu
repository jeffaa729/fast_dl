#include <dl/kernels/layernorm.hpp>

#include <cuda_runtime.h>

#include <cmath>

namespace {

__global__ void layernorm_kernel(const float* input, float* output, int rows,
                                 int cols, float eps) {
    extern __shared__ float shared[];
    float* sums = shared;
    float* sq_sums = shared + blockDim.x;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float value = input[row * cols + col];
        local_sum += value;
        local_sq_sum += value * value;
    }

    sums[tid] = local_sum;
    sq_sums[tid] = local_sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            sq_sums[tid] += sq_sums[tid + stride];
        }
        __syncthreads();
    }

    const float mean = sums[0] / static_cast<float>(cols);
    const float mean_square = sq_sums[0] / static_cast<float>(cols);
    const float variance = fmaxf(mean_square - mean * mean, 0.0f);
    const float inv_std = rsqrtf(variance + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        const float value = input[row * cols + col];
        output[row * cols + col] = (value - mean) * inv_std;
    }
}

__global__ void layernorm_backward_kernel(const float* input,
                                          const float* grad_output,
                                          float* grad_input,
                                          int rows,
                                          int cols,
                                          float eps) {
    extern __shared__ float shared[];
    float* sums = shared;
    float* sq_sums = shared + blockDim.x;
    float* grad_sums = shared + 2 * blockDim.x;
    float* grad_norm_sums = shared + 3 * blockDim.x;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float value = input[row * cols + col];
        local_sum += value;
        local_sq_sum += value * value;
    }

    sums[tid] = local_sum;
    sq_sums[tid] = local_sq_sum;
    grad_sums[tid] = 0.0f;
    grad_norm_sums[tid] = 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            sq_sums[tid] += sq_sums[tid + stride];
        }
        __syncthreads();
    }

    const float mean = sums[0] / static_cast<float>(cols);
    const float mean_square = sq_sums[0] / static_cast<float>(cols);
    const float variance = fmaxf(mean_square - mean * mean, 0.0f);
    const float inv_std = rsqrtf(variance + eps);

    float local_grad_sum = 0.0f;
    float local_grad_norm_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const int index = row * cols + col;
        const float normalized = (input[index] - mean) * inv_std;
        const float grad = grad_output[index];
        local_grad_sum += grad;
        local_grad_norm_sum += grad * normalized;
    }

    grad_sums[tid] = local_grad_sum;
    grad_norm_sums[tid] = local_grad_norm_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            grad_sums[tid] += grad_sums[tid + stride];
            grad_norm_sums[tid] += grad_norm_sums[tid + stride];
        }
        __syncthreads();
    }

    const float mean_grad = grad_sums[0] / static_cast<float>(cols);
    const float mean_grad_norm = grad_norm_sums[0] / static_cast<float>(cols);
    for (int col = tid; col < cols; col += blockDim.x) {
        const int index = row * cols + col;
        const float normalized = (input[index] - mean) * inv_std;
        grad_input[index] =
            inv_std * (grad_output[index] - mean_grad -
                       normalized * mean_grad_norm);
    }
}

int next_power_of_two_at_most_1024(int value) {
    int result = 1;
    while (result < value && result < 1024) {
        result <<= 1;
    }
    return result;
}

void launch_layernorm_kernel(const float* input, float* output, int rows,
                             int cols, float eps) {
    const int threads = next_power_of_two_at_most_1024(cols);
    const int shared_bytes = 2 * threads * static_cast<int>(sizeof(float));
    layernorm_kernel<<<rows, threads, shared_bytes>>>(input, output, rows, cols,
                                                      eps);
}

void launch_layernorm_backward_kernel(const float* input,
                                      const float* grad_output,
                                      float* grad_input,
                                      int rows,
                                      int cols,
                                      float eps) {
    const int threads = next_power_of_two_at_most_1024(cols);
    const int shared_bytes = 4 * threads * static_cast<int>(sizeof(float));
    layernorm_backward_kernel<<<rows, threads, shared_bytes>>>(
        input, grad_output, grad_input, rows, cols, eps);
}

}  // namespace

namespace dl::kernels {

void layernorm(const float* input, float* output, int rows, int cols,
               float eps) {
    launch_layernorm_kernel(input, output, rows, cols, eps);
}

void layernorm_backward(const float* input, const float* grad_output,
                        float* grad_input, int rows, int cols, float eps) {
    launch_layernorm_backward_kernel(input, grad_output, grad_input, rows, cols,
                                     eps);
}

}  // namespace dl::kernels
