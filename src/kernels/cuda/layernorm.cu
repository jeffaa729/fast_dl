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

}  // namespace

namespace dl::kernels {

void layernorm(const float* input, float* output, int rows, int cols,
               float eps) {
    launch_layernorm_kernel(input, output, rows, cols, eps);
}

}  // namespace dl::kernels
