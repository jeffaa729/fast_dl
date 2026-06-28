#include <dl/kernels/softmax.hpp>

#include <cuda_runtime.h>

#include <cmath>

namespace {

constexpr int kSoftmaxThreadsPerBlock = 128;
constexpr int kWarpSize = 32;

__device__ float warp_reduce_max(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void softmax_shared_memory_kernel(float* output, const float* input,
                                             int rows, int cols) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int block_size = blockDim.x;
    if (row >= rows) {
        return;
    }

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float max_value = -INFINITY;
    for (int col = tid; col < cols; col += block_size) {
        max_value = fmaxf(max_value, row_input[col]);
    }

    shared[tid] = max_value;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    max_value = shared[0];

    float sum = 0.0f;
    for (int col = tid; col < cols; col += block_size) {
        const float value = expf(row_input[col] - max_value);
        row_output[col] = value;
        sum += value;
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    sum = shared[0];

    const float norm = 1.0f / sum;
    for (int col = tid; col < cols; col += block_size) {
        row_output[col] *= norm;
    }
}

void launch_softmax_shared_memory(const float* input, float* output,
                                  std::size_t rows, std::size_t cols) {
    const std::size_t shared_bytes = kSoftmaxThreadsPerBlock * sizeof(float);
    softmax_shared_memory_kernel<<<static_cast<unsigned int>(rows),
                                   kSoftmaxThreadsPerBlock, shared_bytes>>>(
        output, input, static_cast<int>(rows), static_cast<int>(cols));
}

template <int ELEMS_PER_THREAD>
__global__ void softmax_warpshfl_reg_cache_kernel(float* output,
                                                  const float* input,
                                                  int rows, int cols) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int block_size = blockDim.x;
    const int lane = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;
    const int warps_per_block = block_size / kWarpSize;
    if (row >= rows) {
        return;
    }

    const int vec_cols = cols / 4;
    const float4* input_row4 =
        reinterpret_cast<const float4*>(input + row * cols);
    float4* output_row4 = reinterpret_cast<float4*>(output + row * cols);
    __shared__ float warp_values[kSoftmaxThreadsPerBlock / kWarpSize];

    float4 reg[ELEMS_PER_THREAD];
    float max_value = -INFINITY;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int vec_col = tid + i * block_size;
        if (vec_col < vec_cols) {
            reg[i] = input_row4[vec_col];
            max_value = fmaxf(
                max_value,
                fmaxf(fmaxf(reg[i].x, reg[i].y), fmaxf(reg[i].z, reg[i].w)));
        }
    }

    max_value = warp_reduce_max(max_value);
    if (lane == 0) {
        warp_values[warp_id] = max_value;
    }
    __syncthreads();

    max_value = tid < warps_per_block ? warp_values[lane] : -INFINITY;
    if (warp_id == 0) {
        max_value = warp_reduce_max(max_value);
    }
    if (tid == 0) {
        warp_values[0] = max_value;
    }
    __syncthreads();
    max_value = warp_values[0];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int vec_col = tid + i * block_size;
        if (vec_col < vec_cols) {
            reg[i].x = expf(reg[i].x - max_value);
            reg[i].y = expf(reg[i].y - max_value);
            reg[i].z = expf(reg[i].z - max_value);
            reg[i].w = expf(reg[i].w - max_value);
            sum += reg[i].x + reg[i].y + reg[i].z + reg[i].w;
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        warp_values[warp_id] = sum;
    }
    __syncthreads();

    sum = tid < warps_per_block ? warp_values[lane] : 0.0f;
    if (warp_id == 0) {
        sum = warp_reduce_sum(sum);
    }
    if (tid == 0) {
        warp_values[0] = sum;
    }
    __syncthreads();
    sum = warp_values[0];

    const float norm = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int vec_col = tid + i * block_size;
        if (vec_col < vec_cols) {
            float4 out_value;
            out_value.x = reg[i].x * norm;
            out_value.y = reg[i].y * norm;
            out_value.z = reg[i].z * norm;
            out_value.w = reg[i].w * norm;
            output_row4[vec_col] = out_value;
        }
    }
}

void launch_softmax_warpshfl_reg_cache(const float* input, float* output,
                                       std::size_t rows, std::size_t cols) {
    constexpr int elems_per_thread = 8;
    constexpr int max_cols =
        kSoftmaxThreadsPerBlock * elems_per_thread * 4;

    if (cols % 4 != 0 || cols > max_cols) {
        launch_softmax_shared_memory(input, output, rows, cols);
        return;
    }

    softmax_warpshfl_reg_cache_kernel<elems_per_thread>
        <<<static_cast<unsigned int>(rows), kSoftmaxThreadsPerBlock>>>(
            output, input, static_cast<int>(rows), static_cast<int>(cols));
}

}  // namespace

namespace dl::kernels {

void softmax(const float* input, float* output, std::size_t rows,
             std::size_t cols) {
    launch_softmax_warpshfl_reg_cache(input, output, rows, cols);
}

}  // namespace dl::kernels
