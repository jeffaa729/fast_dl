#include <dl/kernels/softmax.hpp>

#include <cuda_runtime.h>

#include <cfloat>
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

// Global memory baseline: one thread computes one row.
__global__ void softmax_naive_kernel(const float* input, float* output,
                                     std::size_t rows, std::size_t cols) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float max_value = -FLT_MAX;
    for (std::size_t col = 0; col < cols; ++col) {
        max_value = fmaxf(max_value, row_input[col]);
    }

    float sum = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
        const float value = expf(row_input[col] - max_value);
        row_output[col] = value;
        sum += value;
    }

    for (std::size_t col = 0; col < cols; ++col) {
        row_output[col] /= sum;
    }
}

void launch_softmax_naive(const float* input, float* output, std::size_t rows,
                          std::size_t cols) {
    softmax_naive_kernel<<<static_cast<unsigned int>(rows), 1>>>(
        input, output, rows, cols);
}

__global__ void softmax_shared_memory_kernel(float* output, const float* input,
                                             int rows, int cols) {
    // One block owns one row. Shared memory stores one partial value per thread
    // during the block-level max and sum reductions.
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int block_size = blockDim.x;
    if (row >= rows) {
        return;
    }

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // Pass 1: each thread scans a strided part of the row, then the block
    // reduces those partial maxima into shared[0].
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

    // Pass 2: compute exp(x - max) for numerical stability, write it to global
    // memory, and reduce the row sum across the block.
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

    // Pass 3: normalize every stored exp value by the row sum.
    const float norm = 1.0f / sum;
    for (int col = tid; col < cols; col += block_size) {
        row_output[col] *= norm;
    }
}

void launch_softmax_shared_memory(const float* input, float* output,
                                  std::size_t rows, std::size_t cols) {
    // Dynamic shared memory size: one float slot per thread.
    const std::size_t shared_bytes = kSoftmaxThreadsPerBlock * sizeof(float);
    softmax_shared_memory_kernel<<<static_cast<unsigned int>(rows),
                                   kSoftmaxThreadsPerBlock, shared_bytes>>>(
        output, input, static_cast<int>(rows), static_cast<int>(cols));
}

template <int ELEMS_PER_THREAD>
__global__ void softmax_warpshfl_reg_cache_kernel(float* output,
                                                  const float* input,
                                                  int rows, int cols) {
    // Compile-time ELEMS_PER_THREAD keeps reg[] fixed-size, so the compiler can
    // place the cached float4 values in registers.
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

    // Register cache: each thread owns ELEMS_PER_THREAD float4 values.
    // For ELEMS_PER_THREAD=8, that is 8 float4 = 32 floats = 128 bytes/thread.
    float4 reg[ELEMS_PER_THREAD];

    // Pass 1: load input into registers and compute the local max. This is the
    // only global read made by this kernel.
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

    // Warp-shuffle max reduction: first reduce inside each warp, then reduce
    // the per-warp results with warp 0.
    max_value = warp_reduce_max(max_value);
    if (lane == 0) {
        warp_values[warp_id] = max_value;
    }
    __syncthreads();

    max_value =
        tid < warps_per_block ? warp_values[lane] : -INFINITY;
    if (warp_id == 0) {
        max_value = warp_reduce_max(max_value);
    }
    if (tid == 0) {
        warp_values[0] = max_value;
    }
    __syncthreads();
    max_value = warp_values[0];

    // Pass 2: compute exp from registers and accumulate the sum. This pass does
    // not touch global memory.
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

    // Warp-shuffle sum reduction, using shared memory only to exchange one
    // partial sum per warp.
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

    // Pass 3: normalize from registers and write output. This is the only
    // global write made by this kernel.
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

const char* to_string(SoftmaxAlgo algo) {
    switch (algo) {
        case SoftmaxAlgo::Naive:
            return "naive";
        case SoftmaxAlgo::SharedMemory:
            return "shared_memory";
        case SoftmaxAlgo::WarpShuffleRegCache:
            return "warp_shuffle_reg_cache";
    }
    return "unknown";
}

void softmax(const float* input, float* output, std::size_t rows,
             std::size_t cols, SoftmaxAlgo algo) {
    switch (algo) {
        case SoftmaxAlgo::Naive:
            launch_softmax_naive(input, output, rows, cols);
            return;
        case SoftmaxAlgo::SharedMemory:
            launch_softmax_shared_memory(input, output, rows, cols);
            return;
        case SoftmaxAlgo::WarpShuffleRegCache:
            launch_softmax_warpshfl_reg_cache(input, output, rows, cols);
            return;
    }
}

}  // namespace dl::kernels
