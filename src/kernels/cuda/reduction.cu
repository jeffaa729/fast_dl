#include <dl/kernels/reduction.hpp>

#include <cuda_runtime.h>

namespace {

__global__ void reduction_address_kernel(const float* input, float* output) {
    __shared__ float sdata[1024];
    const std::size_t tid = threadIdx.x;
    sdata[tid] = input[blockIdx.x * 1024 + tid];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < static_cast<std::size_t>(s)) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void launch_reduction_address(const float* input, float* output,
                              std::size_t size) {
    constexpr int threads_per_block = 1024;
    const int blocks = static_cast<int>(size / threads_per_block);
    reduction_address_kernel<<<blocks, threads_per_block>>>(input, output);
}

}  // namespace

namespace dl::kernels {

void reduction(const float* input, float* output, std::size_t size) {
    launch_reduction_address(input, output, size);
}

}  // namespace dl::kernels
