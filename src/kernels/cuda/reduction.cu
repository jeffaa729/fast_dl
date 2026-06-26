#include <dl/kernels/reduction.hpp>

#include <cuda_runtime.h>

namespace {

// bad kernel, demostrate warp divergence becoz of condition tid%2
__global__ void reduction_interleave_kernel(const float* input, float* output) {
    __shared__ float sdata[1024];
    const std::size_t tid = threadIdx.x;
    sdata[tid] = input[blockIdx.x * 1024 + tid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void launch_reduction_interleave(const float* input, float* output,
                                 std::size_t size) {
    constexpr int threads_per_block = 1024;
    const int blocks = static_cast<int>(size / threads_per_block);
    reduction_interleave_kernel<<<blocks, threads_per_block>>>(input, output);
}

// Sequential Addressing
__global__ void reduction_address_kernel(const float* input, float* output) {
    __shared__ float sdata[1024];
    const std::size_t tid = threadIdx.x;
    sdata[tid] = input[blockIdx.x * 1024 + tid];
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {                          // focus on low thread id
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

const char* to_string(ReductionAlgo algo) {
    switch (algo) {
        case ReductionAlgo::Interleave:
            return "interleave";
        case ReductionAlgo::Address:
            return "address";
    }
    return "unknown";
}

void reduction(const float* input, float* output, std::size_t size,
               ReductionAlgo algo) {
    switch (algo) {
        case ReductionAlgo::Interleave:
            launch_reduction_interleave(input, output, size);
            return;
        case ReductionAlgo::Address:
            launch_reduction_address(input, output, size);
            return;
    }
}

}  // namespace dl::kernels
