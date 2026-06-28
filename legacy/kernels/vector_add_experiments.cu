#include <dl/kernels/vector_add.hpp>

#include <cuda_runtime.h>

namespace {

__global__ void vector_add_naive_kernel(const float* a, const float* b, float* c,
                                        std::size_t size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

void launch_vector_add_naive(const float* a, const float* b, float* c,
                             std::size_t size) {
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((size + threads_per_block - 1) /
                                        threads_per_block);
    vector_add_naive_kernel<<<blocks, threads_per_block>>>(a, b, c, size);
}

}  // namespace

namespace dl::kernels {

const char* to_string(VectorAddAlgo algo) {
    switch (algo) {
        case VectorAddAlgo::Naive:
            return "naive";
    }
    return "unknown";
}

void vector_add(const float* a, const float* b, float* c, std::size_t size,
                VectorAddAlgo algo) {
    switch (algo) {
        case VectorAddAlgo::Naive:
            launch_vector_add_naive(a, b, c, size);
            return;
    }
}

}  // namespace dl::kernels
