#include <dl/kernels/random.hpp>

#include <curand_kernel.h>

namespace {

__global__ void uniform_kernel(float* data, int64_t n, float low, float high, uint64_t seed) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        float r = curand_uniform(&state); // (0, 1]
        data[idx] = low + r * (high - low);
    }
}

__global__ void normal_kernel(float* data, int64_t n, float mean, float stddev, uint64_t seed) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        data[idx] = mean + stddev * curand_normal(&state);
    }
}

}  // namespace

namespace dl::kernels {

void uniform_float32(float* data, int64_t n, float low, float high, uint64_t seed) {
    if (n <= 0) {
        return;
    }

    constexpr int block_size = 256;
    const int blocks = static_cast<int>((n + block_size - 1) / block_size);

    uniform_kernel<<<blocks, block_size>>>(data, n, low, high, seed);
}

void normal_float32(float* data, int64_t n, float mean, float stddev, uint64_t seed) {
    if (n <= 0) {
        return;
    }

    constexpr int block_size = 256;
    const int blocks = static_cast<int>((n + block_size - 1) / block_size);

    normal_kernel<<<blocks, block_size>>>(data, n, mean, stddev, seed);
}

}  // namespace dl::kernels
