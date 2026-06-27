#include <dl/kernels/transpose.hpp>

#include <cuda_runtime.h>

namespace {
// coalesced  read : good , strided write : bad
__global__ void transpose_naive_kernel(const float* input, float* output,
                                       std::size_t size) {
    const std::size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const std::size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < size && y < size) {
        output[x * size + y] = input[y * size + x];
    }
}

void launch_transpose_naive(const float* input, float* output,
                            std::size_t size) {
    constexpr int tile_dim = 32;
    const dim3 threads(tile_dim, tile_dim);
    const dim3 blocks(static_cast<unsigned int>((size + tile_dim - 1) / tile_dim),
                      static_cast<unsigned int>((size + tile_dim - 1) / tile_dim));
    transpose_naive_kernel<<<blocks, threads>>>(input, output, size);
}

// while global memory dont allow both row and col coalesced
// try use shared memeory , column read in -> row read out
__global__ void transpose_shared_kernel(const float* input, float* output,
                                       std::size_t size) {
    __shared__ float tile[32][32];
    const std::size_t x = 32 * blockIdx.x + threadIdx.x;
    const std::size_t y = 32 * blockIdx.y + threadIdx.y;
    if (x < size && y < size) {
        tile[threadIdx.y][threadIdx.x] = input[y * size + x]; // row read, row write shared
    }
    __syncthreads();

    int ox = 32 * blockIdx.y + threadIdx.x;
    int oy = 32 * blockIdx.x + threadIdx.y;
    if (ox < size && oy < size) {
        output[oy * size + ox] = tile[threadIdx.x][threadIdx.y]; // col read shared
    }
}

void launch_transpose_shared(const float* input, float* output,
                            std::size_t size) {
    constexpr int tile_dim = 32;
    const dim3 threads(tile_dim, tile_dim);
    const dim3 blocks(static_cast<unsigned int>((size + tile_dim - 1) / tile_dim),
                      static_cast<unsigned int>((size + tile_dim - 1) / tile_dim));
    transpose_shared_kernel<<<blocks, threads>>>(input, output, size);
}

__global__ void transpose_padding_kernel(const float* input, float* output,
                                       std::size_t size) {
    __shared__ float tile[32][33];
    const std::size_t x = 32 * blockIdx.x + threadIdx.x;
    const std::size_t y = 32 * blockIdx.y + threadIdx.y;
    if (x < size && y < size) {
        tile[threadIdx.y][threadIdx.x] = input[y * size + x]; // row read, row write shared
    }
    __syncthreads();

    int ox = 32 * blockIdx.y + threadIdx.x;
    int oy = 32 * blockIdx.x + threadIdx.y;
    if (ox < size && oy < size) {
        output[oy * size + ox] = tile[threadIdx.x][threadIdx.y]; // col read shared
    }
}

void launch_transpose_padding(const float* input, float* output,
                            std::size_t size) {
    constexpr int tile_dim = 32;
    const dim3 threads(tile_dim, tile_dim);
    const dim3 blocks(static_cast<unsigned int>((size + tile_dim - 1) / tile_dim),
                      static_cast<unsigned int>((size + tile_dim - 1) / tile_dim));
    transpose_padding_kernel<<<blocks, threads>>>(input, output, size);
}

}  // namespace

namespace dl::kernels {

const char* to_string(TransposeAlgo algo) {
    switch (algo) {
        case TransposeAlgo::Naive:
            return "naive";
        case TransposeAlgo::Shared:
            return "shared";
        case TransposeAlgo::Padding:
            return "padding";
    }
    return "unknown";
}

void transpose(const float* input, float* output, std::size_t size,
               TransposeAlgo algo) {
    switch (algo) {
        case TransposeAlgo::Naive:
            launch_transpose_naive(input, output, size);
            return;
        case TransposeAlgo::Shared:
            launch_transpose_shared(input, output, size);
            return;
        case TransposeAlgo::Padding:
            launch_transpose_padding(input, output, size);
            return;
    }
}

}  // namespace dl::kernels
