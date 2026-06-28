#include <dl/kernels/transpose.hpp>

#include <cuda_runtime.h>

namespace {

__global__ void transpose_padding_kernel(const float* input, float* output,
                                         std::size_t size) {
    __shared__ float tile[32][33];
    const std::size_t x = 32 * blockIdx.x + threadIdx.x;
    const std::size_t y = 32 * blockIdx.y + threadIdx.y;
    if (x < size && y < size) {
        tile[threadIdx.y][threadIdx.x] = input[y * size + x];
    }
    __syncthreads();

    const std::size_t ox = 32 * blockIdx.y + threadIdx.x;
    const std::size_t oy = 32 * blockIdx.x + threadIdx.y;
    if (ox < size && oy < size) {
        output[oy * size + ox] = tile[threadIdx.x][threadIdx.y];
    }
}

void launch_transpose_padding(const float* input, float* output,
                              std::size_t size) {
    constexpr int tile_dim = 32;
    const dim3 threads(tile_dim, tile_dim);
    const dim3 blocks(
        static_cast<unsigned int>((size + tile_dim - 1) / tile_dim),
        static_cast<unsigned int>((size + tile_dim - 1) / tile_dim));
    transpose_padding_kernel<<<blocks, threads>>>(input, output, size);
}

}  // namespace

namespace dl::kernels {

void transpose(const float* input, float* output, std::size_t size) {
    launch_transpose_padding(input, output, size);
}

}  // namespace dl::kernels
