#include <dl/kernels/bias.hpp>

namespace {

__global__ void add_row_bias_kernel(float* output, const float* bias,
                                    int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = rows * cols;
    if (idx < n) {
        output[idx] += bias[idx % cols];
    }
}

void launch_add_row_bias_kernel(float* output, const float* bias,
                                int rows, int cols) {
    constexpr int block_size = 256;
    const int n = rows * cols;
    const int num_blocks = (n + block_size - 1) / block_size;
    add_row_bias_kernel<<<num_blocks, block_size>>>(output, bias, rows, cols);
}

}  // namespace

namespace dl::kernels {

void add_row_bias(float* output, const float* bias, int rows, int cols) {
    launch_add_row_bias_kernel(output, bias, rows, cols);
}

}  // namespace dl::kernels
