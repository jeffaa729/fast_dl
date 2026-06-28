#include <dl/kernels/bias.hpp>

namespace {

__global__ void add_row_bias_kernel(const float* input, const float* bias,
                                    float* output, int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = rows * cols;
    if (idx < n) {
        output[idx] = input[idx] + bias[idx % cols];
    }
}

void launch_add_row_bias_kernel(const float* input, const float* bias,
                                float* output, int rows, int cols) {
    constexpr int block_size = 256;
    const int n = rows * cols;
    const int num_blocks = (n + block_size - 1) / block_size;
    add_row_bias_kernel<<<num_blocks, block_size>>>(input, bias, output, rows,
                                                    cols);
}

__global__ void sum_rows_kernel(const float* input, float* output,
                                int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

void launch_sum_rows_kernel(const float* input, float* output,
                            int rows, int cols) {
    constexpr int block_size = 256;
    const int num_blocks = (cols + block_size - 1) / block_size;
    sum_rows_kernel<<<num_blocks, block_size>>>(input, output, rows, cols);
}

}  // namespace

namespace dl::kernels {

void add_row_bias(const float* input, const float* bias, float* output,
                  int rows, int cols) {
    launch_add_row_bias_kernel(input, bias, output, rows, cols);
}

void sum_rows(const float* input, float* output, int rows, int cols) {
    launch_sum_rows_kernel(input, output, rows, cols);
}

}  // namespace dl::kernels
