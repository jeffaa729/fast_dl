#include <dl/kernels/optimizer.hpp>

namespace {

__global__ void sgd_step_kernel(float* param, const float* grad,
                                float learning_rate, int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        param[idx] -= learning_rate * grad[idx];
    }
}

}  // namespace

namespace dl::kernels {

void sgd_step_float32(float* param, const float* grad, float learning_rate,
                      int64_t n) {
    if (n <= 0) {
        return;
    }

    constexpr int block_size = 256;
    const int blocks = static_cast<int>((n + block_size - 1) / block_size);
    sgd_step_kernel<<<blocks, block_size>>>(param, grad, learning_rate, n);
}

}  // namespace dl::kernels
