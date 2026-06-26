#include <dl/kernels/elementwise.hpp>

namespace {

__global__ void elementwise_kernel(const float* a, const float* b, float* c,
                                   int n, dl::kernels::ElementwiseOp op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        switch (op) {
            case dl::kernels::ElementwiseOp::Add:
                c[idx] = a[idx] + b[idx];
                break;
            case dl::kernels::ElementwiseOp::Sub:
                c[idx] = a[idx] - b[idx];
                break;
            case dl::kernels::ElementwiseOp::Mul:
                c[idx] = a[idx] * b[idx];
                break;
            case dl::kernels::ElementwiseOp::Div:
                c[idx] = a[idx] / b[idx];
                break;
        }
    }
}

void launch_elementwise_kernel(const float* a, const float* b, float* c,
                               int n, dl::kernels::ElementwiseOp op) {
    constexpr int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    elementwise_kernel<<<num_blocks, block_size>>>(a, b, c, n, op);
}

}  // namespace

namespace dl::kernels {

const char* to_string(ElementwiseOp op) {
    switch (op) {
        case ElementwiseOp::Add:
            return "add";
        case ElementwiseOp::Sub:
            return "sub";
        case ElementwiseOp::Mul:
            return "mul";
        case ElementwiseOp::Div:
            return "div";
    }
    return "unknown";
}

void elementwise(const float* a, const float* b, float* c, int n,
                 ElementwiseOp op) {
    launch_elementwise_kernel(a, b, c, n, op);
}

}  // namespace dl::kernels
