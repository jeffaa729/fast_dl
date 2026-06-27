#include <dl/ops/Elementwise.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/elementwise.hpp>
#include <dl/ops/OpUtils.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace dl {

Tensor operator+(const Tensor& a, const Tensor& b) {
    return ops::add(a, b);
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    return ops::sub(a, b);
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    return ops::mul(a, b);
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    return ops::div(a, b);
}

}

namespace dl::ops {

namespace {

Tensor run_elementwise(const Tensor& a, const Tensor& b,
                       dl::kernels::ElementwiseOp kernel_op,
                       const char* op_name) {
    check_binary_float_cuda_op(a, b, op_name);
    Tensor output(a.shape(), a.dtype(), a.device());
    dl::cuda::check(cudaSetDevice(a.device().index), "cudaSetDevice failed");
    dl::kernels::elementwise(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(a.numel()),
        kernel_op);
    dl::cuda::check(cudaGetLastError(), (std::string(op_name) + " kernel launch failed").c_str());
    return output;
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
    return run_elementwise(a, b, dl::kernels::ElementwiseOp::Add, "add");
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return run_elementwise(a, b, dl::kernels::ElementwiseOp::Sub, "sub");
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return run_elementwise(a, b, dl::kernels::ElementwiseOp::Mul, "mul");
}

Tensor div(const Tensor& a, const Tensor& b) {
    return run_elementwise(a, b, dl::kernels::ElementwiseOp::Div, "div");
}

}  // namespace dl::ops
