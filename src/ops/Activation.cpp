#include <dl/ops/Activation.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/activation.hpp>
#include <dl/ops/OpUtils.hpp>

#include <cuda_runtime.h>

namespace dl::ops {

namespace {

Tensor run_activation(const Tensor& input, dl::kernels::ActivationOp kernel_op,
                      const char* op_name, float alpha = 0.01f) {
    check_defined(input, op_name);
    check_float32(input, op_name);
    check_cuda(input, op_name);
    Tensor output(input.shape(), input.dtype(), input.device());
    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::activation(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(input.numel()),
        kernel_op,
        alpha);
    dl::cuda::check(cudaGetLastError(), "activation kernel launch failed");
    return output;
}

}  // namespace

Tensor relu(const Tensor& input) {
    return run_activation(input, dl::kernels::ActivationOp::ReLU, "relu");
}

Tensor leaky_relu(const Tensor& input, float alpha) {
    return run_activation(input, dl::kernels::ActivationOp::LeakyReLU,
                          "leaky_relu", alpha);
}

Tensor gelu(const Tensor& input) {
    return run_activation(input, dl::kernels::ActivationOp::GELU, "gelu");
}

Tensor sigmoid(const Tensor& input) {
    return run_activation(input, dl::kernels::ActivationOp::Sigmoid, "sigmoid");
}

Tensor tanh(const Tensor& input) {
    return run_activation(input, dl::kernels::ActivationOp::Tanh, "tanh");
}

}  // namespace dl::ops
