#include <dl/ops/Activation.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/activation.hpp>

#include <cuda_runtime.h>

namespace dl::ops {

Tensor relu(const Tensor& input) {
    if (!input.defined()) {
        throw std::runtime_error("relu input tensor is not defined");
    }

    if (!input.device().is_cuda()) {
        throw std::runtime_error("relu currently supports CUDA tensors only");
    }

    Tensor output(input.shape(), input.dtype(), input.device());
    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::activation(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(input.numel()),
        dl::kernels::ActivationOp::ReLU);
    dl::cuda::check(cudaGetLastError(), "relu kernel launch failed");
    return output;
}

Tensor leaky_relu(const Tensor& input, float alpha) {
    if (!input.defined()) {
        throw std::runtime_error("leaky_relu input tensor is not defined");
    }

    if (!input.device().is_cuda()) {
        throw std::runtime_error("leaky_relu currently supports CUDA tensors only");
    }

    Tensor output(input.shape(), input.dtype(), input.device());
    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::activation(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(input.numel()),
        dl::kernels::ActivationOp::LeakyReLU,
        alpha);
    dl::cuda::check(cudaGetLastError(), "leaky_relu kernel launch failed");
    return output;
}

Tensor gelu(const Tensor& input) {
    if (!input.defined()) {
        throw std::runtime_error("gelu input tensor is not defined");
    }

    if (!input.device().is_cuda()) {
        throw std::runtime_error("gelu currently supports CUDA tensors only");
    }

    Tensor output(input.shape(), input.dtype(), input.device());
    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::activation(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(input.numel()),
        dl::kernels::ActivationOp::GELU);
    dl::cuda::check(cudaGetLastError(), "gelu kernel launch failed");
    return output;
}

Tensor sigmoid(const Tensor& input) {
    if (!input.defined()) {
        throw std::runtime_error("sigmoid input tensor is not defined");
    }

    if (!input.device().is_cuda()) {
        throw std::runtime_error("sigmoid currently supports CUDA tensors only");
    }

    Tensor output(input.shape(), input.dtype(), input.device());
    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::activation(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(input.numel()),
        dl::kernels::ActivationOp::Sigmoid);
    dl::cuda::check(cudaGetLastError(), "sigmoid kernel launch failed");
    return output;
}

Tensor tanh(const Tensor& input) {
    if (!input.defined()) {
        throw std::runtime_error("tanh input tensor is not defined");
    }

    if (!input.device().is_cuda()) {
        throw std::runtime_error("tanh currently supports CUDA tensors only");
    }

    Tensor output(input.shape(), input.dtype(), input.device());
    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::activation(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(input.numel()),
        dl::kernels::ActivationOp::Tanh);
    dl::cuda::check(cudaGetLastError(), "tanh kernel launch failed");
    return output;  
}

}  // namespace dl::ops