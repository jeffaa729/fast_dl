#include <dl/ops/Elementwise.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/elementwise.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <cstdint>

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
Tensor add(const Tensor& a, const Tensor& b) {
    if (!a.defined() || !b.defined()) {
        throw std::runtime_error("add input tensors are not defined");
    }

    if (a.shape()[0] != b.shape()[0] || a.shape()[1] != b.shape()[1]) {
        throw std::runtime_error("add input tensors must have the same shape");
    }

    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("add input tensors must have the same dtype");
    }

    if (!a.device().is_cuda() || !b.device().is_cuda()) {
        throw std::runtime_error("add currently supports CUDA tensors only");
    }

    if (a.device().index != b.device().index) {
        throw std::runtime_error("add inputs must be on the same CUDA device");
    }

    Tensor output(a.shape(), a.dtype(), a.device());
    dl::cuda::check(cudaSetDevice(a.device().index), "cudaSetDevice failed");
    dl::kernels::elementwise(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(a.numel()),
        dl::kernels::ElementwiseOp::Add);
    dl::cuda::check(cudaGetLastError(), "add kernel launch failed");    
    return output;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    if (!a.defined() || !b.defined()) {
        throw std::runtime_error("sub input tensors are not defined");
    }

    if (a.shape()[0] != b.shape()[0] || a.shape()[1] != b.shape()[1]) {
        throw std::runtime_error("sub input tensors must have the same shape");
    }

    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("sub input tensors must have the same dtype");
    }

    if (!a.device().is_cuda() || !b.device().is_cuda()) {
        throw std::runtime_error("sub currently supports CUDA tensors only");
    }

    if (a.device().index != b.device().index) {
        throw std::runtime_error("sub inputs must be on the same CUDA device");
    }

    Tensor output(a.shape(), a.dtype(), a.device());
    dl::cuda::check(cudaSetDevice(a.device().index), "cudaSetDevice failed");
    dl::kernels::elementwise(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(a.numel()),
        dl::kernels::ElementwiseOp::Sub);
    dl::cuda::check(cudaGetLastError(), "sub kernel launch failed");    
    return output;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (!a.defined() || !b.defined()) {
        throw std::runtime_error("mul input tensors are not defined");
    }

    if (a.shape()[0] != b.shape()[0] || a.shape()[1] != b.shape()[1]) {
        throw std::runtime_error("mul input tensors must have the same shape");
    }

    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("mul input tensors must have the same dtype");
    }

    if (!a.device().is_cuda() || !b.device().is_cuda()) {
        throw std::runtime_error("mul currently supports CUDA tensors only");
    }

    if (a.device().index != b.device().index) {
        throw std::runtime_error("mul inputs must be on the same CUDA device");
    }

    Tensor output(a.shape(), a.dtype(), a.device());
    dl::cuda::check(cudaSetDevice(a.device().index), "cudaSetDevice failed");
    dl::kernels::elementwise(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(a.numel()),
        dl::kernels::ElementwiseOp::Mul);
    dl::cuda::check(cudaGetLastError(), "mul kernel launch failed");    
    return output;
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (!a.defined() || !b.defined()) {
        throw std::runtime_error("div input tensors are not defined");
    }

    if (a.shape()[0] != b.shape()[0] || a.shape()[1] != b.shape()[1]) {
        throw std::runtime_error("div input tensors must have the same shape");
    }

    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("div input tensors must have the same dtype");
    }

    if (!a.device().is_cuda() || !b.device().is_cuda()) {
        throw std::runtime_error("div currently supports CUDA tensors only");
    }

    if (a.device().index != b.device().index) {
        throw std::runtime_error("div inputs must be on the same CUDA device");
    }

    Tensor output(a.shape(), a.dtype(), a.device());
    dl::cuda::check(cudaSetDevice(a.device().index), "cudaSetDevice failed");
    dl::kernels::elementwise(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(a.numel()),
        dl::kernels::ElementwiseOp::Div);
    dl::cuda::check(cudaGetLastError(), "div kernel launch failed");    
    return output;
}

}