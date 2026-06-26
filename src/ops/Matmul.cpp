#include <dl/ops/Matmul.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/gemm.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace dl::ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (!a.defined() || !b.defined()) {
        throw std::runtime_error("matmul input tensors are not defined");
    }

    if (!a.device().is_cuda() || !b.device().is_cuda()) {
        throw std::runtime_error("matmul currently supports CUDA tensors only");
    }

    if (a.device().index != b.device().index) {
        throw std::runtime_error("matmul inputs must be on the same CUDA device");
    }

    if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
        throw std::runtime_error("matmul currently supports Float32 tensors only");
    }

    if (a.shape().rank() != 2 || b.shape().rank() != 2) {
        throw std::runtime_error("matmul input tensors must be 2D");
    }

    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("matmul input tensors have incompatible shapes");
    }

    const int m = static_cast<int>(a.shape()[0]);
    const int k = static_cast<int>(a.shape()[1]);
    const int n = static_cast<int>(b.shape()[1]);

    Tensor output(Shape({a.shape()[0], b.shape()[1]}), a.dtype(), a.device());
    dl::cuda::check(cudaSetDevice(a.device().index), "cudaSetDevice failed");
    dl::kernels::gemm(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        m,
        n,
        k,
        dl::kernels::GemmAlgo::Cublas);
    dl::cuda::check(cudaGetLastError(), "matmul kernel launch failed");
    return output;
}

}  // namespace dl::ops
