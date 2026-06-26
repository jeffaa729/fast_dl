#include <dl/ops/Softmax.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/softmax.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace dl::ops {

Tensor softmax(const Tensor& input) {
    
    if (!input.defined()) {
        throw std::runtime_error("softmax input tensor is not defined");
    }   

    if(input.shape().rank() != 2) {
        throw std::runtime_error("softmax input tensor must be 2D");
    }

    std::size_t rows = static_cast<std::size_t>(input.shape()[0]);
    std::size_t cols = static_cast<std::size_t>(input.shape()[1]);
    if (!input.defined()) {
        throw std::runtime_error("softmax input tensor is not defined");
    }

    if (!input.device().is_cuda()) {
        throw std::runtime_error("softmax currently supports CUDA tensors only");
    }

    if (input.dtype() != DType::Float32) {
        throw std::runtime_error("softmax currently supports Float32 tensors only");
    }

    if (rows == 0 || cols == 0) {
        throw std::runtime_error("softmax rows and cols must be non-zero");
    }

    if (input.numel() != static_cast<int64_t>(rows * cols)) {
        throw std::runtime_error("softmax shape does not match rows * cols");
    }

    Tensor output(input.shape(), input.dtype(), input.device());

    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::softmax(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        rows,
        cols,
        dl::kernels::SoftmaxAlgo::WarpShuffleRegCache);
    dl::cuda::check(cudaGetLastError(), "softmax kernel launch failed");

    return output;
}

}  // namespace dl::ops
