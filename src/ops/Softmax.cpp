#include <dl/ops/Softmax.hpp>

#include <dl/kernels/softmax.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " +
                                 cudaGetErrorString(status));
    }
}

}  // namespace

namespace dl::ops {

::Tensor softmax(const ::Tensor& input,
                 std::size_t rows,
                 std::size_t cols) {
    if (!input.defined()) {
        throw std::runtime_error("softmax input tensor is not defined");
    }

    if (!input.device().is_cuda()) {
        throw std::runtime_error("softmax currently supports CUDA tensors only");
    }

    if (input.dtype() != ::DType::Float32) {
        throw std::runtime_error("softmax currently supports Float32 tensors only");
    }

    if (rows == 0 || cols == 0) {
        throw std::runtime_error("softmax rows and cols must be non-zero");
    }

    if (input.numel() != static_cast<int64_t>(rows * cols)) {
        throw std::runtime_error("softmax shape does not match rows * cols");
    }

    ::Tensor output(input.shape(), input.dtype(), input.device());

    check_cuda(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::softmax(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        rows,
        cols,
        dl::kernels::SoftmaxAlgo::WarpShuffleRegCache);
    check_cuda(cudaGetLastError(), "softmax kernel launch failed");

    return output;
}

}  // namespace dl::ops
