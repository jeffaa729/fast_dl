#include <dl/ops/Linear.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/bias.hpp>
#include <dl/ops/Matmul.hpp>
#include <dl/ops/OpUtils.hpp>

#include <cuda_runtime.h>

#include <stdexcept>

namespace dl::ops {
    
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    check_defined(input, "linear", "input");
    check_defined(weight, "linear", "weight");
    check_defined(bias, "linear", "bias");
    check_rank(input, 2, "linear", "input");
    check_rank(weight, 2, "linear", "weight");
    check_rank(bias, 1, "linear", "bias");
    check_float32(input, "linear", "input");
    check_same_dtype(input, weight, "linear", "input", "weight");
    check_same_dtype(input, bias, "linear", "input", "bias");
    check_cuda(input, "linear", "input");
    check_cuda(weight, "linear", "weight");
    check_cuda(bias, "linear", "bias");
    check_same_device(input, weight, "linear", "input", "weight");
    check_same_device(input, bias, "linear", "input", "bias");

    if (input.shape()[1] != weight.shape()[0]) {
        throw std::runtime_error("linear input and weight tensors have incompatible shapes");
    }

    if (weight.shape()[1] != bias.shape()[0]) {
        throw std::runtime_error("linear bias shape must be [out_features]");
    }

    Tensor output = matmul(input, weight);
    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::add_row_bias(
        static_cast<float*>(output.data()),
        static_cast<const float*>(bias.data()),
        static_cast<int>(output.shape()[0]),
        static_cast<int>(output.shape()[1]));
    dl::cuda::check(cudaGetLastError(), "linear bias kernel launch failed");
    return output;
}

}  // namespace dl::ops
