#include <dl/ops/Linear.hpp>

#include <dl/ops/Elementwise.hpp>
#include <dl/ops/Matmul.hpp>
#include <dl/ops/OpUtils.hpp>

#include <stdexcept>

namespace dl::ops {
    
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    check_defined(input, "linear", "input");
    check_defined(weight, "linear", "weight");
    check_defined(bias, "linear", "bias");
    check_rank(input, 2, "linear", "input");
    check_rank(weight, 2, "linear", "weight");
    check_rank(bias, 2, "linear", "bias");
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

    if (input.shape()[0] != bias.shape()[0] || weight.shape()[1] != bias.shape()[1]) {
        throw std::runtime_error("linear bias shape must match output shape");
    }

    return matmul(input, weight) + bias;
}

}  // namespace dl::ops
