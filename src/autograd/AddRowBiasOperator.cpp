#include <dl/autograd/AddRowBiasOperator.hpp>

#include <stdexcept>

#include <cuda_runtime.h>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/bias.hpp>

namespace dl::autograd {

std::vector<Tensor> AddRowBiasOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("AddRowBiasOperator backward requires one grad output");
    }

    const std::vector<Tensor>& op_inputs = inputs();
    if (op_inputs.size() != 2) {
        throw std::runtime_error("AddRowBiasOperator backward requires two inputs");
    }

    const Tensor& input = op_inputs[0];
    const Tensor& bias = op_inputs[1];
    const Tensor& grad_output = grad_outputs[0];

    Tensor grad_bias(bias.shape(), bias.dtype(), bias.device());
    const int rows = static_cast<int>(input.shape()[0]);
    const int cols = static_cast<int>(input.shape()[1]);

    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::sum_rows(
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_bias.data()),
        rows,
        cols);
    dl::cuda::check(cudaGetLastError(), "sum rows kernel launch failed");

    return {grad_output, grad_bias};
}

}  // namespace dl::autograd
