#include <dl/autograd/LayerNormOperator.hpp>

#include <stdexcept>

#include <cuda_runtime.h>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/layernorm.hpp>

namespace dl::autograd {

LayerNormOperator::LayerNormOperator(float eps) : eps_(eps) {}

std::vector<Tensor> LayerNormOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("LayerNormOperator backward requires one grad output");
    }

    const std::vector<Tensor>& op_inputs = inputs();
    if (op_inputs.size() != 1) {
        throw std::runtime_error("LayerNormOperator backward requires one input");
    }

    const Tensor& input = op_inputs[0];
    const Tensor& grad_output = grad_outputs[0];  // dL/doutput
    Tensor grad_input(input.shape(), input.dtype(), input.device());  // dL/dinput

    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::layernorm_backward(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_input.data()),
        static_cast<int>(input.shape()[0]),
        static_cast<int>(input.shape()[1]),
        eps_);
    dl::cuda::check(cudaGetLastError(), "layernorm backward kernel launch failed");

    return {
        grad_input,  // dL/dinput
    };
}

}  // namespace dl::autograd
