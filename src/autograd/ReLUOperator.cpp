#include <dl/autograd/ReLUOperator.hpp>

#include <stdexcept>

#include <cuda_runtime.h>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/activation.hpp>

namespace dl::autograd {

std::vector<Tensor> ReLUOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("ReLUOperator backward requires one grad output");
    }

    const std::vector<Tensor>& op_inputs = inputs();
    if (op_inputs.size() != 1) {
        throw std::runtime_error("ReLUOperator backward requires one input");
    }

    const Tensor& input = op_inputs[0];
    const Tensor& grad_output = grad_outputs[0];  // dL/doutput
    Tensor grad_input(input.shape(), input.dtype(), input.device());  // dL/dinput

    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::relu_backward(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_input.data()),
        static_cast<int>(input.numel()));
    dl::cuda::check(cudaGetLastError(), "relu backward kernel launch failed");

    return {
        grad_input,  // dL/dinput
    };
}

}  // namespace dl::autograd
