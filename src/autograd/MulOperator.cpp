#include <dl/autograd/MulOperator.hpp>

#include <stdexcept>

#include <dl/ops/Elementwise.hpp>

namespace dl::autograd {

std::vector<Tensor> MulOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("MulOperator backward requires one grad output");
    }

    const std::vector<Tensor>& op_inputs = inputs();
    if (op_inputs.size() != 2) {
        throw std::runtime_error("MulOperator backward requires two inputs");
    }

    const Tensor& grad_output = grad_outputs[0];
    return {
        grad_output * op_inputs[1],
        grad_output * op_inputs[0],
    };
}

}  // namespace dl::autograd
