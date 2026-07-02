#include <dl/autograd/AddOperator.hpp>

#include <stdexcept>

namespace dl::autograd {

std::vector<Tensor> AddOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("AddOperator backward requires one grad output");
    }

    const Tensor& grad_output = grad_outputs[0];  // dL/doutput
    return {
        grad_output,  // dL/dleft
        grad_output,  // dL/dright
    };
}

}  // namespace dl::autograd
