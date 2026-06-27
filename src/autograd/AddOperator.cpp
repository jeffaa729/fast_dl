#include <dl/autograd/AddOperator.hpp>

#include <stdexcept>

namespace dl::autograd {

std::vector<Tensor> AddOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("AddOperator backward requires one grad output");
    }

    return {grad_outputs[0], grad_outputs[0]};
}

}  // namespace dl::autograd
