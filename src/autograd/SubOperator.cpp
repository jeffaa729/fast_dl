#include <dl/autograd/SubOperator.hpp>

#include <dl/ops/Elementwise.hpp>

namespace dl::autograd {

std::vector<Tensor> SubOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("SubOperator backward requires one grad output");
    }

    const Tensor& grad_output = grad_outputs[0];
    Tensor negative_grad = Tensor::zeros_like(grad_output) - grad_output;
    return {grad_output, negative_grad};
}

}  // namespace dl::autograd
