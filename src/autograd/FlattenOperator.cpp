#include <dl/autograd/FlattenOperator.hpp>

#include <stdexcept>

namespace dl::autograd {

FlattenOperator::FlattenOperator(const Shape& input_shape)
    : input_shape_(input_shape) {}

std::vector<Tensor> FlattenOperator::backward(const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("FlattenOperator backward requires one grad output");
    }
    const Tensor& grad_output = grad_outputs[0];  // dL/dflattened_output
    Tensor grad_input(input_shape_, grad_output.dtype(), grad_output.device());  // dL/doriginal_input
    grad_input.copy_from(grad_output);

    return {
        grad_input,  // dL/dinput
    };
}

}  // namespace dl::autograd
