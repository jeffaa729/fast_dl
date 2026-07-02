#include <dl/autograd/DivOperator.hpp>

#include <stdexcept>

#include <dl/ops/Elementwise.hpp>

namespace dl::autograd {

std::vector<Tensor> DivOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("DivOperator backward requires one grad output");
    }

    const std::vector<Tensor>& op_inputs = inputs();
    if (op_inputs.size() != 2) {
        throw std::runtime_error("DivOperator backward requires two inputs");
    }

    const Tensor& grad_output = grad_outputs[0];  // dL/doutput
    const Tensor& a = op_inputs[0];
    const Tensor& b = op_inputs[1];

    Tensor grad_a = grad_output / b;  // dL/da
    Tensor grad_b = Tensor::zeros_like(grad_output) - ((grad_output * a) / (b * b));  // dL/db

    return {
        grad_a,  // dL/da
        grad_b,  // dL/db
    };
}

}  // namespace dl::autograd
