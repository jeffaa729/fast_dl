#include "ParametricReLU.hpp"

Eigen::MatrixXf ParametricReLU::forward(const Eigen::MatrixXf& input) {
    cached_input  = input;
    cached_output = parametric_relu_forward(input, alpha);
    return cached_output;
}

Eigen::MatrixXf ParametricReLU::backward(const Eigen::MatrixXf& grad_output) {
    return parametric_relu_backward(grad_output, cached_input, alpha, grad_alpha);
}