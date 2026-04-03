#include "SigmoidLayer.hpp"

Eigen::MatrixXf SigmoidLayer::forward(const Eigen::MatrixXf& input) {
    cached_output = sigmoid_forward(input);
    return cached_output;
}

Eigen::MatrixXf SigmoidLayer::backward(const Eigen::MatrixXf& grad_output) {
    return sigmoid_backward(grad_output, cached_output);
}
