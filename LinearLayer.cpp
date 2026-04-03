#include "LinearLayer.hpp"
#include <random>

LinearLayer::LinearLayer(int input_features, int output_features) {
    // Simple random init with small stddev
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 0.01f);
    // Initialize the weight matrix and bias vector
    weight = Eigen::MatrixXf(output_features, input_features).unaryExpr([&](float) { return dist(gen); });
    bias = Eigen::VectorXf::Zero(output_features);

    grad_weight = Eigen::MatrixXf::Zero(output_features, input_features);
    grad_bias = Eigen::VectorXf::Zero(output_features);
}

Eigen::MatrixXf LinearLayer::forward(const Eigen::MatrixXf& input) {
    cached_input = input;
    return linear_forward(input, weight, bias);
}

Eigen::MatrixXf LinearLayer::backward(const Eigen::MatrixXf& grad_output) {
    Eigen::MatrixXf grad_input;
    linear_backward(grad_output, cached_input, weight, grad_input, grad_weight, grad_bias);
    return grad_input;
}
