// Layers.cpp
#include "Layers.hpp"
#include <cmath>

Eigen::MatrixXf linear_forward(const Eigen::MatrixXf& input,
                               const Eigen::MatrixXf& weight,
                               const Eigen::VectorXf& bias) {
    // output = input * weight^T + bias
    // input: (batch_size, input_features)
    // weight: (output_features, input_features)
    // bias: (output_features,)
    // output: (batch_size, output_features)
    
    Eigen::MatrixXf output = input * weight.transpose();
    
    // Add bias to each row (broadcasting)
    for (int i = 0; i < output.rows(); ++i) {
        output.row(i) += bias.transpose();
    }
    
    return output;
}

void linear_backward(const Eigen::MatrixXf& grad_output,
                     const Eigen::MatrixXf& input,
                     const Eigen::MatrixXf& weight,
                     Eigen::MatrixXf& grad_input,
                     Eigen::MatrixXf& grad_weight,
                     Eigen::VectorXf& grad_bias) {
    // grad_output: (batch_size, output_features)
    // input: (batch_size, input_features)
    // weight: (output_features, input_features)
    
    // Gradient w.r.t. input: grad_input = grad_output * weight
    // (batch_size, output_features) * (output_features, input_features) = (batch_size, input_features)
    grad_input = grad_output * weight;
    
    // Gradient w.r.t. weight: grad_weight = grad_output^T * input
    // (output_features, batch_size) * (batch_size, input_features) = (output_features, input_features)
    grad_weight = grad_output.transpose() * input;
    
    // Gradient w.r.t. bias: grad_bias = sum(grad_output, axis=0)
    // Sum over batch dimension
    grad_bias = grad_output.colwise().sum();
}

Eigen::MatrixXf sigmoid_forward(const Eigen::MatrixXf& input) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    // Apply element-wise
    return input.unaryExpr([](float x) {
        return 1.0f / (1.0f + std::exp(-x));
    });
}

Eigen::MatrixXf sigmoid_backward(const Eigen::MatrixXf& grad_output,
                                 const Eigen::MatrixXf& sigmoid_output) {
    // grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)
    // This is the derivative of sigmoid: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    return grad_output.array() * sigmoid_output.array() * (1.0f - sigmoid_output.array());
}


Eigen::MatrixXf parametric_relu_forward(const Eigen::MatrixXf& input,
                                        const Eigen::VectorXf& alpha) {
    Eigen::MatrixXf output = input;

    const int batch_size = input.rows();
    const int features   = input.cols();

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < features; ++j) {
            float x = input(i, j);
            output(i, j) = (x >= 0.0f) ? x : alpha(j) * x;
        }
    }
    return output;
}

Eigen::MatrixXf parametric_relu_backward(const Eigen::MatrixXf& grad_output,
                                         const Eigen::MatrixXf& input,
                                         const Eigen::VectorXf& alpha,
                                         Eigen::VectorXf& grad_alpha) {
    const int batch_size = input.rows();
    const int features   = input.cols();

    Eigen::MatrixXf grad_input = grad_output;
    grad_alpha.setZero();   // accumulate grad w.r.t. alpha over batch

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < features; ++j) {
            float x  = input(i, j);
            float go = grad_output(i, j);

            if (x >= 0.0f) {
                // y = x ⇒ dy/dx = 1, dy/dalpha = 0
                grad_input(i, j) = go;
            } else {
                // y = alpha_j * x
                // dy/dx = alpha_j, dy/dalpha_j = x
                grad_input(i, j) = go * alpha(j);
                grad_alpha(j)   += go * x;
            }
        }
    }
    return grad_input;
}
