#pragma once

#include <Eigen/Dense>

/**
 * Linear Layer Forward Pass
 * Computes: output = input * weight^T + bias
 * 
 * @param input: Input matrix of shape (batch_size, input_features)
 * @param weight: Weight matrix of shape (output_features, input_features)
 * @param bias: Bias vector of shape (output_features,)
 * @return Output matrix of shape (batch_size, output_features)
 */
Eigen::MatrixXf linear_forward(const Eigen::MatrixXf& input,
                               const Eigen::MatrixXf& weight,
                               const Eigen::VectorXf& bias);

/**
 * Linear Layer Backward Pass
 * Computes gradients with respect to input, weight, and bias
 * 
 * @param grad_output: Gradient from next layer, shape (batch_size, output_features)
 * @param input: Original input, shape (batch_size, input_features)
 * @param weight: Weight matrix, shape (output_features, input_features)
 * @param grad_input: Output parameter for gradient w.r.t. input (modified in place)
 * @param grad_weight: Output parameter for gradient w.r.t. weight (modified in place)
 * @param grad_bias: Output parameter for gradient w.r.t. bias (modified in place)
 */
void linear_backward(const Eigen::MatrixXf& grad_output,
                     const Eigen::MatrixXf& input,
                     const Eigen::MatrixXf& weight,
                     Eigen::MatrixXf& grad_input,
                     Eigen::MatrixXf& grad_weight,
                     Eigen::VectorXf& grad_bias);

/**
 * Sigmoid Activation Forward Pass
 * Computes: output = 1 / (1 + exp(-input))
 * 
 * @param input: Input matrix
 * @return Output matrix with sigmoid applied element-wise
 */
Eigen::MatrixXf sigmoid_forward(const Eigen::MatrixXf& input);

/**
 * Sigmoid Activation Backward Pass
 * Computes: grad_input = grad_output * sigmoid(input) * (1 - sigmoid(input))
 * 
 * @param grad_output: Gradient from next layer
 * @param sigmoid_output: Output from sigmoid_forward (cached)
 * @return Gradient with respect to input
 */
Eigen::MatrixXf sigmoid_backward(const Eigen::MatrixXf& grad_output,
                                 const Eigen::MatrixXf& sigmoid_output);

Eigen::MatrixXf parametric_relu_forward(const Eigen::MatrixXf& input,
                                        const Eigen::VectorXf& alpha);

Eigen::MatrixXf parametric_relu_backward(const Eigen::MatrixXf& grad_output,
                                         const Eigen::MatrixXf& input,
                                         const Eigen::VectorXf& alpha,
                                         Eigen::VectorXf& grad_alpha);