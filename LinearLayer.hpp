#pragma once

#include "Layer.hpp"
#include "Layers.hpp"  // existing functional implementations

class LinearLayer : public Layer {
public:
    LinearLayer(int input_features, int output_features);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

    bool has_parameters() const override { return true; }
    std::vector<Eigen::MatrixXf*> parameters() override { return {&weight}; }
    std::vector<Eigen::MatrixXf*> parameter_grads() override { return {&grad_weight}; }
    std::vector<Eigen::VectorXf*> bias_parameters() override { return {&bias}; }
    std::vector<Eigen::VectorXf*> bias_grads() override { return {&grad_bias}; }

    // Accessors for testing
    Eigen::MatrixXf& weights() { return weight; }
    Eigen::VectorXf& biases() { return bias; }
    const Eigen::MatrixXf& grad_weights() const { return grad_weight; }
    const Eigen::VectorXf& grad_biases() const { return grad_bias; }

private:
    Eigen::MatrixXf weight;
    Eigen::VectorXf bias;
    Eigen::MatrixXf grad_weight;
    Eigen::VectorXf grad_bias;
    Eigen::MatrixXf cached_input;
};
