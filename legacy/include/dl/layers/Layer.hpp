#pragma once

#include <Eigen/Dense>
#include <vector>

class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass: input -> output
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;

    // Backward pass: upstream grad -> grad w.r.t. input
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) = 0;

    // Parameter helpers (default: no parameters)
    virtual bool has_parameters() const { return false; }
    virtual std::vector<Eigen::MatrixXf*> parameters() { return {}; }
    virtual std::vector<Eigen::MatrixXf*> parameter_grads() { return {}; }
    virtual std::vector<Eigen::VectorXf*> bias_parameters() { return {}; }
    virtual std::vector<Eigen::VectorXf*> bias_grads() { return {}; }
};
