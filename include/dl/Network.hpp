#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "Layer.hpp"
#include "Optimizer.hpp"
#include "SoftmaxCrossEntropyLoss.hpp"

class Network {
public:
    Network() = default;

    void add_layer(std::unique_ptr<Layer> layer);

    void set_optimizer(Optimizer* opt) { optimizer_ = opt; }

    // Forward through all layers; caches last output.
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);

    // Compute loss using Softmax + Cross Entropy on the last output.
    float compute_loss(const Eigen::VectorXi& labels);

    // Backward pass: from loss back through all layers.
    void backward();

    // Update all layers' parameters using the optimizer.
    void update();

    Eigen::MatrixXf get_cached_logits() { return cached_logits_; };

private:
    std::vector<std::unique_ptr<Layer>> layers_;
    Optimizer* optimizer_ = nullptr;  // not owned
    SoftmaxCrossEntropyLoss loss_;
    Eigen::MatrixXf cached_logits_;
};

