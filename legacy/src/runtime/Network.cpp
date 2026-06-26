#include "Network.hpp"

void Network::add_layer(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Eigen::MatrixXf Network::forward(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf x = input;
    for (auto& layer : layers_) {
        x = layer->forward(x);
    }
    cached_logits_ = x;
    return x;
}

float Network::compute_loss(const Eigen::VectorXi& labels) {
    return loss_.forward(cached_logits_, labels);
}

void Network::backward() {
    // Start from gradient of loss w.r.t logits
    Eigen::MatrixXf grad = loss_.backward();
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        grad = layers_[i]->backward(grad);
    }
}

// Update the weights of the layers using the optimizer
void Network::update() {
    if (!optimizer_) return;
    for (auto& layer : layers_) {
        optimizer_->step(*layer);
    }
}

