#include "SoftmaxCrossEntropyLoss.hpp"

#include <cmath>

float SoftmaxCrossEntropyLoss::forward(const Eigen::MatrixXf& logits,
                                       const Eigen::VectorXi& labels) {
    const int batch_size = logits.rows();
    const int num_classes = logits.cols();

    labels_ = labels;

    // Numerically stable softmax: subtract max per row
    Eigen::VectorXf row_max = logits.rowwise().maxCoeff();
    Eigen::MatrixXf shifted = logits;
    // Broadcast row_max to each row: subtract max from each row
    for (int i = 0; i < batch_size; ++i) {
        shifted.row(i).array() -= row_max(i);
    }

    Eigen::MatrixXf exp_scores = shifted.array().exp();
    Eigen::VectorXf exp_sums = exp_scores.rowwise().sum();

    probs_ = exp_scores.array().colwise() / exp_sums.array();

    // Cross-entropy loss: average over batch
    float loss = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        const int y = labels_(i);
        if (y < 0 || y >= num_classes) {
            continue; // ignore invalid label
        }
        float p = std::max(probs_(i, y), 1e-12f); // avoid log(0)
        loss -= std::log(p);
    }
    loss /= static_cast<float>(batch_size);
    return loss;
}

Eigen::MatrixXf SoftmaxCrossEntropyLoss::backward() const {
    const int batch_size = probs_.rows();
    const int num_classes = probs_.cols();

    Eigen::MatrixXf grad = probs_;
    for (int i = 0; i < batch_size; ++i) {
        int y = labels_(i);
        if (y >= 0 && y < num_classes) {
            grad(i, y) -= 1.0f;
        }
    }
    return grad / static_cast<float>(batch_size);
}

