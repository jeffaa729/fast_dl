#pragma once

#include <Eigen/Dense>


// Combined Softmax + Cross Entropy loss.
// Assumes integer class labels in [0, num_classes).
class SoftmaxCrossEntropyLoss {
public:
    // Forward pass: logits (batch_size x num_classes), labels (batch_size)
    // Returns average loss over the batch.
    float forward(const Eigen::MatrixXf& logits, const Eigen::VectorXi& labels);

    // Backward pass: returns gradient w.r.t. logits (same shape as logits).
    // Uses cached probabilities and labels from the last forward pass.
    Eigen::MatrixXf backward() const;

    const Eigen::MatrixXf& probabilities() const { return probs_; }

private:
    Eigen::MatrixXf probs_;   // softmax probabilities
    Eigen::VectorXi labels_;  // ground-truth labels
};

