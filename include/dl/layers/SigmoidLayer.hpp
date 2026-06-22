#pragma once

#include "Layer.hpp"
#include "Layers.hpp"  // existing functional implementations


class SigmoidLayer : public Layer {
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

private:
    Eigen::MatrixXf cached_output;
};
