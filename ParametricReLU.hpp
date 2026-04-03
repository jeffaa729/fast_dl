#pragma once

#include "Layer.hpp"
#include "Layers.hpp"

class ParametricReLU : public Layer {
public:
    ParametricReLU(int features, float init_alpha = 0.01f)
        : alpha(features),
          grad_alpha(features)
    {
        alpha.setConstant(init_alpha);
        grad_alpha.setZero();
    }
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

private:
    Eigen::MatrixXf cached_output;
    Eigen::MatrixXf cached_input;
    Eigen::VectorXf alpha;
    Eigen::VectorXf grad_alpha;
};
