#pragma once
#include <Eigen/Dense>

class MSELoss {
public:
    MSELoss() = default;

    float forward(const Eigen::MatrixXf& pred,
                  const Eigen::MatrixXf& target) {
        pred_cache   = pred;
        target_cache = target;

        Eigen::MatrixXf diff = pred - target;
        float sum_sq = diff.array().square().sum();
        int batch_size = pred.rows();

        loss_cache = 0.5f * sum_sq / static_cast<float>(batch_size);
        return loss_cache;
    }

    Eigen::MatrixXf backward() const {
        int batch_size = pred_cache.rows();
        Eigen::MatrixXf diff = pred_cache - target_cache;
        return diff / static_cast<float>(batch_size);
    }

    float value() const { return loss_cache; }

private:
    Eigen::MatrixXf pred_cache;
    Eigen::MatrixXf target_cache;
    float loss_cache = 0.0f;
};