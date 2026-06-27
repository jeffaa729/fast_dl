#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

#include <dl/dl.hpp>

namespace {

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1.0e-4f;
}

std::vector<float> cross_entropy_grad_cpu(const std::vector<float>& logits,
                                          const std::vector<int64_t>& labels,
                                          std::size_t batch,
                                          std::size_t classes) {
    std::vector<float> grad(logits.size(), 0.0f);

    for (std::size_t row = 0; row < batch; ++row) {
        const std::size_t offset = row * classes;
        float max_logit = logits[offset];
        for (std::size_t col = 1; col < classes; ++col) {
            max_logit = std::max(max_logit, logits[offset + col]);
        }

        float sum_exp = 0.0f;
        for (std::size_t col = 0; col < classes; ++col) {
            sum_exp += std::exp(logits[offset + col] - max_logit);
        }

        for (std::size_t col = 0; col < classes; ++col) {
            const float softmax =
                std::exp(logits[offset + col] - max_logit) / sum_exp;
            const float target = col == static_cast<std::size_t>(labels[row])
                                     ? 1.0f
                                     : 0.0f;
            grad[offset + col] =
                (softmax - target) / static_cast<float>(batch);
        }
    }

    return grad;
}

}  // namespace

int main() {
    bool passed = true;

    try {
        constexpr std::size_t batch = 2;
        constexpr std::size_t classes = 3;
        const dl::Device device(dl::DeviceType::CUDA, 0);

        const std::vector<float> logits = {
            1.0f, 2.0f, 3.0f,
            1.0f, 3.0f, 2.0f,
        };
        const std::vector<int64_t> labels = {2, 1};

        dl::Tensor logits_tensor = dl::Tensor::from_host<float>(
            logits,
            dl::Shape({static_cast<int64_t>(batch), static_cast<int64_t>(classes)}),
            device);
        dl::Tensor labels_tensor = dl::Tensor::from_host<int64_t>(
            labels,
            dl::Shape({static_cast<int64_t>(batch)}),
            device);

        logits_tensor.set_requires_grad(true);

        dl::Tensor loss = dl::ops::cross_entropy(logits_tensor, labels_tensor);
        loss.backward();

        const std::vector<float> grad = logits_tensor.grad().to_host<float>();
        const std::vector<float> expected =
            cross_entropy_grad_cpu(logits, labels, batch, classes);

        passed = loss.requires_grad() &&
                 loss.creator() != nullptr &&
                 grad.size() == expected.size() &&
                 !labels_tensor.grad().defined();

        for (std::size_t i = 0; i < grad.size(); ++i) {
            if (!close_enough(grad[i], expected[i])) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "autograd_cross_entropy_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
