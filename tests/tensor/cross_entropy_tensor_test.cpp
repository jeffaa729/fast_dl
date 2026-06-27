#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

float cross_entropy_cpu(const std::vector<float>& logits,
                        const std::vector<int64_t>& labels,
                        std::size_t batch,
                        std::size_t classes) {
    float loss = 0.0f;

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

        loss += std::log(sum_exp) + max_logit -
                logits[offset + static_cast<std::size_t>(labels[row])];
    }

    return loss / static_cast<float>(batch);
}

}  // namespace

int main() {
    constexpr std::size_t batch = 2;
    constexpr std::size_t classes = 3;

    const std::vector<float> logits = {
        1.0f, 2.0f, 3.0f,
        1.0f, 3.0f, 2.0f,
    };
    const std::vector<int64_t> labels = {2, 1};

    dl::Tensor logits_tensor = dl::Tensor::from_host<float>(
        logits,
        dl::Shape({static_cast<int64_t>(batch), static_cast<int64_t>(classes)}),
        dl::Device(dl::DeviceType::CUDA, 0));
    dl::Tensor labels_tensor = dl::Tensor::from_host<int64_t>(
        labels,
        dl::Shape({static_cast<int64_t>(batch)}),
        dl::Device(dl::DeviceType::CUDA, 0));

    dl::Tensor loss = dl::ops::cross_entropy(logits_tensor, labels_tensor);
    const std::vector<float> output = loss.to_host<float>();
    const float expected = cross_entropy_cpu(logits, labels, batch, classes);

    const bool passed = loss.shape().rank() == 1 && loss.shape()[0] == 1 &&
                        std::fabs(output[0] - expected) < 1e-4f;

    std::cout << "cross_entropy_tensor_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
