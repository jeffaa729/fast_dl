#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

std::vector<float> linear_cpu(const std::vector<float>& input,
                              const std::vector<float>& weight,
                              const std::vector<float>& bias,
                              std::size_t batch,
                              std::size_t in_features,
                              std::size_t out_features) {
    std::vector<float> output(batch * out_features, 0.0f);

    for (std::size_t row = 0; row < batch; ++row) {
        for (std::size_t col = 0; col < out_features; ++col) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < in_features; ++k) {
                sum += input[row * in_features + k] *
                       weight[k * out_features + col];
            }
            output[row * out_features + col] =
                sum + bias[row * out_features + col];
        }
    }

    return output;
}

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1e-4f;
}

}  // namespace

int main() {
    constexpr std::size_t batch = 2;
    constexpr std::size_t in_features = 3;
    constexpr std::size_t out_features = 2;

    const std::vector<float> input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> weight = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const std::vector<float> bias = {
        0.5f, 1.0f,
        1.5f, 2.0f,
    };

    dl::Tensor x = dl::Tensor::from_host<float>(
        input,
        dl::Shape({static_cast<int64_t>(batch), static_cast<int64_t>(in_features)}),
        dl::Device(dl::DeviceType::CUDA, 0));
    dl::Tensor w = dl::Tensor::from_host<float>(
        weight,
        dl::Shape({static_cast<int64_t>(in_features), static_cast<int64_t>(out_features)}),
        dl::Device(dl::DeviceType::CUDA, 0));
    dl::Tensor b = dl::Tensor::from_host<float>(
        bias,
        dl::Shape({static_cast<int64_t>(batch), static_cast<int64_t>(out_features)}),
        dl::Device(dl::DeviceType::CUDA, 0));

    dl::Tensor y = dl::ops::linear(x, w, b);
    const std::vector<float> output = y.to_host<float>();
    const std::vector<float> expected =
        linear_cpu(input, weight, bias, batch, in_features, out_features);

    bool passed = y.shape().rank() == 2 &&
                  y.shape()[0] == static_cast<int64_t>(batch) &&
                  y.shape()[1] == static_cast<int64_t>(out_features);
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!close_enough(output[i], expected[i])) {
            passed = false;
            break;
        }
    }

    std::cout << "linear_tensor_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
