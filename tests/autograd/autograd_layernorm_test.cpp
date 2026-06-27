#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

std::vector<float> layernorm_backward_cpu(const std::vector<float>& input,
                                          const std::vector<float>& grad_output,
                                          std::size_t rows,
                                          std::size_t cols,
                                          float eps) {
    std::vector<float> grad_input(input.size(), 0.0f);

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * cols;

        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            const float value = input[offset + col];
            sum += value;
            sq_sum += value * value;
        }

        const float mean = sum / static_cast<float>(cols);
        const float mean_square = sq_sum / static_cast<float>(cols);
        const float variance = std::max(mean_square - mean * mean, 0.0f);
        const float inv_std = 1.0f / std::sqrt(variance + eps);

        float grad_sum = 0.0f;
        float grad_norm_sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            const float normalized = (input[offset + col] - mean) * inv_std;
            const float grad = grad_output[offset + col];
            grad_sum += grad;
            grad_norm_sum += grad * normalized;
        }

        const float mean_grad = grad_sum / static_cast<float>(cols);
        const float mean_grad_norm = grad_norm_sum / static_cast<float>(cols);
        for (std::size_t col = 0; col < cols; ++col) {
            const float normalized = (input[offset + col] - mean) * inv_std;
            grad_input[offset + col] =
                inv_std * (grad_output[offset + col] - mean_grad -
                           normalized * mean_grad_norm);
        }
    }

    return grad_input;
}

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1e-4f;
}

}  // namespace

int main() {
    bool passed = true;

    try {
        constexpr std::size_t rows = 2;
        constexpr std::size_t cols = 4;
        constexpr float eps = 1.0e-5f;

        const dl::Device device(dl::DeviceType::CUDA, 0);
        const std::vector<float> input_host = {
            1.0f, 2.0f, 3.0f, 4.0f,
            2.0f, 4.0f, 6.0f, 8.0f,
        };
        const std::vector<float> grad_output_host = {
            1.0f, 0.5f, -0.5f, 2.0f,
            -1.0f, 1.5f, 0.25f, 0.75f,
        };

        dl::Tensor x = dl::Tensor::from_host<float>(
            input_host,
            dl::Shape({static_cast<int64_t>(rows), static_cast<int64_t>(cols)}),
            device);
        x.set_requires_grad(true);

        dl::Tensor grad_output = dl::Tensor::from_host<float>(
            grad_output_host,
            dl::Shape({static_cast<int64_t>(rows), static_cast<int64_t>(cols)}),
            device);

        dl::Tensor y = dl::ops::layernorm(x, eps);
        y.backward(grad_output);

        const std::vector<float> grad = x.grad().to_host<float>();
        const std::vector<float> expected =
            layernorm_backward_cpu(input_host, grad_output_host, rows, cols, eps);

        passed = y.requires_grad() &&
                 y.creator() != nullptr &&
                 grad.size() == expected.size();

        for (std::size_t i = 0; i < grad.size(); ++i) {
            if (!close_enough(grad[i], expected[i])) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "autograd_layernorm_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
