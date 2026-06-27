#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

std::vector<float> layernorm_cpu(const std::vector<float>& input,
                                 std::size_t rows,
                                 std::size_t cols,
                                 float eps) {
    std::vector<float> output(input.size(), 0.0f);

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

        for (std::size_t col = 0; col < cols; ++col) {
            output[offset + col] = (input[offset + col] - mean) * inv_std;
        }
    }

    return output;
}

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1e-4f;
}

}  // namespace

int main() {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 4;
    constexpr float eps = 1.0e-5f;

    const std::vector<float> input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 4.0f, 6.0f, 8.0f,
    };

    dl::Tensor x = dl::Tensor::from_host<float>(
        input,
        dl::Shape({static_cast<int64_t>(rows), static_cast<int64_t>(cols)}),
        dl::Device(dl::DeviceType::CUDA, 0));

    dl::Tensor y = dl::ops::layernorm(x, eps);
    const std::vector<float> output = y.to_host<float>();
    const std::vector<float> expected = layernorm_cpu(input, rows, cols, eps);

    bool passed = y.shape().rank() == 2 &&
                  y.shape()[0] == static_cast<int64_t>(rows) &&
                  y.shape()[1] == static_cast<int64_t>(cols);
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!close_enough(output[i], expected[i])) {
            passed = false;
            break;
        }
    }

    std::cout << "layernorm_tensor_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
