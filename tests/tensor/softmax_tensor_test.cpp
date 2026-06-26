#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

std::vector<float> softmax_cpu(const std::vector<float>& input,
                               std::size_t rows,
                               std::size_t cols) {
    std::vector<float> output(input.size(), 0.0f);

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * cols;
        float max_value = input[offset];
        for (std::size_t col = 1; col < cols; ++col) {
            max_value = std::max(max_value, input[offset + col]);
        }

        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            const float value = std::exp(input[offset + col] - max_value);
            output[offset + col] = value;
            sum += value;
        }

        for (std::size_t col = 0; col < cols; ++col) {
            output[offset + col] /= sum;
        }
    }

    return output;
}

}  // namespace

int main() {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 4;

    const std::vector<float> input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 3.0f, 2.0f, 0.0f,
    };

    dl::Tensor x = dl::Tensor::from_host<float>(
        input,
        dl::Shape({static_cast<int64_t>(rows), static_cast<int64_t>(cols)}),
        dl::Device(dl::DeviceType::CUDA, 0));

    dl::Tensor y = dl::ops::softmax(x);

    const std::vector<float> output = y.to_host<float>();

    const std::vector<float> expected = softmax_cpu(input, rows, cols);

    bool passed = true;
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - expected[i]) > 1e-4f) {
            passed = false;
            break;
        }
    }

    for (std::size_t row = 0; row < rows; ++row) {
        float row_sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            row_sum += output[row * cols + col];
        }
        if (std::fabs(row_sum - 1.0f) > 1e-4f) {
            passed = false;
            break;
        }
    }

    std::cout << "softmax_tensor_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
