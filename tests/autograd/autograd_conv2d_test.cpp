#include <dl/dl.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1.0e-5f;
}

bool check_vector(const std::vector<float>& actual,
                  const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!close_enough(actual[i], expected[i])) {
            return false;
        }
    }

    return true;
}

}  // namespace

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        const std::vector<float> input = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
        };
        const std::vector<float> weight = {
            1.0f, 0.0f,
            0.0f, 1.0f,
        };
        const std::vector<float> bias = {0.0f};

        const std::vector<float> expected_grad_input = {
            1.0f, 1.0f, 0.0f,
            1.0f, 2.0f, 1.0f,
            0.0f, 1.0f, 1.0f,
        };
        const std::vector<float> expected_grad_weight = {
            12.0f, 16.0f,
            24.0f, 28.0f,
        };
        const std::vector<float> expected_grad_bias = {4.0f};

        dl::Tensor x = dl::Tensor::from_host<float>(
            input,
            dl::Shape({1, 1, 3, 3}),
            device);
        dl::Tensor w = dl::Tensor::from_host<float>(
            weight,
            dl::Shape({1, 1, 2, 2}),
            device);
        dl::Tensor b = dl::Tensor::from_host<float>(
            bias,
            dl::Shape({1}),
            device);

        x.set_requires_grad(true);
        w.set_requires_grad(true);
        b.set_requires_grad(true);

        dl::Tensor y = dl::ops::conv2d(x, w, b);
        y.backward();

        const std::vector<float> grad_input = x.grad().to_host<float>();
        const std::vector<float> grad_weight = w.grad().to_host<float>();
        const std::vector<float> grad_bias = b.grad().to_host<float>();

        passed = y.requires_grad() &&
                 y.creator() != nullptr &&
                 check_vector(grad_input, expected_grad_input) &&
                 check_vector(grad_weight, expected_grad_weight) &&
                 check_vector(grad_bias, expected_grad_bias);
    } catch (...) {
        passed = false;
    }

    std::cout << "autograd_conv2d_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
