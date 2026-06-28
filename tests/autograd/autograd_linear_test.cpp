#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1.0e-4f;
}

}  // namespace

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        const std::vector<float> input_host = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        };
        const std::vector<float> weight_host = {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
        };
        const std::vector<float> bias_host = {
            0.5f, 1.0f,
        };

        const std::vector<float> expected_input_grad = {
             3.0f,  7.0f, 11.0f,
             3.0f,  7.0f, 11.0f,
        };
        const std::vector<float> expected_weight_grad = {
             5.0f, 5.0f,
             7.0f, 7.0f,
             9.0f, 9.0f,
        };
        const std::vector<float> expected_bias_grad = {
             2.0f, 2.0f,
        };

        dl::Tensor input = dl::Tensor::from_host<float>(
            input_host,
            dl::Shape({2, 3}),
            device);
        dl::Tensor weight = dl::Tensor::from_host<float>(
            weight_host,
            dl::Shape({3, 2}),
            device);
        dl::Tensor bias = dl::Tensor::from_host<float>(
            bias_host,
            dl::Shape({2}),
            device);

        input.set_requires_grad(true);
        weight.set_requires_grad(true);
        bias.set_requires_grad(true);

        dl::Tensor output = dl::ops::linear(input, weight, bias);
        output.backward();

        const std::vector<float> input_grad = input.grad().to_host<float>();
        const std::vector<float> weight_grad = weight.grad().to_host<float>();
        const std::vector<float> bias_grad = bias.grad().to_host<float>();

        passed = output.requires_grad() &&
                 output.creator() != nullptr &&
                 input_grad.size() == expected_input_grad.size() &&
                 weight_grad.size() == expected_weight_grad.size() &&
                 bias_grad.size() == expected_bias_grad.size();

        for (std::size_t i = 0; i < input_grad.size(); ++i) {
            if (!close_enough(input_grad[i], expected_input_grad[i])) {
                passed = false;
                break;
            }
        }
        for (std::size_t i = 0; i < weight_grad.size(); ++i) {
            if (!close_enough(weight_grad[i], expected_weight_grad[i])) {
                passed = false;
                break;
            }
        }
        for (std::size_t i = 0; i < bias_grad.size(); ++i) {
            if (!close_enough(bias_grad[i], expected_bias_grad[i])) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "autograd_linear_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
