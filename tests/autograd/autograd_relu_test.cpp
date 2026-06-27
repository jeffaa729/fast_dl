#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

}  // namespace

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        const std::vector<float> input_host = {
            -2.0f, 0.0f, 3.0f, 4.0f
        };
        const std::vector<float> expected_grad = {
            0.0f, 0.0f, 1.0f, 1.0f
        };

        dl::Tensor x = dl::Tensor::from_host<float>(
            input_host,
            dl::Shape({4}),
            device);
        x.set_requires_grad(true);

        dl::Tensor y = dl::ops::relu(x);
        y.backward();

        const std::vector<float> grad = x.grad().to_host<float>();

        passed = y.requires_grad() &&
                 y.creator() != nullptr &&
                 grad.size() == expected_grad.size();

        for (std::size_t i = 0; i < grad.size(); ++i) {
            if (!close_enough(grad[i], expected_grad[i])) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "autograd_relu_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
