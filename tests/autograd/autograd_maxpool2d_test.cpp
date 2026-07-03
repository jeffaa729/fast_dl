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
             1.0f,  2.0f,  3.0f,  4.0f,
             5.0f,  6.0f,  7.0f,  8.0f,
             9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
        };
        const std::vector<float> expected_grad_input = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
        };

        dl::Tensor x = dl::Tensor::from_host<float>(
            input,
            dl::Shape({1, 1, 4, 4}),
            device);
        x.set_requires_grad(true);

        dl::Tensor y = dl::ops::max_pool2d(x, 2, 2);
        y.backward();

        const std::vector<float> grad_input = x.grad().to_host<float>();

        passed = y.requires_grad() &&
                 y.creator() != nullptr &&
                 x.grad().defined() &&
                 x.grad().shape().rank() == 4 &&
                 x.grad().shape()[0] == 1 &&
                 x.grad().shape()[1] == 1 &&
                 x.grad().shape()[2] == 4 &&
                 x.grad().shape()[3] == 4 &&
                 check_vector(grad_input, expected_grad_input);
    } catch (...) {
        passed = false;
    }

    std::cout << "autograd_maxpool2d_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
