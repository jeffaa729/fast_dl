#include <dl/dl.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1.0e-5f;
}

}  // namespace

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        const std::vector<float> values = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
        };

        dl::Tensor x = dl::Tensor::from_host<float>(
            values,
            dl::Shape({2, 2, 2}),
            device);
        x.set_requires_grad(true);

        dl::Tensor y = dl::ops::flatten(x);
        dl::Tensor grad_output = dl::Tensor::ones_like(y);
        y.backward(grad_output);

        dl::Tensor grad = x.grad();
        const std::vector<float> grad_host = grad.to_host<float>();

        passed = y.requires_grad() &&
                 y.shape().rank() == 2 &&
                 y.shape()[0] == 2 &&
                 y.shape()[1] == 4 &&
                 grad.defined() &&
                 grad.shape().rank() == 3 &&
                 grad.shape()[0] == 2 &&
                 grad.shape()[1] == 2 &&
                 grad.shape()[2] == 2 &&
                 grad_host.size() == values.size();

        for (float value : grad_host) {
            if (!close_enough(value, 1.0f)) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "autograd_flatten_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
