#include <dl/dl.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        dl::nn::MaxPool2D layer(2);

        const std::vector<float> input(
            static_cast<std::size_t>(2 * 3 * 32 * 32),
            1.0f);

        dl::Tensor x = dl::Tensor::from_host<float>(
            input,
            dl::Shape({2, 3, 32, 32}),
            device);
        x.set_requires_grad(true);

        dl::Tensor y = layer(x);
        y.backward();

        passed = y.defined() &&
                 y.requires_grad() &&
                 y.shape().rank() == 4 &&
                 y.shape()[0] == 2 &&
                 y.shape()[1] == 3 &&
                 y.shape()[2] == 16 &&
                 y.shape()[3] == 16 &&
                 layer.parameters().empty() &&
                 x.grad().defined() &&
                 x.grad().shape().rank() == 4;
    } catch (...) {
        passed = false;
    }

    std::cout << "maxpool2d_module_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
