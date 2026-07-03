#include <dl/dl.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        dl::nn::Conv2D layer(3, 4, 3, device, 1, 1);

        const std::vector<float> input(
            static_cast<std::size_t>(2 * 3 * 5 * 5),
            1.0f);

        dl::Tensor x = dl::Tensor::from_host<float>(
            input,
            dl::Shape({2, 3, 5, 5}),
            device);

        dl::Tensor y = layer(x);
        y.backward();

        const std::vector<dl::Tensor*> params = layer.parameters();

        passed = y.defined() &&
                 y.requires_grad() &&
                 y.shape().rank() == 4 &&
                 y.shape()[0] == 2 &&
                 y.shape()[1] == 4 &&
                 y.shape()[2] == 5 &&
                 y.shape()[3] == 5 &&
                 params.size() == 2 &&
                 params[0] != nullptr &&
                 params[1] != nullptr &&
                 params[0]->shape().rank() == 4 &&
                 params[0]->shape()[0] == 4 &&
                 params[0]->shape()[1] == 3 &&
                 params[0]->shape()[2] == 3 &&
                 params[0]->shape()[3] == 3 &&
                 params[0]->requires_grad() &&
                 params[0]->grad().defined() &&
                 params[0]->grad().shape().rank() == 4 &&
                 params[1]->shape().rank() == 1 &&
                 params[1]->shape()[0] == 4 &&
                 params[1]->requires_grad() &&
                 params[1]->grad().defined() &&
                 params[1]->grad().shape().rank() == 1 &&
                 layer.num_parameters() == 112;
    } catch (...) {
        passed = false;
    }

    std::cout << "conv2d_module_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
