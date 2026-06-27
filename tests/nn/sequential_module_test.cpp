#include <iostream>
#include <memory>
#include <vector>

#include <dl/dl.hpp>

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);

        dl::nn::Sequential model;
        model.add(std::make_unique<dl::nn::Linear>(3, 4, device));
        model.add(std::make_unique<dl::nn::ReLU>());
        model.add(std::make_unique<dl::nn::Linear>(4, 2, device));

        const std::vector<float> input = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        };

        dl::Tensor x = dl::Tensor::from_host<float>(
            input,
            dl::Shape({2, 3}),
            device);

        dl::Tensor y = model(x);
        const std::vector<dl::Tensor*> params = model.parameters();

        passed = y.defined() &&
                 y.shape().rank() == 2 &&
                 y.shape()[0] == 2 &&
                 y.shape()[1] == 2 &&
                 y.dtype() == dl::DType::Float32 &&
                 y.device().is_cuda() &&
                 params.size() == 4 &&
                 params[0] != nullptr &&
                 params[1] != nullptr &&
                 params[2] != nullptr &&
                 params[3] != nullptr &&
                 params[0]->shape().rank() == 2 &&
                 params[0]->shape()[0] == 3 &&
                 params[0]->shape()[1] == 4 &&
                 params[1]->shape().rank() == 1 &&
                 params[1]->shape()[0] == 4 &&
                 params[2]->shape().rank() == 2 &&
                 params[2]->shape()[0] == 4 &&
                 params[2]->shape()[1] == 2 &&
                 params[3]->shape().rank() == 1 &&
                 params[3]->shape()[0] == 2;
    } catch (...) {
        passed = false;
    }

    std::cout << "sequential_module_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
