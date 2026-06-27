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
        dl::nn::ReLU relu;

        const std::vector<float> input = {
            -2.0f, -0.5f, 0.0f,
             1.5f,  3.0f, -4.0f,
        };
        const std::vector<float> expected = {
            0.0f, 0.0f, 0.0f,
            1.5f, 3.0f, 0.0f,
        };

        dl::Tensor x = dl::Tensor::from_host<float>(
            input,
            dl::Shape({2, 3}),
            device);

        dl::Tensor y = relu(x);
        const std::vector<float> output = y.to_host<float>();

        passed = y.defined() &&
                 y.shape().rank() == 2 &&
                 y.shape()[0] == 2 &&
                 y.shape()[1] == 3 &&
                 y.dtype() == dl::DType::Float32 &&
                 y.device().is_cuda() &&
                 relu.parameters().empty();

        for (std::size_t i = 0; i < output.size(); ++i) {
            if (!close_enough(output[i], expected[i])) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "relu_module_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
