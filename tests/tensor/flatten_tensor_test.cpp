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
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
        };

        dl::Tensor x = dl::Tensor::from_host<float>(
            values,
            dl::Shape({2, 3, 2}),
            device);
        dl::Tensor y = dl::ops::flatten(x);
        const std::vector<float> output = y.to_host<float>();

        passed = y.shape().rank() == 2 &&
                 y.shape()[0] == 2 &&
                 y.shape()[1] == 6 &&
                 y.dtype() == dl::DType::Float32 &&
                 y.device().is_cuda() &&
                 output.size() == values.size();

        for (std::size_t i = 0; i < values.size() && passed; ++i) {
            if (!close_enough(output[i], values[i])) {
                passed = false;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "flatten_tensor_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
