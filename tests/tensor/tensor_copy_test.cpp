#include <cmath>
#include <iostream>
#include <vector>

#include <dl/core/Device.hpp>
#include <dl/core/DType.hpp>
#include <dl/core/Shape.hpp>
#include <dl/tensor/Tensor.hpp>

int main() {
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(input.size(), 0.0f);

    Tensor x(Shape({4}), DType::Float32, Device(DeviceType::CUDA, 0));
    x.copy_from_host(input.data(), input.size() * sizeof(float));
    x.copy_to_host(output.data(), output.size() * sizeof(float));

    bool passed = true;
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (std::fabs(input[i] - output[i]) > 1e-6f) {
            passed = false;
            break;
        }
    }

    std::cout << "tensor_copy_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
