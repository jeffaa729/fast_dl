#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

int main() {
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    dl::Tensor x = dl::Tensor::from_host<float>(
        input,
        dl::Shape({4}),
        dl::Device(dl::DeviceType::CUDA, 0));

    dl::Tensor alias = x;
    const std::vector<float> output = alias.to_host<float>();

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
