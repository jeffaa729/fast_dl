#include <iostream>

#include <dl/core/Device.hpp>
#include <dl/core/DType.hpp>
#include <dl/core/Shape.hpp>
#include <dl/tensor/Tensor.hpp>

int main() {
    Tensor x(Shape({2, 3}), DType::Float32, Device(DeviceType::CUDA, 0));

    const bool passed = (x.numel() == 6 && x.nbytes() == 24);
    std::cout << "tensor_test : " << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
