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
        dl::Tensor a = dl::Tensor::randn(
            dl::Shape({8}),
            dl::DType::Float32,
            dl::Device(dl::DeviceType::CUDA, 0),
            0.0f,
            1.0f,
            4321);
        dl::Tensor b = dl::Tensor::randn(
            dl::Shape({8}),
            dl::DType::Float32,
            dl::Device(dl::DeviceType::CUDA, 0),
            0.0f,
            1.0f,
            4321);

        const std::vector<float> a_host = a.to_host<float>();
        const std::vector<float> b_host = b.to_host<float>();

        bool any_nonzero = false;
        for (std::size_t i = 0; i < a_host.size(); ++i) {
            if (!close_enough(a_host[i], b_host[i])) {
                passed = false;
                break;
            }
            if (std::fabs(a_host[i]) > 1e-6f) {
                any_nonzero = true;
            }
        }

        passed = passed &&
                 any_nonzero &&
                 a.shape().rank() == 1 &&
                 a.shape()[0] == 8 &&
                 a.dtype() == dl::DType::Float32 &&
                 a.device().is_cuda();
    } catch (...) {
        passed = false;
    }

    std::cout << "tensor_randn_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
