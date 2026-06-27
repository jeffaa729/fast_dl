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

        dl::Tensor a = dl::Tensor::from_host<float>(
            {1.0f, 2.0f, 3.0f},
            dl::Shape({3}),
            device);
        dl::Tensor b = dl::Tensor::from_host<float>(
            {4.0f, 5.0f, 6.0f},
            dl::Shape({3}),
            device);

        a.set_requires_grad(true);
        b.set_requires_grad(true);

        dl::Tensor c = a - b;
        c.backward();

        const std::vector<float> grad_a = a.grad().to_host<float>();
        const std::vector<float> grad_b = b.grad().to_host<float>();

        passed = c.requires_grad() &&
                 c.creator() != nullptr &&
                 grad_a.size() == 3 &&
                 grad_b.size() == 3;

        for (std::size_t i = 0; i < grad_a.size(); ++i) {
            if (!close_enough(grad_a[i], 1.0f) ||
                !close_enough(grad_b[i], -1.0f)) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "autograd_sub_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
