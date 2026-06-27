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
        const std::vector<float> a_host = {8.0f, 9.0f};
        const std::vector<float> b_host = {2.0f, 3.0f};
        const std::vector<float> expected_grad_a = {0.5f, 1.0f / 3.0f};
        const std::vector<float> expected_grad_b = {-2.0f, -1.0f};

        dl::Tensor a = dl::Tensor::from_host<float>(
            a_host,
            dl::Shape({2}),
            device);
        dl::Tensor b = dl::Tensor::from_host<float>(
            b_host,
            dl::Shape({2}),
            device);

        a.set_requires_grad(true);
        b.set_requires_grad(true);

        dl::Tensor c = a / b;
        c.backward();

        const std::vector<float> grad_a = a.grad().to_host<float>();
        const std::vector<float> grad_b = b.grad().to_host<float>();

        passed = c.requires_grad() &&
                 c.creator() != nullptr &&
                 grad_a.size() == expected_grad_a.size() &&
                 grad_b.size() == expected_grad_b.size();

        for (std::size_t i = 0; i < grad_a.size(); ++i) {
            if (!close_enough(grad_a[i], expected_grad_a[i]) ||
                !close_enough(grad_b[i], expected_grad_b[i])) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "autograd_div_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
