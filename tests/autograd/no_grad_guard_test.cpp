#include <iostream>
#include <vector>

#include <dl/autograd/GradMode.hpp>
#include <dl/dl.hpp>

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

        passed = dl::autograd::is_grad_enabled();

        dl::Tensor c;
        {
            dl::autograd::NoGradGuard guard;
            passed = passed && !dl::autograd::is_grad_enabled();
            c = a + b;
        }

        passed = passed &&
                 dl::autograd::is_grad_enabled() &&
                 c.defined() &&
                 !c.requires_grad() &&
                 c.creator() == nullptr;
    } catch (...) {
        passed = false;
    }

    std::cout << "no_grad_guard_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
