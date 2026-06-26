#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

bool check_vector(const std::vector<float>& actual,
                  const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!close_enough(actual[i], expected[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const std::vector<float> a = {
        2.0f, 4.0f, 6.0f,
        8.0f, 10.0f, 12.0f,
    };
    const std::vector<float> b = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };

    dl::Tensor ta = dl::Tensor::from_host<float>(
        a,
        dl::Shape({2, 3}),
        dl::Device(dl::DeviceType::CUDA, 0));
    dl::Tensor tb = dl::Tensor::from_host<float>(
        b,
        dl::Shape({2, 3}),
        dl::Device(dl::DeviceType::CUDA, 0));

    const std::vector<float> expected_add = {
        3.0f, 6.0f, 9.0f,
        12.0f, 15.0f, 18.0f,
    };
    const std::vector<float> expected_sub = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> expected_mul = {
        2.0f, 8.0f, 18.0f,
        32.0f, 50.0f, 72.0f,
    };
    const std::vector<float> expected_div = {
        2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f,
    };

    bool passed = true;

    passed = passed && check_vector(dl::ops::add(ta, tb).to_host<float>(),
                                    expected_add);
    passed = passed && check_vector(dl::ops::sub(ta, tb).to_host<float>(),
                                    expected_sub);
    passed = passed && check_vector(dl::ops::mul(ta, tb).to_host<float>(),
                                    expected_mul);
    passed = passed && check_vector(dl::ops::div(ta, tb).to_host<float>(),
                                    expected_div);

    passed = passed && check_vector((ta + tb).to_host<float>(), expected_add);
    passed = passed && check_vector((ta - tb).to_host<float>(), expected_sub);
    passed = passed && check_vector((ta * tb).to_host<float>(), expected_mul);
    passed = passed && check_vector((ta / tb).to_host<float>(), expected_div);

    std::cout << "elementwise_tensor_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
