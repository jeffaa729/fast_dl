#include <dl/dl.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1.0e-5f;
}

bool check_vector(const std::vector<float>& actual,
                  const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!close_enough(actual[i], expected[i])) {
            return false;
        }
    }

    return true;
}

}  // namespace

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);

        {
            const std::vector<float> input = {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
            };
            const std::vector<float> weight = {
                1.0f, 0.0f,
                0.0f, 1.0f,
            };
            const std::vector<float> bias = {0.0f};
            const std::vector<float> expected = {
                6.0f, 8.0f,
                12.0f, 14.0f,
            };

            dl::Tensor x = dl::Tensor::from_host<float>(
                input,
                dl::Shape({1, 1, 3, 3}),
                device);
            dl::Tensor w = dl::Tensor::from_host<float>(
                weight,
                dl::Shape({1, 1, 2, 2}),
                device);
            dl::Tensor b = dl::Tensor::from_host<float>(
                bias,
                dl::Shape({1}),
                device);

            dl::Tensor y = dl::ops::conv2d(x, w, b);
            const std::vector<float> output = y.to_host<float>();

            passed = passed &&
                     y.shape().rank() == 4 &&
                     y.shape()[0] == 1 &&
                     y.shape()[1] == 1 &&
                     y.shape()[2] == 2 &&
                     y.shape()[3] == 2 &&
                     check_vector(output, expected);
        }

        {
            const std::vector<float> input = {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
            };
            const std::vector<float> weight = {
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
            };
            const std::vector<float> bias = {0.0f};
            const std::vector<float> expected = {
                12.0f, 21.0f, 16.0f,
                27.0f, 45.0f, 33.0f,
                24.0f, 39.0f, 28.0f,
            };

            dl::Tensor x = dl::Tensor::from_host<float>(
                input,
                dl::Shape({1, 1, 3, 3}),
                device);
            dl::Tensor w = dl::Tensor::from_host<float>(
                weight,
                dl::Shape({1, 1, 3, 3}),
                device);
            dl::Tensor b = dl::Tensor::from_host<float>(
                bias,
                dl::Shape({1}),
                device);

            dl::Tensor y = dl::ops::conv2d(x, w, b, 1, 1);
            const std::vector<float> output = y.to_host<float>();

            passed = passed &&
                     y.shape().rank() == 4 &&
                     y.shape()[0] == 1 &&
                     y.shape()[1] == 1 &&
                     y.shape()[2] == 3 &&
                     y.shape()[3] == 3 &&
                     check_vector(output, expected);
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "conv2d_tensor_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
