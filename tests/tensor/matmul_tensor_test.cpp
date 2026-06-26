#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

std::vector<float> matmul_cpu(const std::vector<float>& a,
                              const std::vector<float>& b,
                              std::size_t m,
                              std::size_t n,
                              std::size_t k) {
    std::vector<float> c(m * n, 0.0f);
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (std::size_t kk = 0; kk < k; ++kk) {
                sum += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    return c;
}

}  // namespace

int main() {
    constexpr std::size_t m = 2;
    constexpr std::size_t k = 3;
    constexpr std::size_t n = 4;

    const std::vector<float> a = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> b = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };

    dl::Tensor ta = dl::Tensor::from_host<float>(
        a,
        dl::Shape({static_cast<int64_t>(m), static_cast<int64_t>(k)}),
        dl::Device(dl::DeviceType::CUDA, 0));
    dl::Tensor tb = dl::Tensor::from_host<float>(
        b,
        dl::Shape({static_cast<int64_t>(k), static_cast<int64_t>(n)}),
        dl::Device(dl::DeviceType::CUDA, 0));

    dl::Tensor tc = dl::ops::matmul(ta, tb);

    const std::vector<float> output = tc.to_host<float>();

    const std::vector<float> expected = matmul_cpu(a, b, m, n, k);

    bool passed = tc.shape().rank() == 2 && tc.shape()[0] == static_cast<int64_t>(m) &&
                  tc.shape()[1] == static_cast<int64_t>(n);
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - expected[i]) > 1e-4f) {
            passed = false;
            break;
        }
    }

    std::cout << "matmul_tensor_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
