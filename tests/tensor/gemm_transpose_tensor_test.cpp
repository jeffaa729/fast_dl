#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include <dl/core/CudaUtils.hpp>
#include <dl/dl.hpp>
#include <dl/kernels/gemm.hpp>

namespace {

float read_a(const std::vector<float>& a, int row, int col,
             int m, int k, bool trans_a) {
    return trans_a ? a[col * m + row] : a[row * k + col];
}

float read_b(const std::vector<float>& b, int row, int col,
             int n, int k, bool trans_b) {
    return trans_b ? b[col * k + row] : b[row * n + col];
}

std::vector<float> gemm_cpu(const std::vector<float>& a,
                            const std::vector<float>& b,
                            int m, int n, int k,
                            bool trans_a, bool trans_b) {
    std::vector<float> c(static_cast<std::size_t>(m * n), 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += read_a(a, row, kk, m, k, trans_a) *
                       read_b(b, kk, col, n, k, trans_b);
            }
            c[row * n + col] = sum;
        }
    }
    return c;
}

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1.0e-4f;
}

bool run_case(bool trans_a, bool trans_b) {
    constexpr int m = 2;
    constexpr int n = 3;
    constexpr int k = 4;
    const dl::Device device(dl::DeviceType::CUDA, 0);

    const std::size_t a_size = static_cast<std::size_t>(trans_a ? k * m : m * k);
    const std::size_t b_size = static_cast<std::size_t>(trans_b ? n * k : k * n);

    std::vector<float> a(a_size);
    std::vector<float> b(b_size);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i + 1);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(i + 1) * 0.5f;
    }

    dl::Tensor ta = dl::Tensor::from_host<float>(
        a,
        dl::Shape({static_cast<int64_t>(a.size())}),
        device);
    dl::Tensor tb = dl::Tensor::from_host<float>(
        b,
        dl::Shape({static_cast<int64_t>(b.size())}),
        device);
    dl::Tensor tc(dl::Shape({m, n}), dl::DType::Float32, device);

    dl::kernels::gemm(
        static_cast<const float*>(ta.data()),
        static_cast<const float*>(tb.data()),
        static_cast<float*>(tc.data()),
        m,
        n,
        k,
        trans_a,
        trans_b);
    dl::cuda::check(cudaGetLastError(), "gemm transpose test launch failed");
    dl::cuda::check(cudaDeviceSynchronize(), "gemm transpose test synchronize failed");

    const std::vector<float> output = tc.to_host<float>();
    const std::vector<float> expected = gemm_cpu(a, b, m, n, k, trans_a, trans_b);

    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!close_enough(output[i], expected[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    bool passed = true;

    try {
        passed = run_case(false, false) &&
                 run_case(true, false) &&
                 run_case(false, true) &&
                 run_case(true, true);
    } catch (...) {
        passed = false;
    }

    std::cout << "gemm_transpose_tensor_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
