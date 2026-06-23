#include <dl/bench/benchmark.hpp>
#include <dl/bench/benchmarks.hpp>
#include <dl/bench/cuda_utils.cuh>
#include <dl/kernels/gemm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

CUDA_BENCH_NOINLINE void gemm_cpu(const float* a, const float* b, float* c,
                                  std::size_t n) {
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < n; ++k) {
                sum += a[row * n + k] * b[k * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

}  // namespace

namespace dl::bench {

namespace {

bool run_and_validate_gemm(dl::kernels::GemmAlgo algo,
                           dl::bench::DeviceBuffer<float>& device_a,
                           dl::bench::DeviceBuffer<float>& device_b,
                           dl::bench::DeviceBuffer<float>& device_c,
                           float* gpu_result, const float* cpu_result,
                           std::size_t size, std::size_t n) {
    dl::kernels::gemm(device_a.data(), device_b.data(), device_c.data(),
              static_cast<int>(n), algo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_c.copy_to_host(gpu_result);
    for (std::size_t i = 0; i < size; ++i) {
        const float tolerance =
            1.0e-3f * std::max(1.0f, std::abs(cpu_result[i]));
        if (std::abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            return false;
        }
    }

    return true;
}

}  // namespace

int gemm_benchmark(std::size_t n) {
    const std::size_t size = n * n;

    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> cpu_result(size);
    std::vector<float> gpu_result(size);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(a.begin(), a.end(), [&] { return distribution(generator); });
    std::generate(b.begin(), b.end(), [&] { return distribution(generator); });

    gemm_cpu(a.data(), b.data(), cpu_result.data(), n);

    dl::bench::DeviceBuffer<float> device_a(size);
    dl::bench::DeviceBuffer<float> device_b(size);
    dl::bench::DeviceBuffer<float> device_c(size);
    device_a.copy_from_host(a.data());
    device_b.copy_from_host(b.data());

    const bool naive_valid = run_and_validate_gemm(
        dl::kernels::GemmAlgo::Naive, device_a, device_b, device_c, gpu_result.data(),
        cpu_result.data(), size, n);

    const bool tiled_valid = run_and_validate_gemm(
        dl::kernels::GemmAlgo::Tiled, device_a, device_b, device_c, gpu_result.data(),
        cpu_result.data(), size, n);

    const bool register_valid = run_and_validate_gemm(
        dl::kernels::GemmAlgo::Register, device_a, device_b, device_c,
        gpu_result.data(), cpu_result.data(), size, n);

    const bool cublas_valid = run_and_validate_gemm(
        dl::kernels::GemmAlgo::Cublas, device_a, device_b, device_c, gpu_result.data(),
        cpu_result.data(), size, n);

    return naive_valid && tiled_valid && register_valid && cublas_valid
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

}  // namespace dl::bench
