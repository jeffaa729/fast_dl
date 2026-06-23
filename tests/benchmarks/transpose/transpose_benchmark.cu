#include <dl/bench/benchmark.hpp>
#include <dl/bench/benchmarks.hpp>
#include <dl/bench/cuda_utils.cuh>
#include <dl/kernels/transpose.hpp>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// transpose : B = A^T
CUDA_BENCH_NOINLINE void transpose_cpu(const float* a, float* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = 0; j < size; ++j) {
            b[j * size + i] = a[i * size + j];
        }
    }
}

}  // namespace

namespace dl::bench {

namespace {

bool run_and_validate_transpose(dl::kernels::TransposeAlgo algo,
                                dl::bench::DeviceBuffer<float>& device_a,
                                dl::bench::DeviceBuffer<float>& device_b,
                                float* gpu_result, const float* cpu_result,
                                std::size_t n) {
    dl::kernels::transpose(device_a.data(), device_b.data(), n, algo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_b.copy_to_host(gpu_result);
    for (std::size_t i = 0; i < n * n; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            return false;
        }
    }

    return true;
}

}  // namespace

int transpose_benchmark(std::size_t n) {
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::vector<float> a(n * n);
    std::vector<float> cpu_result(n * n);
    std::vector<float> gpu_result(n * n);

    for (float& value : a) {
        value = distribution(generator);
    }
    transpose_cpu(a.data(), cpu_result.data(), n);

    dl::bench::DeviceBuffer<float> device_a(n * n);
    dl::bench::DeviceBuffer<float> device_b(n * n);
    device_a.copy_from_host(a.data());

    const bool naive_valid = run_and_validate_transpose(
        dl::kernels::TransposeAlgo::Naive, device_a, device_b, gpu_result.data(),
        cpu_result.data(), n);

    const bool shared_valid = run_and_validate_transpose(
        dl::kernels::TransposeAlgo::Shared, device_a, device_b, gpu_result.data(),
        cpu_result.data(), n);

    const bool padding_valid = run_and_validate_transpose(
        dl::kernels::TransposeAlgo::Padding, device_a, device_b, gpu_result.data(),
        cpu_result.data(), n);

    return naive_valid && shared_valid && padding_valid ? EXIT_SUCCESS
                                                        : EXIT_FAILURE;
}

}  // namespace dl::bench
