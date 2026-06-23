#include <dl/bench/benchmark.hpp>
#include <dl/bench/benchmarks.hpp>
#include <dl/bench/cuda_utils.cuh>
#include <dl/kernels/vector_add.hpp>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

CUDA_BENCH_NOINLINE void vector_add_cpu(const float* a, const float* b, float* c,
                                        std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace

namespace dl::bench {

int vector_add_benchmark(std::size_t size) {
    constexpr dl::kernels::VectorAddAlgo algo = dl::kernels::VectorAddAlgo::Naive;

    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> cpu_result(size);
    std::vector<float> gpu_result(size);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(a.begin(), a.end(), [&] { return distribution(generator); });
    std::generate(b.begin(), b.end(), [&] { return distribution(generator); });
    vector_add_cpu(a.data(), b.data(), cpu_result.data(), size);

    dl::bench::DeviceBuffer<float> device_a(size);
    dl::bench::DeviceBuffer<float> device_b(size);
    dl::bench::DeviceBuffer<float> device_c(size);
    device_a.copy_from_host(a.data());
    device_b.copy_from_host(b.data());

    dl::kernels::vector_add(device_a.data(), device_b.data(), device_c.data(), size,
                    algo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_c.copy_to_host(gpu_result.data());
    bool valid = true;
    for (std::size_t i = 0; i < size; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            valid = false;
            break;
        }
    }

    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace dl::bench
