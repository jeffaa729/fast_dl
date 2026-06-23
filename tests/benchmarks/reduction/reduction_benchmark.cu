#include <dl/bench/benchmark.hpp>
#include <dl/bench/benchmarks.hpp>
#include <dl/bench/cuda_utils.cuh>
#include <dl/kernels/reduction.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

CUDA_BENCH_NOINLINE void reduction_interleave_cpu(const float* input,
                                                  float* output,
                                                  std::size_t size) {
    constexpr std::size_t block_size = 1024;
    const std::size_t blocks = size / block_size;

    for (std::size_t block = 0; block < blocks; ++block) {
        std::array<float, block_size> data{};
        for (std::size_t i = 0; i < block_size; ++i) {
            data[i] = input[block * block_size + i];
        }

        for (std::size_t stride = 1; stride < block_size; stride *= 2) {
            for (std::size_t tid = 0; tid < block_size; tid += 2 * stride) {
                data[tid] += data[tid + stride];
            }
        }

        output[block] = data[0];
    }
}

CUDA_BENCH_NOINLINE void reduction_sequential_cpu(const float* input,
                                                  float* output,
                                                  std::size_t size) {
    constexpr std::size_t block_size = 1024;
    const std::size_t blocks = size / block_size;

    for (std::size_t block = 0; block < blocks; ++block) {
        std::array<float, block_size> data{};
        for (std::size_t i = 0; i < block_size; ++i) {
            data[i] = input[block * block_size + i];
        }

        for (std::size_t stride = block_size / 2; stride > 0; stride /= 2) {
            for (std::size_t tid = 0; tid < stride; ++tid) {
                data[tid] += data[tid + stride];
            }
        }

        output[block] = data[0];
    }
}

bool run_and_validate_reduction(dl::kernels::ReductionAlgo algo,
                                dl::bench::DeviceBuffer<float>& device_input,
                                dl::bench::DeviceBuffer<float>& device_output,
                                float* gpu_result, const float* cpu_result,
                                std::size_t output_size, std::size_t size) {
    dl::kernels::reduction(device_input.data(), device_output.data(), size, algo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_output.copy_to_host(gpu_result);
    for (std::size_t i = 0; i < output_size; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            return false;
        }
    }

    return true;
}

}  // namespace

namespace dl::bench {

int reduction_benchmark(std::size_t size) {
    constexpr std::size_t block_size = 1024;

    if (size % block_size != 0) {
        return EXIT_FAILURE;
    }

    const std::size_t output_size = size / block_size;
    std::vector<float> input(size);
    std::vector<float> interleave_cpu_result(output_size);
    std::vector<float> sequential_cpu_result(output_size);
    std::vector<float> gpu_result(output_size);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(input.begin(), input.end(),
                  [&] { return distribution(generator); });

    reduction_interleave_cpu(input.data(), interleave_cpu_result.data(), size);
    reduction_sequential_cpu(input.data(), sequential_cpu_result.data(), size);

    dl::bench::DeviceBuffer<float> device_input(size);
    dl::bench::DeviceBuffer<float> device_output(output_size);
    device_input.copy_from_host(input.data());

    const bool interleave_valid = run_and_validate_reduction(
        dl::kernels::ReductionAlgo::Interleave, device_input, device_output,
        gpu_result.data(), interleave_cpu_result.data(), output_size, size);

    const bool sequential_valid = run_and_validate_reduction(
        dl::kernels::ReductionAlgo::Address, device_input, device_output,
        gpu_result.data(), sequential_cpu_result.data(), output_size, size);

    return interleave_valid && sequential_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace dl::bench
