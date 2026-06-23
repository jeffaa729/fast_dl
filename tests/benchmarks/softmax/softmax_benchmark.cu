#include <dl/bench/benchmarks.hpp>
#include <dl/bench/cuda_utils.cuh>
#include <dl/kernels/softmax.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <random>
#include <vector>

namespace {

void softmax_cpu(const float* input, float* output, std::size_t rows,
                 std::size_t cols) {
    for (std::size_t row = 0; row < rows; ++row) {
        const float* row_input = input + row * cols;
        float* row_output = output + row * cols;

        float max_value = -FLT_MAX;
        for (std::size_t col = 0; col < cols; ++col) {
            max_value = std::max(max_value, row_input[col]);
        }

        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            const float value = std::exp(row_input[col] - max_value);
            row_output[col] = value;
            sum += value;
        }

        for (std::size_t col = 0; col < cols; ++col) {
            row_output[col] /= sum;
        }
    }
}

bool validate_softmax(const float* cpu_result, const float* gpu_result,
                      std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        const float tolerance =
            1.0e-5f * std::max(1.0f, std::abs(cpu_result[i]));
        if (std::abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

bool run_and_validate_softmax(dl::kernels::SoftmaxAlgo algo,
                              dl::bench::DeviceBuffer<float>& device_input,
                              dl::bench::DeviceBuffer<float>& device_output,
                              float* gpu_result, const float* cpu_result,
                              std::size_t rows, std::size_t cols) {
    dl::kernels::softmax(device_input.data(), device_output.data(), rows, cols, algo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_output.copy_to_host(gpu_result);
    return validate_softmax(cpu_result, gpu_result, rows * cols);
}

}  // namespace

namespace dl::bench {

int softmax_benchmark(std::size_t rows, std::size_t cols) {
    const std::size_t size = rows * cols;

    std::vector<float> input(size);
    std::vector<float> cpu_result(size);
    std::vector<float> gpu_result(size);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    std::generate(input.begin(), input.end(),
                  [&] { return distribution(generator); });

    softmax_cpu(input.data(), cpu_result.data(), rows, cols);

    dl::bench::DeviceBuffer<float> device_input(size);
    dl::bench::DeviceBuffer<float> device_output(size);
    device_input.copy_from_host(input.data());

    const bool naive_valid = run_and_validate_softmax(
        dl::kernels::SoftmaxAlgo::Naive, device_input, device_output, gpu_result.data(),
        cpu_result.data(), rows, cols);

    const bool shared_memory_valid = run_and_validate_softmax(
        dl::kernels::SoftmaxAlgo::SharedMemory, device_input, device_output,
        gpu_result.data(), cpu_result.data(), rows, cols);

    const bool reg_cache_valid = run_and_validate_softmax(
        dl::kernels::SoftmaxAlgo::WarpShuffleRegCache, device_input, device_output,
        gpu_result.data(), cpu_result.data(), rows, cols);

    return naive_valid && shared_memory_valid && reg_cache_valid ? EXIT_SUCCESS
                                                                 : EXIT_FAILURE;
}

}  // namespace dl::bench
