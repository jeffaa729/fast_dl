#include <cuda_runtime.h>

#include <dl/autograd/GradMode.hpp>
#include <dl/core/CudaUtils.hpp>
#include <dl/dl.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Result {
    std::string op;
    std::string shape;
    int iterations = 0;
    int warmup = 0;
    float total_ms = 0.0f;
    float avg_ms = 0.0f;
};

int env_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }

    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : default_value;
}

std::string env_string(const char* name, const std::string& default_value) {
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : std::string(value);
}

template <typename Fn>
float time_cuda_ms(int warmup, int iterations, Fn fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    dl::cuda::check(cudaDeviceSynchronize(), "benchmark warmup synchronize failed");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    dl::cuda::check(cudaEventCreate(&start), "cudaEventCreate start failed");
    dl::cuda::check(cudaEventCreate(&stop), "cudaEventCreate stop failed");

    dl::cuda::check(cudaEventRecord(start), "cudaEventRecord start failed");
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    dl::cuda::check(cudaEventRecord(stop), "cudaEventRecord stop failed");
    dl::cuda::check(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

    float total_ms = 0.0f;
    dl::cuda::check(cudaEventElapsedTime(&total_ms, start, stop),
                    "cudaEventElapsedTime failed");

    dl::cuda::check(cudaEventDestroy(start), "cudaEventDestroy start failed");
    dl::cuda::check(cudaEventDestroy(stop), "cudaEventDestroy stop failed");
    return total_ms;
}

Result make_result(const std::string& op,
                   const std::string& shape,
                   int warmup,
                   int iterations,
                   float total_ms) {
    return {
        op,
        shape,
        iterations,
        warmup,
        total_ms,
        total_ms / static_cast<float>(iterations),
    };
}

dl::Tensor randn(const dl::Shape& shape, const dl::Device& device,
                 uint64_t seed) {
    return dl::Tensor::randn(shape, dl::DType::Float32, device, 0.0f, 1.0f,
                             seed);
}

Result benchmark_matmul(const dl::Device& device, int warmup, int iterations) {
    constexpr int m = 1024;
    constexpr int n = 1024;
    constexpr int k = 1024;

    dl::Tensor a = randn(dl::Shape({m, k}), device, 100);
    dl::Tensor b = randn(dl::Shape({k, n}), device, 101);

    const float total_ms = time_cuda_ms(warmup, iterations, [&] {
        dl::Tensor y = dl::ops::matmul(a, b);
        (void)y.data();
    });

    return make_result("matmul", "1024x1024x1024", warmup, iterations,
                       total_ms);
}

Result benchmark_linear(const dl::Device& device, int warmup, int iterations) {
    constexpr int batch = 1024;
    constexpr int in_features = 784;
    constexpr int out_features = 128;

    dl::Tensor x = randn(dl::Shape({batch, in_features}), device, 200);
    dl::Tensor weight = randn(dl::Shape({in_features, out_features}), device, 201);
    dl::Tensor bias = dl::Tensor::zeros(dl::Shape({out_features}),
                                        dl::DType::Float32, device);

    const float total_ms = time_cuda_ms(warmup, iterations, [&] {
        dl::Tensor y = dl::ops::linear(x, weight, bias);
        (void)y.data();
    });

    return make_result("linear", "1024x784x128", warmup, iterations, total_ms);
}

template <typename Fn>
Result benchmark_unary(const std::string& op,
                       const dl::Device& device,
                       int warmup,
                       int iterations,
                       uint64_t seed,
                       Fn fn) {
    constexpr int64_t elements = 1 << 20;
    dl::Tensor x = randn(dl::Shape({elements}), device, seed);

    const float total_ms = time_cuda_ms(warmup, iterations, [&] {
        dl::Tensor y = fn(x);
        (void)y.data();
    });

    return make_result(op, "1048576", warmup, iterations, total_ms);
}

template <typename Fn>
Result benchmark_binary(const std::string& op,
                        const dl::Device& device,
                        int warmup,
                        int iterations,
                        uint64_t seed,
                        Fn fn) {
    constexpr int64_t elements = 1 << 20;
    dl::Tensor a = randn(dl::Shape({elements}), device, seed);
    dl::Tensor b = randn(dl::Shape({elements}), device, seed + 1);

    const float total_ms = time_cuda_ms(warmup, iterations, [&] {
        dl::Tensor y = fn(a, b);
        (void)y.data();
    });

    return make_result(op, "1048576", warmup, iterations, total_ms);
}

Result benchmark_softmax(const dl::Device& device, int warmup, int iterations) {
    constexpr int rows = 4096;
    constexpr int cols = 1024;
    dl::Tensor x = randn(dl::Shape({rows, cols}), device, 400);

    const float total_ms = time_cuda_ms(warmup, iterations, [&] {
        dl::Tensor y = dl::ops::softmax(x);
        (void)y.data();
    });

    return make_result("softmax", "4096x1024", warmup, iterations, total_ms);
}

Result benchmark_layernorm(const dl::Device& device, int warmup, int iterations) {
    constexpr int rows = 4096;
    constexpr int cols = 1024;
    dl::Tensor x = randn(dl::Shape({rows, cols}), device, 500);

    const float total_ms = time_cuda_ms(warmup, iterations, [&] {
        dl::Tensor y = dl::ops::layernorm(x);
        (void)y.data();
    });

    return make_result("layernorm", "4096x1024", warmup, iterations, total_ms);
}

Result benchmark_cross_entropy(const dl::Device& device,
                               int warmup,
                               int iterations) {
    constexpr int batch = 4096;
    constexpr int classes = 1000;
    dl::Tensor logits = randn(dl::Shape({batch, classes}), device, 600);

    std::vector<int64_t> labels(static_cast<std::size_t>(batch));
    for (int i = 0; i < batch; ++i) {
        labels[static_cast<std::size_t>(i)] = i % classes;
    }
    dl::Tensor y = dl::Tensor::from_host<int64_t>(
        labels, dl::Shape({batch}), device);

    const float total_ms = time_cuda_ms(warmup, iterations, [&] {
        dl::Tensor loss = dl::ops::cross_entropy(logits, y);
        (void)loss.data();
    });

    return make_result("cross_entropy", "4096x1000", warmup, iterations,
                       total_ms);
}

Result benchmark_conv2d_3x3(const dl::Device& device,
                            int warmup,
                            int iterations) {
    constexpr int batch = 64;
    constexpr int c_in = 16;
    constexpr int height = 16;
    constexpr int width = 16;
    constexpr int c_out = 32;
    constexpr int kernel = 3;

    dl::Tensor x = randn(dl::Shape({batch, c_in, height, width}), device, 700);
    dl::Tensor weight = randn(
        dl::Shape({c_out, c_in, kernel, kernel}), device, 701);
    dl::Tensor bias = dl::Tensor::zeros(
        dl::Shape({c_out}), dl::DType::Float32, device);

    const float total_ms = time_cuda_ms(warmup, iterations, [&] {
        dl::Tensor y = dl::ops::conv2d(x, weight, bias, 1, 1);
        (void)y.data();
    });

    return make_result("conv2d_3x3", "64x16x16x16->32", warmup, iterations,
                       total_ms);
}

void write_csv(const std::string& path, const std::vector<Result>& results) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open benchmark CSV: " + path);
    }

    file << "op,shape,iterations,warmup,total_ms,avg_ms\n";
    for (const Result& result : results) {
        file << result.op << ','
             << result.shape << ','
             << result.iterations << ','
             << result.warmup << ','
             << result.total_ms << ','
             << result.avg_ms << '\n';
    }
}

void print_results(const std::vector<Result>& results) {
    std::cout << "ops_benchmark : results\n";
    std::cout << std::left
              << std::setw(18) << "op"
              << std::setw(20) << "shape"
              << std::right
              << std::setw(12) << "iters"
              << std::setw(10) << "warmup"
              << std::setw(14) << "total_ms"
              << std::setw(12) << "avg_ms"
              << "\n";
    std::cout << std::string(86, '-') << "\n";

    std::cout << std::fixed << std::setprecision(4);
    for (const Result& result : results) {
        std::cout << std::left
                  << std::setw(18) << result.op
                  << std::setw(20) << result.shape
                  << std::right
                  << std::setw(12) << result.iterations
                  << std::setw(10) << result.warmup
                  << std::setw(14) << result.total_ms
                  << std::setw(12) << result.avg_ms
                  << "\n";
    }
}

}  // namespace

int main() {
    try {
        const int warmup = env_int("OPS_BENCH_WARMUP", 10);
        const int iterations = env_int("OPS_BENCH_ITERS", 100);
        const std::string csv_path =
            env_string("OPS_BENCH_CSV", "benchmark_results/ops.csv");
        const dl::Device device(dl::DeviceType::CUDA, 0);

        dl::autograd::NoGradGuard no_grad;
        std::vector<Result> results;
        results.push_back(benchmark_binary("add", device, warmup, iterations,
                                           10, dl::ops::add));
        results.push_back(benchmark_binary("sub", device, warmup, iterations,
                                           20, dl::ops::sub));
        results.push_back(benchmark_binary("mul", device, warmup, iterations,
                                           30, dl::ops::mul));
        results.push_back(benchmark_binary("div", device, warmup, iterations,
                                           40, dl::ops::div));
        results.push_back(benchmark_unary("relu", device, warmup, iterations,
                                          300, dl::ops::relu));
        results.push_back(benchmark_unary("leaky_relu", device, warmup,
                                          iterations, 310,
                                          [](const dl::Tensor& x) {
                                              return dl::ops::leaky_relu(x);
                                          }));
        results.push_back(benchmark_unary("gelu", device, warmup, iterations,
                                          320, dl::ops::gelu));
        results.push_back(benchmark_unary("sigmoid", device, warmup, iterations,
                                          330, dl::ops::sigmoid));
        results.push_back(benchmark_unary("tanh", device, warmup, iterations,
                                          340, dl::ops::tanh));
        results.push_back(benchmark_matmul(device, warmup, iterations));
        results.push_back(benchmark_linear(device, warmup, iterations));
        results.push_back(benchmark_conv2d_3x3(device, warmup, iterations));
        results.push_back(benchmark_softmax(device, warmup, iterations));
        results.push_back(benchmark_layernorm(device, warmup, iterations));
        results.push_back(benchmark_cross_entropy(device, warmup, iterations));

        write_csv(csv_path, results);
        print_results(results);
        std::cout << "ops_benchmark_csv : " << csv_path << "\n";
        std::cout << "ops_benchmark : passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ops_benchmark_error : " << error.what() << "\n";
    } catch (...) {
        std::cerr << "ops_benchmark_error : unknown\n";
    }

    std::cout << "ops_benchmark : not passed\n";
    return EXIT_FAILURE;
}
