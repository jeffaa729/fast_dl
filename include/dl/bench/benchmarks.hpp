#pragma once

#include <cstddef>

namespace dl::bench {

int vector_add_benchmark(std::size_t size);
int transpose_benchmark(std::size_t n);
int reduction_benchmark(std::size_t size);
int gemm_benchmark(std::size_t m, std::size_t n, std::size_t k);
int softmax_benchmark(std::size_t rows, std::size_t cols);

}  // namespace dl::bench
