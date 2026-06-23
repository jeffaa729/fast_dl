#include <dl/bench/benchmark.hpp>
#include <dl/bench/benchmarks.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program << " all\n"
              << "  " << program << " vector_add [elements]\n"
              << "  " << program << " transpose [n]\n"
              << "  " << program << " reduction [elements]\n"
              << "  " << program << " gemm [n]\n"
              << "  " << program << " softmax [rows] [cols]\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string benchmark = argc > 1 ? argv[1] : "all";

    if (benchmark == "all") {
        int status = EXIT_SUCCESS;
        status |= dl::bench::vector_add_benchmark(1ULL << 24);
        std::cout << '\n';
        status |= dl::bench::transpose_benchmark(4096);
        std::cout << '\n';
        status |= dl::bench::reduction_benchmark(1ULL << 24);
        std::cout << '\n';
        status |= dl::bench::gemm_benchmark(512);
        std::cout << '\n';
        status |= dl::bench::softmax_benchmark(4096, 1024);
        return status == EXIT_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (benchmark == "vector_add") {
        const std::size_t size =
            argc > 2 ? dl::bench::parse_positive_size(argv[2], "vector size")
                     : 1ULL << 24;
        return dl::bench::vector_add_benchmark(size);
    }

    if (benchmark == "transpose") {
        const std::size_t n =
            argc > 2 ? dl::bench::parse_positive_size(argv[2], "n") : 4096;
        return dl::bench::transpose_benchmark(n);
    }

    if (benchmark == "reduction") {
        const std::size_t size =
            argc > 2 ? dl::bench::parse_positive_size(argv[2], "size")
                     : 1ULL << 24;
        return dl::bench::reduction_benchmark(size);
    }

    if (benchmark == "gemm") {
        const std::size_t n =
            argc > 2 ? dl::bench::parse_positive_size(argv[2], "n") : 512;
        return dl::bench::gemm_benchmark(n);
    }

    if (benchmark == "softmax") {
        const std::size_t rows =
            argc > 2 ? dl::bench::parse_positive_size(argv[2], "rows") : 4096;
        const std::size_t cols =
            argc > 3 ? dl::bench::parse_positive_size(argv[3], "cols") : 1024;
        return dl::bench::softmax_benchmark(rows, cols);
    }

    std::cerr << "Unknown benchmark: " << benchmark << "\n\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
}
