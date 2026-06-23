#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
#define CUDA_BENCH_NOINLINE __declspec(noinline)
#else
#define CUDA_BENCH_NOINLINE __attribute__((noinline))
#endif

namespace dl::bench {

inline std::size_t parse_positive_size(const char* text, const char* name) {
    try {
        const unsigned long long value = std::stoull(text);
        if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
            throw std::out_of_range("invalid range");
        }
        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        std::cerr << "Invalid " << name << ": " << text << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace dl::bench
