#pragma once

#include <cstddef>

namespace dl::kernels {

enum class TransposeAlgo {
    Naive,
    Shared,
    Padding
};

const char* to_string(TransposeAlgo algo);

void transpose(const float* input, float* output, std::size_t size,
               TransposeAlgo algo = TransposeAlgo::Padding);

}  // namespace dl::kernels
