#pragma once

#include <cstddef>

namespace dl::kernels {

enum class SoftmaxAlgo {
    Naive,
    SharedMemory,
    WarpShuffleRegCache,
};

const char* to_string(SoftmaxAlgo algo);

void softmax(const float* input, float* output, std::size_t rows,
             std::size_t cols, SoftmaxAlgo algo = SoftmaxAlgo::Naive);

}  // namespace dl::kernels
