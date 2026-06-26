#pragma once

#include <cstddef>

namespace dl::kernels {

enum class GemmAlgo {
    Naive,
    Tiled,
    Register,
    Cublas,
};

const char* to_string(GemmAlgo algo);

void gemm(const float* a, const float* b, float* c, int m, int n, int k,
          GemmAlgo algo = GemmAlgo::Cublas);

}  // namespace dl::kernels
