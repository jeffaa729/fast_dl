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
          bool trans_a, bool trans_b, GemmAlgo algo = GemmAlgo::Cublas);

inline void gemm(const float* a, const float* b, float* c,
                 int m, int n, int k, GemmAlgo algo = GemmAlgo::Cublas) {
    gemm(a, b, c, m, n, k, false, false, algo);
}

}  // namespace dl::kernels
