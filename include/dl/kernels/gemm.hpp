#pragma once

namespace dl::kernels {

void gemm(const float* a, const float* b, float* c,
          int m, int n, int k, bool trans_a, bool trans_b);

inline void gemm(const float* a, const float* b, float* c,
                 int m, int n, int k) {
    gemm(a, b, c, m, n, k, false, false);
}

}  // namespace dl::kernels
