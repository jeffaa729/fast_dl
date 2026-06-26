#include <dl/kernels/gemm.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace {

constexpr int TS = 16;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

void cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS call failed");
    }
}

// Naive kernel: one thread computes one C[row, col].
__global__ void gemm_naive_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < m && col < n) {
        float res = 0.0f;
        for (int kk = 0; kk < k; kk++) {
            res += a[row * k + kk] * b[kk * n + col];
        }
        c[row * n + col] = res;
    }
}

void launch_gemm_naive(const float* a, const float* b, float* c,
                       int m, int n, int k) {
    constexpr int tile_dim = 16;
    const dim3 threads(tile_dim, tile_dim);
    const dim3 blocks((n + tile_dim - 1) / tile_dim,
                      (m + tile_dim - 1) / tile_dim);
    gemm_naive_kernel<<<blocks, threads>>>(a, b, c, m, n, k);
}

// Tiled kernel: cache A and B tiles in shared memory.
__global__ void gemm_tiled_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TS + ty;
    const int col = blockIdx.x * TS + tx;
    float acc = 0.0f;

    for (int t = 0; t < (k + TS - 1) / TS; t++) {
        const int tiled_col = t * TS + tx;
        const int tiled_row = t * TS + ty;
        As[ty][tx] = (row < m && tiled_col < k) ? a[row * k + tiled_col] : 0.0f;
        Bs[ty][tx] = (tiled_row < k && col < n) ? b[tiled_row * n + col] : 0.0f;
        __syncthreads();

        for (int kk = 0; kk < TS; kk++) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

void launch_gemm_tiled(const float* a, const float* b, float* c,
                       int m, int n, int k) {
    const dim3 threads(TS, TS);
    const dim3 blocks((n + TS - 1) / TS, (m + TS - 1) / TS);
    gemm_tiled_kernel<<<blocks, threads>>>(a, b, c, m, n, k);
}

// Register-blocked kernel: each thread computes an 8x8 C tile.
__global__ void gemm_register_kernel(const float* A, const float* B, float* C,
                                     int m, int n, int k) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    constexpr int threads_per_block = (BM / TM) * (BN / TN);

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int row_base = block_row + ty * TM;
    const int col_base = block_col + tx * TN;

    float c_reg[TM][TN] = {};

    for (int tile_k = 0; tile_k < (k + BK - 1) / BK; ++tile_k) {
        for (int idx = tid; idx < BM * BK; idx += threads_per_block) {
            const int row = idx / BK;
            const int col = idx % BK;
            const int global_row = block_row + row;
            const int global_col = tile_k * BK + col;
            As[row][col] =
                (global_row < m && global_col < k)
                    ? A[global_row * k + global_col]
                    : 0.0f;
        }

        for (int idx = tid; idx < BK * BN; idx += threads_per_block) {
            const int row = idx / BN;
            const int col = idx % BN;
            const int global_row = tile_k * BK + row;
            const int global_col = block_col + col;
            Bs[row][col] =
                (global_row < k && global_col < n)
                    ? B[global_row * n + global_col]
                    : 0.0f;
        }
        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            float a_reg[TM];
            float b_reg[TN];

            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[ty * TM + i][kk];
            }
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[kk][tx * TN + j];
            }

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int row = row_base + i;
        if (row >= m) {
            continue;
        }
        for (int j = 0; j < TN; ++j) {
            const int col = col_base + j;
            if (col < n) {
                C[row * n + col] = c_reg[i][j];
            }
        }
    }
}

void launch_gemm_register(const float* a, const float* b, float* c,
                          int m, int n, int k) {
    const dim3 threads(BN / TN, BM / TM);
    const dim3 blocks((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_register_kernel<<<blocks, threads>>>(a, b, c, m, n, k);
}

void launch_gemm_cublas(const float* a, const float* b, float* c,
                        int m, int n, int k) {
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle));
    cublas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             &alpha, b, n, a, k, &beta, c, n));
    cublas_check(cublasDestroy(handle));
}

}  // namespace

namespace dl::kernels {

const char* to_string(GemmAlgo algo) {
    switch (algo) {
        case GemmAlgo::Naive:
            return "naive";
        case GemmAlgo::Tiled:
            return "tiled";
        case GemmAlgo::Register:
            return "register";
        case GemmAlgo::Cublas:
            return "cublas";
    }
    return "unknown";
}

void gemm(const float* a, const float* b, float* c,
          int m, int n, int k, GemmAlgo algo) {
    switch (algo) {
        case GemmAlgo::Naive:
            launch_gemm_naive(a, b, c, m, n, k);
            return;
        case GemmAlgo::Tiled:
            launch_gemm_tiled(a, b, c, m, n, k);
            return;
        case GemmAlgo::Register:
            launch_gemm_register(a, b, c, m, n, k);
            return;
        case GemmAlgo::Cublas:
            launch_gemm_cublas(a, b, c, m, n, k);
            return;
    }
}

}  // namespace dl::kernels
