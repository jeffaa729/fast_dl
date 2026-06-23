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
                                  int N) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < N && col < N) {
        float res = 0.0f;
        for (int k = 0; k < N; k++) {
            res += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = res;
    }
}

void launch_gemm_naive(const float* a, const float* b, float* c, int N) {
    constexpr int tile_dim = 16;
    const dim3 threads(tile_dim, tile_dim);
    const dim3 blocks((N + tile_dim - 1) / tile_dim,
                      (N + tile_dim - 1) / tile_dim);
    gemm_naive_kernel<<<blocks, threads>>>(a, b, c, N);
}

// Tiled kernel: cache A and B tiles in shared memory.
__global__ void gemm_tiled_kernel(const float* a, const float* b, float* c,
                                  int N) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TS + ty;
    const int col = blockIdx.x * TS + tx;
    float acc = 0.0f;

    for (int t = 0; t < (N + TS - 1) / TS; t++) {
        const int tiled_col = t * TS + tx;
        const int tiled_row = t * TS + ty;
        As[ty][tx] = (row < N && tiled_col < N) ? a[row * N + tiled_col] : 0.0f;
        Bs[ty][tx] = (tiled_row < N && col < N) ? b[tiled_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TS; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        c[row * N + col] = acc;
    }
}

void launch_gemm_tiled(const float* a, const float* b, float* c, int N) {
    const dim3 threads(TS, TS);
    const dim3 blocks((N + TS - 1) / TS, (N + TS - 1) / TS);
    gemm_tiled_kernel<<<blocks, threads>>>(a, b, c, N);
}

// Register-blocked kernel: each thread computes an 8x8 C tile.
__global__ void gemm_register_kernel(const float* A, const float* B, float* C,
                                     int N) {
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

    for (int tile_k = 0; tile_k < (N + BK - 1) / BK; ++tile_k) {
        for (int idx = tid; idx < BM * BK; idx += threads_per_block) {
            const int row = idx / BK;
            const int col = idx % BK;
            const int global_row = block_row + row;
            const int global_col = tile_k * BK + col;
            As[row][col] =
                (global_row < N && global_col < N)
                    ? A[global_row * N + global_col]
                    : 0.0f;
        }

        for (int idx = tid; idx < BK * BN; idx += threads_per_block) {
            const int row = idx / BN;
            const int col = idx % BN;
            const int global_row = tile_k * BK + row;
            const int global_col = block_col + col;
            Bs[row][col] =
                (global_row < N && global_col < N)
                    ? B[global_row * N + global_col]
                    : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float a_reg[TM];
            float b_reg[TN];

            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[ty * TM + i][k];
            }
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[k][tx * TN + j];
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
        if (row >= N) {
            continue;
        }
        for (int j = 0; j < TN; ++j) {
            const int col = col_base + j;
            if (col < N) {
                C[row * N + col] = c_reg[i][j];
            }
        }
    }
}

void launch_gemm_register(const float* a, const float* b, float* c, int N) {
    const dim3 threads(BN / TN, BM / TM);
    const dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);
    gemm_register_kernel<<<blocks, threads>>>(a, b, c, N);
}

void launch_gemm_cublas(const float* a, const float* b, float* c, int N) {
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle));
    cublas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha, b, N, a, N, &beta, c, N));
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

void gemm(const float* a, const float* b, float* c, int N, GemmAlgo algo) {
    switch (algo) {
        case GemmAlgo::Naive:
            launch_gemm_naive(a, b, c, N);
            return;
        case GemmAlgo::Tiled:
            launch_gemm_tiled(a, b, c, N);
            return;
        case GemmAlgo::Register:
            launch_gemm_register(a, b, c, N);
            return;
        case GemmAlgo::Cublas:
            launch_gemm_cublas(a, b, c, N);
            return;
    }
}

}  // namespace dl::kernels
