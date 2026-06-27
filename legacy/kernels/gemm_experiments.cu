#include <dl/kernels/gemm.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <unordered_map>

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

class CublasHandleCache {
public:
    ~CublasHandleCache() {
        for (auto& entry : handles_) {
            cublasDestroy(entry.second);
        }
    }

    cublasHandle_t get() {
        int device = 0;
        cudaGetDevice(&device);

        auto it = handles_.find(device);
        if (it != handles_.end()) {
            return it->second;
        }

        cublasHandle_t handle;
        cublas_check(cublasCreate(&handle));
        cublas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        handles_[device] = handle;
        return handle;
    }

private:
    std::unordered_map<int, cublasHandle_t> handles_;
};

cublasHandle_t cublas_handle() {
    thread_local CublasHandleCache cache;
    return cache.get();
}

__device__ float read_a(const float* a, int row, int col, int m, int k,
                        bool trans_a) {
    return trans_a ? a[col * m + row] : a[row * k + col];
}

__device__ float read_b(const float* b, int row, int col, int n, int k,
                        bool trans_b) {
    return trans_b ? b[col * k + row] : b[row * n + col];
}

// Naive kernel: one thread computes one C[row, col].
__global__ void gemm_naive_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k,
                                  bool trans_a, bool trans_b) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < m && col < n) {
        float res = 0.0f;
        for (int kk = 0; kk < k; kk++) {
            res += read_a(a, row, kk, m, k, trans_a) *
                   read_b(b, kk, col, n, k, trans_b);
        }
        c[row * n + col] = res;
    }
}

void launch_gemm_naive(const float* a, const float* b, float* c,
                       int m, int n, int k, bool trans_a, bool trans_b) {
    constexpr int tile_dim = 16;
    const dim3 threads(tile_dim, tile_dim);
    const dim3 blocks((n + tile_dim - 1) / tile_dim,
                      (m + tile_dim - 1) / tile_dim);
    gemm_naive_kernel<<<blocks, threads>>>(a, b, c, m, n, k, trans_a, trans_b);
}

// Tiled kernel: cache A and B tiles in shared memory.
__global__ void gemm_tiled_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k,
                                  bool trans_a, bool trans_b) {
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
        As[ty][tx] = (row < m && tiled_col < k)
                         ? read_a(a, row, tiled_col, m, k, trans_a)
                         : 0.0f;
        Bs[ty][tx] = (tiled_row < k && col < n)
                         ? read_b(b, tiled_row, col, n, k, trans_b)
                         : 0.0f;
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
                       int m, int n, int k, bool trans_a, bool trans_b) {
    const dim3 threads(TS, TS);
    const dim3 blocks((n + TS - 1) / TS, (m + TS - 1) / TS);
    gemm_tiled_kernel<<<blocks, threads>>>(a, b, c, m, n, k, trans_a, trans_b);
}

// Register-blocked kernel: each thread computes an 8x8 C tile.
__global__ void gemm_register_kernel(const float* A, const float* B, float* C,
                                     int m, int n, int k,
                                     bool trans_a, bool trans_b) {
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
                    ? read_a(A, global_row, global_col, m, k, trans_a)
                    : 0.0f;
        }

        for (int idx = tid; idx < BK * BN; idx += threads_per_block) {
            const int row = idx / BN;
            const int col = idx % BN;
            const int global_row = tile_k * BK + row;
            const int global_col = block_col + col;
            Bs[row][col] =
                (global_row < k && global_col < n)
                    ? read_b(B, global_row, global_col, n, k, trans_b)
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
                          int m, int n, int k, bool trans_a, bool trans_b) {
    const dim3 threads(BN / TN, BM / TM);
    const dim3 blocks((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_register_kernel<<<blocks, threads>>>(a, b, c, m, n, k, trans_a, trans_b);
}

void launch_gemm_cublas(const float* a, const float* b, float* c,
                        int m, int n, int k, bool trans_a, bool trans_b) {
    cublasHandle_t handle = cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int ldb = trans_b ? k : n;
    const int lda = trans_a ? m : k;
    cublas_check(cublasSgemm(handle, op_b, op_a, n, m, k,
                             &alpha, b, ldb, a, lda, &beta, c, n));
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
          int m, int n, int k, bool trans_a, bool trans_b, GemmAlgo algo) {
    switch (algo) {
        case GemmAlgo::Naive:
            launch_gemm_naive(a, b, c, m, n, k, trans_a, trans_b);
            return;
        case GemmAlgo::Tiled:
            launch_gemm_tiled(a, b, c, m, n, k, trans_a, trans_b);
            return;
        case GemmAlgo::Register:
            launch_gemm_register(a, b, c, m, n, k, trans_a, trans_b);
            return;
        case GemmAlgo::Cublas:
            launch_gemm_cublas(a, b, c, m, n, k, trans_a, trans_b);
            return;
    }
}

}  // namespace dl::kernels
