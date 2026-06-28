#include <dl/kernels/gemm.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <unordered_map>

namespace {

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

void gemm(const float* a, const float* b, float* c,
          int m, int n, int k, bool trans_a, bool trans_b) {
    launch_gemm_cublas(a, b, c, m, n, k, trans_a, trans_b);
}

}  // namespace dl::kernels
