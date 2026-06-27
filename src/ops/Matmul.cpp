#include <dl/ops/Matmul.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/gemm.hpp>
#include <dl/ops/OpUtils.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace dl::ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
    check_defined(a, "matmul", "left");
    check_defined(b, "matmul", "right");
    check_rank(a, 2, "matmul", "left");
    check_rank(b, 2, "matmul", "right");
    check_float32(a, "matmul", "left");
    check_float32(b, "matmul", "right");
    check_cuda(a, "matmul", "left");
    check_cuda(b, "matmul", "right");
    check_same_device(a, b, "matmul");

    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("matmul input tensors have incompatible shapes");
    }

    const int m = static_cast<int>(a.shape()[0]);
    const int k = static_cast<int>(a.shape()[1]);
    const int n = static_cast<int>(b.shape()[1]);

    Tensor output(Shape({a.shape()[0], b.shape()[1]}), a.dtype(), a.device());
    dl::cuda::check(cudaSetDevice(a.device().index), "cudaSetDevice failed");
    dl::kernels::gemm(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        m,
        n,
        k,
        dl::kernels::GemmAlgo::Cublas);
    dl::cuda::check(cudaGetLastError(), "matmul kernel launch failed");
    return output;
}

}  // namespace dl::ops
