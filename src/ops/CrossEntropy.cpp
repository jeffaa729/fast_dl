#include <dl/ops/CrossEntropy.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/cross_entropy.hpp>
#include <dl/ops/OpUtils.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace dl::ops {

Tensor cross_entropy(const Tensor& logits, const Tensor& labels) {
    check_defined(logits, "cross_entropy", "logits");
    check_defined(labels, "cross_entropy", "labels");
    check_rank(logits, 2, "cross_entropy", "logits");
    check_rank(labels, 1, "cross_entropy", "labels");
    check_float32(logits, "cross_entropy", "logits");
    check_dtype(labels, DType::Int64, "cross_entropy", "labels");
    check_cuda(logits, "cross_entropy", "logits");
    check_cuda(labels, "cross_entropy", "labels");
    check_same_device(logits, labels, "cross_entropy", "logits", "labels");

    if (logits.shape()[0] != labels.shape()[0]) {
        throw std::runtime_error("cross_entropy logits and labels batch size mismatch");
    }

    if (logits.shape()[1] <= 0) {
        throw std::runtime_error("cross_entropy class count must be positive");
    }

    Tensor loss = Tensor::zeros(Shape({1}), DType::Float32, logits.device());
    dl::cuda::check(cudaSetDevice(logits.device().index), "cudaSetDevice failed");
    dl::kernels::cross_entropy(
        static_cast<const float*>(logits.data()),
        static_cast<const int64_t*>(labels.data()),
        static_cast<float*>(loss.data()),
        static_cast<int>(logits.shape()[0]),
        static_cast<int>(logits.shape()[1]));
    dl::cuda::check(cudaGetLastError(), "cross_entropy kernel launch failed");
    return loss;
}

}  // namespace dl::ops
