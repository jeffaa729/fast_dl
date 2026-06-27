#include <dl/ops/LayerNorm.hpp>

#include <dl/autograd/GradMode.hpp>
#include <dl/autograd/LayerNormOperator.hpp>
#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/layernorm.hpp>
#include <dl/ops/OpUtils.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace dl::ops {

Tensor layernorm(const Tensor& input, float eps) {
    check_defined(input, "layernorm");
    check_rank(input, 2, "layernorm");
    check_float32(input, "layernorm");
    check_cuda(input, "layernorm");

    const int rows = static_cast<int>(input.shape()[0]);
    const int cols = static_cast<int>(input.shape()[1]);
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("layernorm rows and cols must be positive");
    }

    Tensor output(input.shape(), input.dtype(), input.device());
    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::layernorm(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        rows,
        cols,
        eps);
    dl::cuda::check(cudaGetLastError(), "layernorm kernel launch failed");

    if (dl::autograd::is_grad_enabled() && input.requires_grad()) {
        auto op = std::make_shared<dl::autograd::LayerNormOperator>(eps);
        op->setup_computation_graph({input}, {output});
        output.set_requires_grad(true);
        output.set_creator(op);
    }

    return output;
}

}  // namespace dl::ops
