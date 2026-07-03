#include <dl/ops/Flatten.hpp>

#include <cuda_runtime.h>
#include <dl/ops/OpUtils.hpp>
#include <dl/autograd/FlattenOperator.hpp>
#include <dl/autograd/GradMode.hpp>
#include <dl/core/CudaUtils.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace dl::ops {
    Tensor flatten(const Tensor& a) {
        check_defined(a, "flatten", "a");
        check_float32(a, "flatten", "a");
        check_cuda(a, "flatten", "a");
        if (a.shape().rank() < 2) {
            throw std::runtime_error("flatten input rank must be >= 2");
        }

        const int64_t batch_size = a.shape()[0];
        const int64_t feature_count = a.numel() / batch_size;
        Tensor output(Shape({batch_size, feature_count}), a.dtype(), a.device());
        output.copy_from(a);
        if (dl::autograd::is_grad_enabled() && a.requires_grad()) {
            auto op = std::make_shared<dl::autograd::FlattenOperator>(a.shape());
            op->setup_computation_graph({a}, {output});
            output.set_requires_grad(true);
            output.set_creator(op);
        }
        return output;
    }
}  // namespace dl::ops
