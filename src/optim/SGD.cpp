#include <dl/optim/SGD.hpp>

#include <stdexcept>

#include <cuda_runtime.h>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/optimizer.hpp>

namespace dl::optim {

SGD::SGD(const std::vector<Tensor*>& parameters, float learning_rate)
    : parameters_(parameters),
      learning_rate_(learning_rate) {
    if (learning_rate <= 0.0f) {
        throw std::runtime_error("SGD learning rate must be positive");
    }
}

void SGD::zero_grad() {
    for (Tensor* parameter : parameters_) {
        if (parameter != nullptr && parameter->defined()) {
            parameter->zero_grad();
        }
    }
}

void SGD::step() {
    for (Tensor* parameter : parameters_) {
        if (parameter == nullptr || !parameter->defined()) {
            continue;
        }

        Tensor grad = parameter->grad();
        if (!grad.defined()) {
            continue;
        }

        if (parameter->dtype() != DType::Float32 || grad.dtype() != DType::Float32) {
            throw std::runtime_error("SGD currently supports Float32 tensors only");
        }
        if (!parameter->device().is_cuda() || !grad.device().is_cuda()) {
            throw std::runtime_error("SGD currently supports CUDA tensors only");
        }
        if (parameter->nbytes() != grad.nbytes()) {
            throw std::runtime_error("SGD parameter and gradient size mismatch");
        }

        dl::cuda::check(cudaSetDevice(parameter->device().index), "cudaSetDevice failed");
        dl::kernels::sgd_step_float32(
            static_cast<float*>(parameter->data()),
            static_cast<const float*>(grad.data()),
            learning_rate_,
            parameter->numel());
        dl::cuda::check(cudaGetLastError(), "SGD step kernel launch failed");
    }
}

}  // namespace dl::optim
