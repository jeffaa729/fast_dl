#include <dl/tensor/Tensor.hpp>

#include <dl/autograd/GradMode.hpp>
#include <dl/autograd/Operator.hpp>
#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/random.hpp>
#include <dl/tensor/TensorImpl.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>

namespace dl {

namespace {

const Shape& empty_shape() {
    static const Shape shape;
    return shape;
}

}  // namespace

Tensor::Tensor(const Shape& shape, DType dtype, Device device)
    : impl_(std::make_shared<TensorImpl>(shape, dtype, device)) {}

Tensor Tensor::empty(const Shape& shape, DType dtype, Device device) {
    return Tensor(shape, dtype, device);
}

Tensor Tensor::zeros(const Shape& shape, DType dtype, Device device) {
    Tensor tensor(shape, dtype, device);
    tensor.zero_();
    return tensor;
}

Tensor Tensor::randn(Shape shape, DType dtype, Device device, float mean, float stddev, uint64_t seed) {
    if (dtype != DType::Float32) {
        throw std::runtime_error("Tensor::randn currently supports Float32 only");
    }

    if (!device.is_cuda()) {
        throw std::runtime_error("Tensor::randn currently supports CUDA only");
    }

    Tensor out(shape, dtype, device);

    dl::cuda::check(cudaSetDevice(device.index), "cudaSetDevice failed");

    dl::kernels::normal_float32(
        static_cast<float*>(out.data()),
        out.numel(),
        mean,
        stddev,
        seed);

    dl::cuda::check(cudaGetLastError(), "randn kernel launch failed");

    return out;
}

Tensor Tensor::uniform(Shape shape, DType dtype, Device device, float low, float high, uint64_t seed) {
    if (dtype != DType::Float32) {
        throw std::runtime_error("Tensor::uniform currently supports Float32 only");
    }

    if (!device.is_cuda()) {
        throw std::runtime_error("Tensor::uniform currently supports CUDA only");
    }

    Tensor out(shape, dtype, device);

    dl::cuda::check(cudaSetDevice(device.index), "cudaSetDevice failed");

    dl::kernels::uniform_float32(
        static_cast<float*>(out.data()),
        out.numel(),
        low,
        high,
        seed);

    dl::cuda::check(cudaGetLastError(), "uniform kernel launch failed");

    return out;
}

Tensor Tensor::xavier_uniform(Shape shape, DType dtype, Device device, uint64_t seed) {
    if (shape.rank() < 2) {
        throw std::runtime_error("Tensor::xavier_uniform requires rank >= 2");
    }

    const float fan_in = static_cast<float>(shape[shape.rank() - 2]);
    const float fan_out = static_cast<float>(shape[shape.rank() - 1]);
    const float limit = std::sqrt(6.0f / (fan_in + fan_out));
    return Tensor::uniform(shape, dtype, device, -limit, limit, seed);
}

Tensor Tensor::kaiming_uniform(Shape shape, DType dtype, Device device, uint64_t seed) {
    if (shape.rank() < 2) {
        throw std::runtime_error("Tensor::kaiming_uniform requires rank >= 2");
    }

    const float fan_in = static_cast<float>(shape[shape.rank() - 2]);
    const float limit = std::sqrt(6.0f / fan_in);
    return Tensor::uniform(shape, dtype, device, -limit, limit, seed);
}


Tensor Tensor::empty_like(const Tensor& other) {
    if (!other.defined()) {
        throw std::runtime_error("empty_like requires a defined tensor");
    }
    return Tensor(other.shape(), other.dtype(), other.device());
}

Tensor Tensor::zeros_like(const Tensor& other) {
    Tensor tensor = empty_like(other);
    tensor.zero_();
    return tensor;
}

Tensor Tensor::ones_like(const Tensor& other) {
    if (!other.defined()) {
        throw std::runtime_error("ones_like requires a defined tensor");
    }
    if (other.dtype() != DType::Float32) {
        throw std::runtime_error("ones_like currently supports Float32 only");
    }

    std::vector<float> values(static_cast<std::size_t>(other.numel()), 1.0f);
    return Tensor::from_host<float>(values, other.shape(), other.device());
}

void* Tensor::data() {
    return impl_ ? impl_->data() : nullptr;
}

const void* Tensor::data() const {
    return impl_ ? impl_->data() : nullptr;
}

const Shape& Tensor::shape() const {
    return impl_ ? impl_->shape() : empty_shape();
}

DType Tensor::dtype() const {
    return impl_ ? impl_->dtype() : DType::Float32;
}

Device Tensor::device() const {
    return impl_ ? impl_->device() : Device();
}

int64_t Tensor::numel() const {
    return impl_ ? impl_->numel() : 0;
}

std::size_t Tensor::nbytes() const {
    return impl_ ? impl_->nbytes() : 0;
}

bool Tensor::defined() const {
    return impl_ != nullptr && impl_->data() != nullptr;
}

void Tensor::copy_from_host(const void* src, std::size_t bytes) {
    if (!impl_) {
        throw std::runtime_error("copy_from_host requires a defined tensor");
    }
    impl_->copy_from_host(src, bytes);
}

void Tensor::copy_to_host(void* dst, std::size_t bytes) const {
    if (!impl_) {
        throw std::runtime_error("copy_to_host requires a defined tensor");
    }
    impl_->copy_to_host(dst, bytes);
}

void Tensor::copy_from(const Tensor& src) {
    if (!impl_ || !src.impl_) {
        throw std::runtime_error("copy_from requires defined tensors");
    }
    impl_->copy_from(*src.impl_);
}

void Tensor::copy_to(Tensor& dst) const {
    dst.copy_from(*this);
}

void Tensor::zero_() {
    if (!impl_) {
        throw std::runtime_error("zero_ requires a defined tensor");
    }
    impl_->zero_();
}

std::shared_ptr<TensorImpl> Tensor::impl() const {
    return impl_;
}

bool Tensor::requires_grad() const {
    return impl_ ? impl_->requires_grad() : false;
}

void Tensor::set_requires_grad(bool value) {
    if (!impl_) {
        throw std::runtime_error("set_requires_grad requires a defined tensor");
    }
    impl_->set_requires_grad(value);
}

Tensor Tensor::grad() const {
    return impl_ ? impl_->grad() : Tensor();
}

void Tensor::zero_grad() {
    if (!impl_) {
        throw std::runtime_error("zero_grad requires a defined tensor");
    }
    impl_->zero_grad();
}

int Tensor::generation() const {
    return impl_ ? impl_->generation() : 0;
}

void Tensor::set_creator(std::shared_ptr<dl::autograd::Operator> creator) {
    if (!impl_) {
        throw std::runtime_error("set_creator requires a defined tensor");
    }
    impl_->set_creator(std::move(creator));
}

std::shared_ptr<dl::autograd::Operator> Tensor::creator() const {
    return impl_ ? impl_->creator() : nullptr;
}

void Tensor::accumulate_grad(const Tensor& grad) {
    if (!impl_) {
        throw std::runtime_error("accumulate_grad requires a defined tensor");
    }
    impl_->accumulate_grad(grad);
}

void Tensor::backward() {
    backward(Tensor::ones_like(*this));
}

void Tensor::backward(const Tensor& grad) {
    if (!defined()) {
        throw std::runtime_error("backward requires a defined tensor");
    }
    if (!requires_grad()) {
        throw std::runtime_error("backward called on a tensor that does not require gradients");
    }
    if (!grad.defined()) {
        throw std::runtime_error("backward gradient must be defined");
    }
    if (grad.shape().numel() != shape().numel() || grad.dtype() != dtype()) {
        throw std::runtime_error("backward gradient shape or dtype does not match tensor");
    }
    if (grad.device().type != device().type || grad.device().index != device().index) {
        throw std::runtime_error("backward gradient device does not match tensor");
    }

    dl::autograd::NoGradGuard no_grad;
    accumulate_grad(grad);

    auto root_creator = creator();
    if (!root_creator) {
        return;  // No creator, so this is a leaf tensor
    }
    using OperatorPtr = std::shared_ptr<dl::autograd::Operator>;

    auto cmp = [](const OperatorPtr& a, const OperatorPtr& b) {
        return a->generation() < b->generation();
    };

    std::priority_queue<OperatorPtr, std::vector<OperatorPtr>, decltype(cmp)> queue(cmp);
    queue.push(root_creator);

    while (!queue.empty()) {
        auto op = queue.top();
        queue.pop();

        std::vector<Tensor> grad_outputs = op->grad_outputs();  // dL/doutputs
        std::vector<Tensor> grad_inputs = op->backward(grad_outputs);  // dL/dinputs

        const std::vector<Tensor>& inputs = op->inputs();
        if (grad_inputs.size() != inputs.size()) {
            throw std::runtime_error("backward returned wrong number of gradients");
        }

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            Tensor input = inputs[i];

            if (!input.requires_grad()) {
                continue;
            }

            input.accumulate_grad(grad_inputs[i]);  // accumulate dL/dinput_i

            if (input.creator()) {
                queue.push(input.creator());
            }
        }
    }
}

}  // namespace dl
