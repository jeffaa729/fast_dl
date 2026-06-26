#include <dl/tensor/Tensor.hpp>

#include <dl/tensor/TensorImpl.hpp>

#include <stdexcept>

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

}  // namespace dl
