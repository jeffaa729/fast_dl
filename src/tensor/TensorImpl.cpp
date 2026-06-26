#include <dl/tensor/TensorImpl.hpp>

#include <dl/core/CudaUtils.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

namespace dl {

TensorImpl::TensorImpl(const Shape& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    allocate();
}

TensorImpl::~TensorImpl() {
    release();
}

TensorImpl::TensorImpl(TensorImpl&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      device_(other.device_) {
    other.data_ = nullptr;
}

TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        other.data_ = nullptr;
    }
    return *this;
}

void* TensorImpl::data() {
    return data_;
}

const void* TensorImpl::data() const {
    return data_;
}

const Shape& TensorImpl::shape() const {
    return shape_;
}

DType TensorImpl::dtype() const {
    return dtype_;
}

Device TensorImpl::device() const {
    return device_;
}

int64_t TensorImpl::numel() const {
    return shape_.numel();
}

std::size_t TensorImpl::nbytes() const {
    return static_cast<std::size_t>(numel()) * dtype_size(dtype_);
}

void TensorImpl::allocate() {
    if (nbytes() == 0) {
        return;
    }
    if (!device_.is_cuda()) {
        throw std::runtime_error("Tensor currently supports CUDA allocation only");
    }
    dl::cuda::check(cudaSetDevice(device_.index), "cudaSetDevice failed");
    dl::cuda::check(cudaMalloc(&data_, nbytes()), "cudaMalloc failed");
}

void TensorImpl::release() {
    if (data_ == nullptr) {
        return;
    }
    if (device_.is_cuda()) {
        cudaSetDevice(device_.index);
        cudaFree(data_);
    }
    data_ = nullptr;
}

void TensorImpl::copy_from_host(const void* src, std::size_t bytes) {
    if (bytes > nbytes()) {
        throw std::runtime_error("copy_from_host exceeds tensor size");
    }
    if (bytes == 0) {
        return;
    }

    if (!device_.is_cuda()) {
        throw std::runtime_error("copy_from_host currently supports CUDA tensors only");
    }

    dl::cuda::check(cudaSetDevice(device_.index), "cudaSetDevice failed");
    dl::cuda::check(cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice),
                    "cudaMemcpy host to device failed");
}

void TensorImpl::copy_to_host(void* dst, std::size_t bytes) const {
    if (bytes > nbytes()) {
        throw std::runtime_error("copy_to_host exceeds tensor size");
    }
    if (bytes == 0) {
        return;
    }

    if (!device_.is_cuda()) {
        throw std::runtime_error("copy_to_host currently supports CUDA tensors only");
    }

    dl::cuda::check(cudaSetDevice(device_.index), "cudaSetDevice failed");
    dl::cuda::check(cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy device to host failed");
}

void TensorImpl::copy_from(const TensorImpl& src) {
    if (src.nbytes() != nbytes()) {
        throw std::runtime_error("tensor copy size mismatch");
    }
    if (nbytes() == 0) {
        return;
    }

    if (src.device().is_cuda() && device_.is_cuda()) {
        dl::cuda::check(cudaSetDevice(device_.index), "cudaSetDevice failed");
        dl::cuda::check(cudaMemcpy(data_, src.data(), nbytes(), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy device to device failed");
        return;
    }

    throw std::runtime_error("copy_from currently supports CUDA to CUDA only");
}

void TensorImpl::zero_() {
    if (nbytes() == 0) {
        return;
    }

    if (!device_.is_cuda()) {
        throw std::runtime_error("zero_ currently supports CUDA tensors only");
    }

    dl::cuda::check(cudaSetDevice(device_.index), "cudaSetDevice failed");
    dl::cuda::check(cudaMemset(data_, 0, nbytes()), "cudaMemset failed");
}

}  // namespace dl
