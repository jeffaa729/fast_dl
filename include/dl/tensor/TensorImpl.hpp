#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <dl/core/Device.hpp>
#include <dl/core/DType.hpp>
#include <dl/core/Shape.hpp>
#include <dl/autograd/Operator.hpp>
#include <dl/tensor/Tensor.hpp>

namespace dl {

class TensorImpl {
public:
    TensorImpl(const Shape& shape, DType dtype, Device device);
    ~TensorImpl();

    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;

    TensorImpl(TensorImpl&& other) noexcept;
    TensorImpl& operator=(TensorImpl&& other) noexcept;

    void* data();
    const void* data() const;

    const Shape& shape() const;
    DType dtype() const;
    Device device() const;

    int64_t numel() const;
    std::size_t nbytes() const;

    void copy_from_host(const void* src, std::size_t bytes);
    void copy_to_host(void* dst, std::size_t bytes) const;
    void copy_from(const TensorImpl& src);
    void zero_();

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

    Tensor grad() const { return grad_; }
    void set_grad(const Tensor& grad) { grad_ = grad; }
    void zero_grad() { grad_ = Tensor(); }
    void accumulate_grad(const Tensor& grad);

    std::shared_ptr<dl::autograd::Operator> creator() const { return creator_; }
    void set_creator(const std::shared_ptr<dl::autograd::Operator>& creator) {
        creator_ = creator;
        generation_ = creator ? creator->generation() + 1 : 0;
    }
    int generation() const { return generation_; }


private:
    void* data_ = nullptr;
    Shape shape_;
    DType dtype_ = DType::Float32;
    Device device_;
    bool requires_grad_ = false;
    Tensor grad_;
    std::shared_ptr<dl::autograd::Operator> creator_;
    int generation_ = 0;

    void allocate();
    void release();
};

}  // namespace dl

using TensorImpl = dl::TensorImpl;
