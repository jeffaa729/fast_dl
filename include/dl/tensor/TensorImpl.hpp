#pragma once

#include <cstddef>
#include <cstdint>

#include <dl/core/Device.hpp>
#include <dl/core/DType.hpp>
#include <dl/core/Shape.hpp>

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

private:
    void* data_ = nullptr;
    Shape shape_;
    DType dtype_ = DType::Float32;
    Device device_;

    void allocate();
    void release();
};

}  // namespace dl

using TensorImpl = dl::TensorImpl;
