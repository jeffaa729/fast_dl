#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t cuda_bench_error = (call);                            \
        if (cuda_bench_error != cudaSuccess) {                                  \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__       \
                      << ": " << cudaGetErrorString(cuda_bench_error) << '\n'; \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

namespace dl::bench {

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&data_, bytes()));
    }

    ~DeviceBuffer() { cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void copy_from_host(const T* source) {
        CUDA_CHECK(cudaMemcpy(data_, source, bytes(), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* destination) const {
        CUDA_CHECK(cudaMemcpy(destination, data_, bytes(), cudaMemcpyDeviceToHost));
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace dl::bench
