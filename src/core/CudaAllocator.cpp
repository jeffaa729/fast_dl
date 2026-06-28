#include <dl/core/CudaAllocator.hpp>

#include <dl/core/CudaUtils.hpp>

#include <cuda_runtime.h>

#include <cstddef>

namespace dl::cuda {

void* allocate(std::size_t bytes, int device_index) {
    if (bytes == 0) {
        return nullptr;
    }

    void* ptr = nullptr;
    dl::cuda::check(cudaSetDevice(device_index), "cudaSetDevice failed");
    dl::cuda::check(cudaMallocAsync(&ptr, bytes, 0), "cudaMallocAsync failed");
    return ptr;
}

void deallocate(void* ptr, std::size_t bytes, int device_index) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }

    dl::cuda::check(cudaSetDevice(device_index), "cudaSetDevice failed");
    dl::cuda::check(cudaFreeAsync(ptr, 0), "cudaFreeAsync failed");
}

}  // namespace dl::cuda
