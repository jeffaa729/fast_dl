#include <dl/core/CudaUtils.hpp>

#include <stdexcept>
#include <string>

namespace dl::cuda {

void check(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " +
                                 cudaGetErrorString(status));
    }
}

}  // namespace dl::cuda
