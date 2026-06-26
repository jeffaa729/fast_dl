#pragma once

#include <cuda_runtime.h>

namespace dl::cuda {

void check(cudaError_t status, const char* action);

}  // namespace dl::cuda
