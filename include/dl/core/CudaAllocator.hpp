#pragma once

#include <cstddef>

namespace dl::cuda {

void* allocate(std::size_t bytes, int device_index);
void deallocate(void* ptr, std::size_t bytes, int device_index);

}  // namespace dl::cuda
