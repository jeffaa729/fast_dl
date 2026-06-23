#pragma once

#include <cstddef>

namespace dl::kernels {

enum class ReductionAlgo {
    Interleave,
    Address
};

const char* to_string(ReductionAlgo algo);

void reduction(const float* input, float* output, std::size_t size,
               ReductionAlgo algo = ReductionAlgo::Address);

}  // namespace dl::kernels
