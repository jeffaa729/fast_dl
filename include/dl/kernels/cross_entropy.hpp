#pragma once

#include <cstdint>

namespace dl::kernels {

void cross_entropy(const float* logits, const int64_t* labels, float* loss,
                   int batch, int classes);

}  // namespace dl::kernels
