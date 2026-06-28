#pragma once

#include <cstdint>

namespace dl::kernels {

void cross_entropy(const float* logits, const int64_t* labels, float* loss,
                   int batch, int classes);
void cross_entropy_backward(const float* logits, const int64_t* labels,
                            const float* grad_loss, float* grad_logits,
                            int batch, int classes);

}  // namespace dl::kernels
