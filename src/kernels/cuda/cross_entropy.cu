#include <dl/kernels/cross_entropy.hpp>

#include <cuda_runtime.h>

#include <cmath>

namespace {

__global__ void cross_entropy_kernel(const float* logits, const int64_t* labels,
                                     float* loss, int batch, int classes) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) {
        return;
    }

    const int64_t label = labels[row];
    if (label < 0 || label >= classes) {
        return;
    }

    const float* row_logits = logits + row * classes;
    float max_logit = row_logits[0];
    for (int col = 1; col < classes; ++col) {
        max_logit = fmaxf(max_logit, row_logits[col]);
    }

    float sum_exp = 0.0f;
    for (int col = 0; col < classes; ++col) {
        sum_exp += expf(row_logits[col] - max_logit);
    }

    const float loss_i = logf(sum_exp) + max_logit -
                         row_logits[static_cast<int>(label)];
    atomicAdd(loss, loss_i / static_cast<float>(batch));
}

void launch_cross_entropy_kernel(const float* logits, const int64_t* labels,
                                 float* loss, int batch, int classes) {
    const int threads_per_block = 256;
    const int blocks = (batch + threads_per_block - 1) / threads_per_block;
    cross_entropy_kernel<<<blocks, threads_per_block>>>(logits, labels, loss,
                                                        batch, classes);
}

}  // namespace

namespace dl::kernels {

void cross_entropy(const float* logits, const int64_t* labels, float* loss,
                   int batch, int classes) {
    launch_cross_entropy_kernel(logits, labels, loss, batch, classes);
}

}  // namespace dl::kernels
