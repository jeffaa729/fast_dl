#include <dl/kernels/activation.hpp>
#include <stdexcept>

namespace {

    __global__ void relu_kernel(const float* input, float* output, int n) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = fmaxf(0.0f, input[idx]);
        }
    }

    void lauch_relu_kernel(const float* input, float* output, int n) {
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        relu_kernel<<<numBlocks, blockSize>>>(input, output, n);
    }

    __global__ void sigmoid_kernel(const float* input, float* output, int n) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = 1.0f / (1.0f + expf(-input[idx]));
        }
    }

    void lauch_sigmoid_kernel(const float* input, float* output, int n) {
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        sigmoid_kernel<<<numBlocks, blockSize>>>(input, output, n);
    }

    __global__ void tanh_kernel(const float* input, float* output, int n) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = tanhf(input[idx]);
        }
    }

    void lauch_tanh_kernel(const float* input, float* output, int n) {
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        tanh_kernel<<<numBlocks, blockSize>>>(input, output, n);
    }

    __global__ void gelu_kernel(const float* input, float* output, int n) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            const float x = input[idx];
            output[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
        }
    }

    void lauch_gelu_kernel(const float* input, float* output, int n) {
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        gelu_kernel<<<numBlocks, blockSize>>>(input, output, n);
    }

    __global__ void leaky_relu_kernel(const float* input, float* output, int n, float alpha) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = (input[idx] > 0.0f) ? input[idx] : alpha * input[idx];
        }
    }

    void lauch_leaky_relu_kernel(const float* input, float* output, int n, float alpha) {
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        leaky_relu_kernel<<<numBlocks, blockSize>>>(input, output, n, alpha);
    }
} // namespace

namespace dl::kernels {
    void activation(const float* input, float* output, int n, ActivationOp type, float alpha) {
        switch (type) {
            case ActivationOp::ReLU:
                lauch_relu_kernel(input, output, n);
                break;
            case ActivationOp::Sigmoid:
                lauch_sigmoid_kernel(input, output, n);
                break;
            case ActivationOp::Tanh:
                lauch_tanh_kernel(input, output, n);
                break;
            case ActivationOp::GELU:
                lauch_gelu_kernel(input, output, n);
                break;
            case ActivationOp::LeakyReLU:
                lauch_leaky_relu_kernel(input, output, n, alpha);
                break;
            default:
                throw std::runtime_error("Unsupported activation type");
        }
    }
}  // namespace dl::kernels