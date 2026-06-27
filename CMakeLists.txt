cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build for")
endif()

project(dl
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(DL_BUILD_TESTS "Build dl test executables" ON)
option(DL_BUILD_BENCHMARKS "Build CUDA benchmark executable" ON)

find_package(CUDAToolkit REQUIRED)

add_library(dl_core
    src/autograd/AddOperator.cpp
    src/autograd/AddRowBiasOperator.cpp
    src/autograd/CrossEntropyOperator.cpp
    src/autograd/DivOperator.cpp
    src/autograd/GradMode.cpp
    src/autograd/LayerNormOperator.cpp
    src/autograd/MatmulOperator.cpp
    src/autograd/MulOperator.cpp
    src/autograd/Operator.cpp
    src/autograd/ReLUOperator.cpp
    src/autograd/SubOperator.cpp
    src/core/CudaUtils.cpp
    src/core/Device.cpp
    src/core/DType.cpp
    src/core/Shape.cpp
    src/ops/Activation.cpp
    src/ops/CrossEntropy.cpp
    src/ops/Elementwise.cpp
    src/ops/LayerNorm.cpp
    src/ops/Linear.cpp
    src/ops/OpUtils.cpp
    src/ops/Softmax.cpp
    src/nn/Linear.cpp
    src/nn/ReLU.cpp
    src/nn/Sequential.cpp
    src/optim/SGD.cpp
    src/tensor/Tensor.cpp
    src/tensor/TensorImpl.cpp
    src/ops/Matmul.cpp
)

add_library(dl_cuda_kernels
    src/kernels/cuda/activation.cu
    src/kernels/cuda/bias.cu
    src/kernels/cuda/cross_entropy.cu
    src/kernels/cuda/elementwise.cu
    src/kernels/cuda/gemm.cu
    src/kernels/cuda/layernorm.cu
    src/kernels/cuda/optimizer.cu
    src/kernels/cuda/reduction.cu
    src/kernels/cuda/softmax.cu
    src/kernels/cuda/transpose.cu
    src/kernels/cuda/vector_add.cu
    src/kernels/cuda/random.cu
)

target_include_directories(dl_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/core
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/ops
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/tensor
)

target_include_directories(dl_cuda_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/kernels
)

target_link_libraries(dl_core PUBLIC dl_cuda_kernels CUDA::cudart)
target_link_libraries(dl_cuda_kernels PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)

target_compile_options(dl_core
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

target_compile_options(dl_cuda_kernels
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W4>>
)

if(DL_BUILD_TESTS)
    enable_testing()

    add_executable(tensor_test tests/tensor/test_tensor.cpp)
    target_link_libraries(tensor_test PRIVATE dl_core)

    add_executable(tensor_copy_test tests/tensor/tensor_copy_test.cpp)
    target_link_libraries(tensor_copy_test PRIVATE dl_core)

    add_executable(tensor_uniform_test tests/tensor/tensor_uniform_test.cpp)
    target_link_libraries(tensor_uniform_test PRIVATE dl_core)

    add_executable(tensor_randn_test tests/tensor/tensor_randn_test.cpp)
    target_link_libraries(tensor_randn_test PRIVATE dl_core)

    add_executable(softmax_tensor_test tests/tensor/softmax_tensor_test.cpp)
    target_link_libraries(softmax_tensor_test PRIVATE dl_core)

    add_executable(matmul_tensor_test tests/tensor/matmul_tensor_test.cpp)
    target_link_libraries(matmul_tensor_test PRIVATE dl_core)

    add_executable(gemm_transpose_tensor_test tests/tensor/gemm_transpose_tensor_test.cpp)
    target_link_libraries(gemm_transpose_tensor_test PRIVATE dl_core)

    add_executable(elementwise_tensor_test tests/tensor/elementwise_tensor_test.cpp)
    target_link_libraries(elementwise_tensor_test PRIVATE dl_core)

    add_executable(activation_tensor_test tests/tensor/activation_tensor_test.cpp)
    target_link_libraries(activation_tensor_test PRIVATE dl_core)

    add_executable(linear_tensor_test tests/tensor/linear_tensor_test.cpp)
    target_link_libraries(linear_tensor_test PRIVATE dl_core)

    add_executable(cross_entropy_tensor_test tests/tensor/cross_entropy_tensor_test.cpp)
    target_link_libraries(cross_entropy_tensor_test PRIVATE dl_core)

    add_executable(layernorm_tensor_test tests/tensor/layernorm_tensor_test.cpp)
    target_link_libraries(layernorm_tensor_test PRIVATE dl_core)

    add_executable(linear_module_test tests/nn/linear_module_test.cpp)
    target_link_libraries(linear_module_test PRIVATE dl_core)

    add_executable(linear_init_test tests/nn/linear_init_test.cpp)
    target_link_libraries(linear_init_test PRIVATE dl_core)

    add_executable(linear_kaiming_init_test tests/nn/linear_kaiming_init_test.cpp)
    target_link_libraries(linear_kaiming_init_test PRIVATE dl_core)

    add_executable(relu_module_test tests/nn/relu_module_test.cpp)
    target_link_libraries(relu_module_test PRIVATE dl_core)

    add_executable(sequential_module_test tests/nn/sequential_module_test.cpp)
    target_link_libraries(sequential_module_test PRIVATE dl_core)

    add_executable(custom_module_test tests/nn/custom_module_test.cpp)
    target_link_libraries(custom_module_test PRIVATE dl_core)

    add_executable(module_mode_test tests/nn/module_mode_test.cpp)
    target_link_libraries(module_mode_test PRIVATE dl_core)

    add_executable(autograd_add_test tests/autograd/autograd_add_test.cpp)
    target_link_libraries(autograd_add_test PRIVATE dl_core)

    add_executable(no_grad_guard_test tests/autograd/no_grad_guard_test.cpp)
    target_link_libraries(no_grad_guard_test PRIVATE dl_core)

    add_executable(autograd_sub_test tests/autograd/autograd_sub_test.cpp)
    target_link_libraries(autograd_sub_test PRIVATE dl_core)

    add_executable(autograd_mul_test tests/autograd/autograd_mul_test.cpp)
    target_link_libraries(autograd_mul_test PRIVATE dl_core)

    add_executable(autograd_div_test tests/autograd/autograd_div_test.cpp)
    target_link_libraries(autograd_div_test PRIVATE dl_core)

    add_executable(autograd_relu_test tests/autograd/autograd_relu_test.cpp)
    target_link_libraries(autograd_relu_test PRIVATE dl_core)

    add_executable(autograd_matmul_test tests/autograd/autograd_matmul_test.cpp)
    target_link_libraries(autograd_matmul_test PRIVATE dl_core)

    add_executable(autograd_linear_test tests/autograd/autograd_linear_test.cpp)
    target_link_libraries(autograd_linear_test PRIVATE dl_core)

    add_executable(autograd_cross_entropy_test tests/autograd/autograd_cross_entropy_test.cpp)
    target_link_libraries(autograd_cross_entropy_test PRIVATE dl_core)

    add_executable(autograd_layernorm_test tests/autograd/autograd_layernorm_test.cpp)
    target_link_libraries(autograd_layernorm_test PRIVATE dl_core)

    add_executable(mlp_backward_demo tests/demo/mlp_backward_demo.cpp)
    target_link_libraries(mlp_backward_demo PRIVATE dl_core)

    add_executable(tiny_training_loop_demo tests/demo/tiny_training_loop_demo.cpp)
    target_link_libraries(tiny_training_loop_demo PRIVATE dl_core)

    add_executable(mnist_dataloader_demo tests/demo/mnist_dataloader_demo.cpp)
    target_link_libraries(mnist_dataloader_demo PRIVATE dl_core)

    add_executable(mnist_mlp_demo tests/demo/mnist_mlp_demo.cpp)
    target_link_libraries(mnist_mlp_demo PRIVATE dl_core)

    foreach(test_name IN ITEMS tensor_test tensor_copy_test tensor_uniform_test tensor_randn_test softmax_tensor_test matmul_tensor_test gemm_transpose_tensor_test elementwise_tensor_test activation_tensor_test linear_tensor_test cross_entropy_tensor_test layernorm_tensor_test linear_module_test linear_init_test linear_kaiming_init_test relu_module_test sequential_module_test custom_module_test module_mode_test autograd_add_test no_grad_guard_test autograd_sub_test autograd_mul_test autograd_div_test autograd_relu_test autograd_matmul_test autograd_linear_test autograd_cross_entropy_test autograd_layernorm_test mlp_backward_demo tiny_training_loop_demo mnist_dataloader_demo mnist_mlp_demo)
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endforeach()
endif()

if(DL_BUILD_BENCHMARKS)
    add_executable(dl_benchmarks
        tests/benchmarks/main.cu
        tests/benchmarks/gemm/gemm_benchmark.cu
        tests/benchmarks/reduction/reduction_benchmark.cu
        tests/benchmarks/softmax/softmax_benchmark.cu
        tests/benchmarks/transpose/transpose_benchmark.cu
        tests/benchmarks/vector_add/vector_add_benchmark.cu
    )

    target_include_directories(dl_benchmarks
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/dl
    )

    target_link_libraries(dl_benchmarks PRIVATE dl_cuda_kernels)
endif()
