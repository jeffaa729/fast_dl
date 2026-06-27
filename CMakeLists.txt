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
    src/tensor/Tensor.cpp
    src/tensor/TensorImpl.cpp
    src/ops/Matmul.cpp
)

add_library(dl_cuda_kernels
    src/kernels/cuda/activation.cu
    src/kernels/cuda/cross_entropy.cu
    src/kernels/cuda/elementwise.cu
    src/kernels/cuda/gemm.cu
    src/kernels/cuda/layernorm.cu
    src/kernels/cuda/reduction.cu
    src/kernels/cuda/softmax.cu
    src/kernels/cuda/transpose.cu
    src/kernels/cuda/vector_add.cu
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
target_link_libraries(dl_cuda_kernels PUBLIC CUDA::cudart CUDA::cublas)

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

    add_executable(softmax_tensor_test tests/tensor/softmax_tensor_test.cpp)
    target_link_libraries(softmax_tensor_test PRIVATE dl_core)

    add_executable(matmul_tensor_test tests/tensor/matmul_tensor_test.cpp)
    target_link_libraries(matmul_tensor_test PRIVATE dl_core)

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

    foreach(test_name IN ITEMS tensor_test tensor_copy_test softmax_tensor_test matmul_tensor_test elementwise_tensor_test activation_tensor_test linear_tensor_test cross_entropy_tensor_test layernorm_tensor_test)
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
