cmake_minimum_required(VERSION 3.16)

project(dl
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DL_BUILD_TESTS "Build dl test executables" ON)
option(DL_FETCH_EIGEN "Download Eigen with CMake if it is not found locally" ON)
set(DL_EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen-5.0.0"
    CACHE PATH "Path to an Eigen include directory")

find_package(CUDAToolkit REQUIRED)

add_library(dl_core
    src/core/Device.cpp
    src/core/DType.cpp
    src/core/Shape.cpp
    src/data/DataLoader.cpp
    src/layers/Layer.cpp
    src/layers/LinearLayer.cpp
    src/layers/ParametricReLU.cpp
    src/layers/SigmoidLayer.cpp
    src/ops/Layers.cpp
    src/ops/MseLoss.cpp
    src/ops/SoftmaxCrossEntropyLoss.cpp
    src/optim/Optimizer.cpp
    src/runtime/Network.cpp
    src/tensor/Tensor.cpp
)

target_include_directories(dl_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/core
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/data
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/layers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/ops
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/optim
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dl/tensor
)

if(EXISTS "${DL_EIGEN_INCLUDE_DIR}/Eigen/Dense")
    target_include_directories(dl_core PUBLIC "${DL_EIGEN_INCLUDE_DIR}")
else()
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND)
        target_link_libraries(dl_core PUBLIC Eigen3::Eigen)
    elseif(DL_FETCH_EIGEN)
        include(FetchContent)
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(eigen)
        target_link_libraries(dl_core PUBLIC Eigen3::Eigen)
    else()
        message(FATAL_ERROR
            "Eigen was not found. Set DL_EIGEN_INCLUDE_DIR to a directory containing Eigen/Dense.")
    endif()
endif()

target_link_libraries(dl_core PUBLIC CUDA::cudart)

target_compile_options(dl_core
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(DL_BUILD_TESTS)
    enable_testing()

    add_executable(tensor_test tests/tensor/test_tensor.cpp)
    target_link_libraries(tensor_test PRIVATE dl_core)

    foreach(test_name IN ITEMS tensor_test)
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endforeach()
endif()
