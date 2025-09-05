cmake_minimum_required(VERSION 3.20)
project(AndurilSlamCuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow device code to be split across multiple .cu files
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Arch list — tune for your GPU; comment out to let NVCC pick a default
# set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

add_executable(slam_cuda_demo
    profiler.cu
    frontend.cu
    backend_pcg.cu
    streaming.cu
    state_bus.cu
    state_bus.hpp
    demo_main.cu
)

# Try to link NVTX if available (optional)
find_library(NVTOOLSEXT_LIB nvToolsExt PATHS /usr/lib /usr/local/lib ENV LD_LIBRARY_PATH)
if(NVTOOLSEXT_LIB)
    target_link_libraries(slam_cuda_demo PRIVATE ${NVTOOLSEXT_LIB})
endif()