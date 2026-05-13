cmake_minimum_required(VERSION 3.18)
#
#if(NOT DEFINED ENV{CONDA_PREFIX})
#    message(FATAL_ERROR "'conda activate faiss'")
#endif()

if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc")
endif()


project(faiss_dist LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CONDA_PREFIX $ENV{CONDA_PREFIX})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_library(FAISS_LIB NAMES faiss HINTS ${CONDA_PREFIX}/lib REQUIRED)
find_library(NCCL_LIB NAMES nccl HINTS ${CONDA_PREFIX}/lib REQUIRED)

set(CMAKE_BUILD_RPATH "${CONDA_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH "${CONDA_PREFIX}/lib")

file(GLOB PROJECT_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

set(MPI_CXX_COMPILER "${CONDA_PREFIX}/bin/mpicxx")
find_package(MPI REQUIRED)

add_executable(hpc_app
    main.cu
    ${PROJECT_HEADERS}
    src/NCCLCommunicator.cu
    src/KMeansPartition.cu
    src/SparseP.cu
)

target_include_directories(hpc_app PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CONDA_PREFIX}/include
)

target_link_directories(hpc_app PRIVATE 
    ${CONDA_PREFIX}/lib
)

target_compile_definitions(hpc_app PRIVATE FAISS_ENABLE_GPU)
target_compile_options(hpc_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_link_libraries(hpc_app PRIVATE
    MPI::MPI_CXX
    ${FAISS_LIB}
    ${NCCL_LIB}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)

# Test binary for P_ij verification
add_executable(test_pij
    test_pij.cu
    src/NCCLCommunicator.cu
    src/SparseP.cu
)

target_include_directories(test_pij PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CONDA_PREFIX}/include
)

target_link_directories(test_pij PRIVATE
    ${CONDA_PREFIX}/lib
)

target_compile_definitions(test_pij PRIVATE FAISS_ENABLE_GPU)
target_compile_options(test_pij PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_link_libraries(test_pij PRIVATE
    MPI::MPI_CXX
    ${FAISS_LIB}
    ${NCCL_LIB}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)
