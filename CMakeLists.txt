cmake_minimum_required(VERSION 3.18)

# Local builds use the conda env (conda activate faiss); on the cluster the
# module toolchain is used and faiss comes from FAISS_ROOT below (override
# with -DFAISS_ROOT=... if it moves).
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(MPI_CXX_COMPILER "$ENV{CONDA_PREFIX}/bin/mpicxx")
    set(FAISS_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "faiss install root")
else()
    set(FAISS_ROOT "/mnt/evafs/groups/hpc-gpca-lab/wkrolikowski/software/faiss_cpp" CACHE PATH "faiss install root")
endif()

project(faiss_dist LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(QUADTREE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/QuadTree")

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

find_library(FAISS_LIB NAMES faiss
            HINTS ${FAISS_ROOT}/lib ${FAISS_ROOT}/lib64 REQUIRED)
find_library(NCCL_LIB NAMES nccl
            HINTS ${FAISS_ROOT}/lib ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu REQUIRED)

set(CMAKE_INSTALL_RPATH "${FAISS_ROOT}/lib;${FAISS_ROOT}/lib64")
set(CMAKE_BUILD_RPATH   "${FAISS_ROOT}/lib;${FAISS_ROOT}/lib64")

add_executable(hpc_app
    main.cu
    src/NCCLCommunicator.cu
    src/KMeansPartition.cu
    src/KnnGraph.cu
    src/PCAInit.cu
    src/SymmetrizeP.cu
    src/TsneOptimizer.cu
    ${QUADTREE_DIR}/src/quad_tree_builder.cu
)

file(GLOB PROJECT_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
if(PROJECT_HEADERS)
    target_sources(hpc_app PRIVATE ${PROJECT_HEADERS})
endif()

target_include_directories(hpc_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${QUADTREE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/NCCLwrapper/inc
    ${FAISS_ROOT}/include
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
    CUDA::cusolver
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
)
