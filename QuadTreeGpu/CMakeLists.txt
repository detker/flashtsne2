cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(QuadTree LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add global compiler option
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# Find MPI (Required)
find_package(MPI REQUIRED COMPONENTS CXX)

# Find NCCL
find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS /usr/local/cuda /usr/include)
find_library(NCCL_LIBRARY NAMES nccl HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL could not be found. Please set NCCL_INCLUDE_DIR and NCCL_LIBRARY manually.")
endif()

# Add QuadTree library
file(GLOB_RECURSE LIB_SOURCES "QuadTree/src/*.cu" "QuadTree/src/*.cpp")
add_library(quad_tree_lib STATIC ${LIB_SOURCES})
target_include_directories(quad_tree_lib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/QuadTree/inc>
)

# Add target specific compiler flags    
# target_compile_options(quad_tree_lib PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
# )

# Add NCCL wrapper
file(GLOB_RECURSE WRAPPER_SOURCES "NCCLwrapper/src/*.cu" "NCCLwrapper/src/*.cpp")

add_library(nccl_wrapper_lib STATIC ${WRAPPER_SOURCES})

target_include_directories(nccl_wrapper_lib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/NCCLwrapper/inc>
)

# If this will depend on QuadTreeLib we need to link agains
# target_link_libraries(nccl_wrapper_lib PUBLIC quad_tree_lib)

# Define executable
add_executable(main main.cu)

target_link_libraries(main 
    PRIVATE 
        quad_tree_lib
        nccl_wrapper_lib
        MPI::MPI_CXX
        ${NCCL_LIBRARY}
)

target_include_directories(main PRIVATE ${NCCL_INCLUDE_DIR})