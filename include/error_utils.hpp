#pragma once

#include <cstdio>
#include <cstdlib>

#define ERR(source) (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), exit(EXIT_FAILURE))

#define CUDA_CHECK(call) do {                                                                 \
    cudaError_t e = (call);                                                                   \
    if (e != cudaSuccess) {                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                                              \
    } } while(0)


// void nccl_check(ncclResult_t status, const char *file, int line) {
//   if (status != ncclSuccess) {
//     printf("[NCCL ERROR] at file %s:%d:\n%s\n", file, line,
//            ncclGetErrorString(status));
//     exit(EXIT_FAILURE);
//   }
// }
// #define NCCL_CHECK(err) (nccl_check(err, __FILE__, __LINE__))

// void mpi_check(int status, const char *file, int line) {
//   if (status != MPI_SUCCESS) {
//     char mpi_error[4096];
//     int mpi_error_len = 0;
//     assert(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) ==
//            MPI_SUCCESS);
//     printf("[MPI ERROR] at file %s:%d:\n%.*s\n", file, line, mpi_error_len,
//            mpi_error);
//     exit(EXIT_FAILURE);
//   }
// }
// #define MPI_CHECK(err) (mpi_check(err, __FILE__, __LINE__))

#define NCCL_CHECK(call) do {                                                                 \
    ncclResult_t r = (call);                                                                   \
    if (r != ncclSuccess) {                                                                   \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(1);                                                                              \
    } } while(0)

#define CUSPARSE_CHECK(call) do {                                                  \
    cusparseStatus_t s = (call);                                                   \
    if (s != CUSPARSE_STATUS_SUCCESS) {                                            \
        fprintf(stderr, "cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(1);                                                                   \
    } } while(0)

#define CUBLAS_CHECK(call) do {                                                    \
    cublasStatus_t s = (call);                                                     \
    if (s != CUBLAS_STATUS_SUCCESS) {                                              \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s);   \
        exit(1);                                                                   \
    } } while(0)

#define CUSOLVER_CHECK(call) do {                                                  \
    cusolverStatus_t s = (call);                                                   \
    if (s != CUSOLVER_STATUS_SUCCESS) {                                            \
        fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(1);                                                                   \
    } } while(0)

inline void usage(const char* prog_name) {
    fprintf(stderr, "Usage: %s <data_path> <dim> <k> <niter>\n", prog_name);
    exit(EXIT_FAILURE);
}