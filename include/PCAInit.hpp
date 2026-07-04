#pragma once

#include <thrust/device_vector.h>
#include <cstddef>

class NCCLCommunicator;

struct PCAResult {
    thrust::device_vector<float> scores; // row-major, local_n x r
    thrust::device_vector<float> components; // col-major, D x r
    float init_scale; // scale_factor
    int r;
    size_t local_n;
};

// 1. S = X^T X via cuBLAS
// 2. a = X^T 1 via cuBLAS
// 3. AllReduce S, a
// 4. C = S - (1/N) a a^T locally (X-mu)^T (X-mu) = X^T X - (1/N) a a^T
// 5. Eigen-decomposition C = V Lambda V^T via cuSOLVER syevd
// 6. Project local points: scores = (X-mu) V (local_n x D @ D x r -> local_n x r)
PCAResult distributedPCA(
    NCCLCommunicator& comm,
    const thrust::device_vector<float>& local_x,
    size_t local_n,
    int dim,
    int n_components);

thrust::device_vector<float> extractInit2D(
    const thrust::device_vector<float>& scores,
    size_t n,
    int r,
    float scale);

