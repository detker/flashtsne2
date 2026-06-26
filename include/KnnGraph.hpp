#pragma once

#include <thrust/device_vector.h>
#include <cstddef>
#include <cstdint>

class NCCLCommunicator;

struct KnnGraph {
    thrust::device_vector<float> distances;  // [n_local * k] squared L2 (FAISS)
    thrust::device_vector<int>   indices;    // [n_local * k] LOCAL ids, in [0, n_local)

    thrust::device_vector<float> x;          // [n_local * dim] high-dim coords
    thrust::device_vector<float> y;          // [n_local * 2]   low-dim coords

    size_t  n_local = 0;
    int k = 0;
    int dim = 0;
    int64_t n_total = 0;
    int64_t global_offset = 0; // global row offset of this rank's local rows in the full n_total x dim matrix
};

KnnGraph computeKnnGraph(
    NCCLCommunicator& comm,
    thrust::device_vector<float> x,
    thrust::device_vector<float> y,
    size_t n_local,
    int dim,
    int k,
    cudaStream_t stream);
