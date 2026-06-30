#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cstdint>

#include "KnnGraph.hpp"

// Symmetric high-dimensional affinity matrix P in CSR form, owned on one GPU.
//
// Pipeline (all device-resident; host only sees O(1) scalars):
//   1. distances    D[i,k] from knn.distances (squared L2, precomputed by FAISS)
//   2. conditional  P(j|i) via per-row perplexity (beta) binary search
//   3. emit         2*N*K triplets: (i,j,P(j|i)) and (j,i,P(j|i))
//   4. sort+reduce  group duplicate (row,col), summing -> P(j|i)+P(i|j)
//   5. scale        / (2*N) and pack into CSR
//
// Step 3 is the answer to "where do I write P(i|j)?": you don't write in place.
// The symmetric pattern is the union of the kNN graph and its transpose, which
// is larger than the N*K buffer, so we emit both directions and let the sort
// group them.
struct CsrMatrix {
    thrust::device_vector<int>   row_offsets;  // [N+1]
    thrust::device_vector<int>   col_indices;  // [nnz]
    thrust::device_vector<float> values;       // [nnz]
    int     n   = 0;                           // N (rows == cols)
    int64_t nnz = 0;
};

// Build the symmetric P from a local k-NN graph.
// knn         : KnnGraph from computeKnnGraph (uses .distances, .indices, .n_local, .k)
// perplexity  : target perplexity for the per-row beta binary search
CsrMatrix buildSymmetricP(
    const KnnGraph& knn,
    float           perplexity,
    cudaStream_t    stream = 0);
