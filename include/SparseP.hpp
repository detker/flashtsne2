#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <faiss/Index.h>
#include <vector>

class NCCLCommunicator;

struct SparseMatrix {
    thrust::device_vector<int> row_offsets;    // CSR row pointers (local_n+1)
    thrust::device_vector<int> col_indices;    // CSR column indices (nnz), GLOBAL indices
    thrust::device_vector<float> values;       // CSR values (nnz)
    int n_rows;                                // local_n (owned points)
    int n_cols;                                // N_total (global)
    int64_t nnz;                               // number of nonzeros elements
    int global_row_offset;                     // first global row index owned by this rank

    cusparseSpMatDescr_t descr = nullptr;
};

/*
1. Local kNN
2. Compute conditional P(j|i) for all neighbors
3. Emit COO entries with symmetric duplicates
4. Symmetrize: P_ij = (P(j|i) + P(i|j)) / (2*N_total)
5. COO -> CSR with cuSPARSE descriptor
*/
SparseMatrix buildSparseP(
    NCCLCommunicator& comm,      // communicator for all collective operations
    float* d_local_data,         // device ptr to this rank's cluster points [local_n * dim]
    size_t local_n,              // number of owned points on this rank
    int dim,                     // high-dimensional data dimensionality
    int n_neighbors,             // number of nearest neighbors (typically 3 * perplexity)
    float perplexity,            // target perplexity for sigma calibration
    cudaStream_t stream
);

void destroySparseP(SparseMatrix& mat);
