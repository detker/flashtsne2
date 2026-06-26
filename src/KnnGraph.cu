#include "KnnGraph.hpp"
#include "ICommunicator.hpp"
#include "error_utils.hpp"

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <thrust/device_vector.h>

#include <iostream>
#include <vector>

using MPI_B = CommunicationBackend::MPI;

static constexpr int BLOCK_SIZE = 256;


__global__ void stripSelfKernel(
    const float*        __restrict__ raw_dist,  // [n_local * (k+1)]
    const faiss::idx_t* __restrict__ raw_idx,   // [n_local * (k+1)]
    float*              __restrict__ out_dist,  // [n_local * k]
    int*                __restrict__ out_idx,   // [n_local * k]
    int n_local, int k)
{
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)n_local * k) return;

    int i  = (int)(t / k);
    int jj = (int)(t % k);
    int src = i * (k + 1) + (jj + 1);

    faiss::idx_t j = raw_idx[src];
    if (j < 0 || j >= n_local) j = i;

    out_dist[t] = raw_dist[src];
    out_idx[t]  = (int)j;
}


KnnGraph computeKnnGraph(
    NCCLCommunicator&            comm,
    thrust::device_vector<float> x,
    thrust::device_vector<float> y,
    size_t                       n_local,
    int                          dim,
    int                          k,
    cudaStream_t                 stream)
{
    const int rank    = comm.getRank();
    const int n_ranks = comm.getSize();

    int my_n = (int)n_local;
    std::vector<int> sizes(n_ranks);
    comm.allGather<MPI_B>(&my_n, 1, CommDataType::INT,
                          sizes.data(), 1, CommDataType::INT);

    std::vector<int64_t> rank_off(n_ranks);
    rank_off[0] = 0;
    for (int i = 1; i < n_ranks; ++i) rank_off[i] = rank_off[i - 1] + sizes[i - 1];
    int64_t n_total = rank_off[n_ranks - 1] + sizes[n_ranks - 1];

    if ((int)n_local <= k) {
        std::cerr << "Rank " << rank << ": WARNING n_local (" << n_local
                  << ") <= k (" << k << "); local k-NN is degenerate" << std::endl;
    }

    int device = 0;
    cudaGetDevice(&device);
    std::cout << "Step 8-9: rank " << rank << " dev " << device
              << " local k-NN (k=" << k << ", n_local=" << n_local << ")..." << std::endl;

    const int search_k = k+1;
    thrust::device_vector<float> raw_dist(n_local * search_k);
    thrust::device_vector<faiss::idx_t> raw_idx(n_local * search_k);
    {
        faiss::gpu::StandardGpuResources res;
        faiss::gpu::GpuIndexFlatConfig cfg;
        cfg.device = device;
        faiss::gpu::GpuIndexFlatL2 index(&res, dim, cfg);

        float* d_x = thrust::raw_pointer_cast(x.data());
        index.add(n_local, d_x);
        index.search(n_local, d_x, search_k,
                     thrust::raw_pointer_cast(raw_dist.data()),
                     thrust::raw_pointer_cast(raw_idx.data()));
    }

    KnnGraph g;
    g.distances.resize(n_local * (size_t)k);
    g.indices.resize(n_local * (size_t)k);

    int64_t total = (int64_t)n_local * k;
    int grid = (int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    stripSelfKernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        thrust::raw_pointer_cast(raw_dist.data()),
        thrust::raw_pointer_cast(raw_idx.data()),
        thrust::raw_pointer_cast(g.distances.data()),
        thrust::raw_pointer_cast(g.indices.data()),
        (int)n_local, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    g.x = std::move(x);
    g.y = std::move(y);
    g.n_local = n_local;
    g.k = k;
    g.dim = dim;
    g.n_total = n_total;
    g.global_offset = rank_off[rank];

    std::cout << "Rank " << rank << ": k-NN graph ready - " << n_local
              << " x " << k << " (global ids [" << g.global_offset << ", "
              << g.global_offset + (int64_t)n_local << ") of " << n_total << ")"
              << std::endl;

    return g;
}
