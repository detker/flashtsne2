#include "SparseP.hpp"
#include "ICommunicator.hpp"
#include "error_utils.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include <cusparse.h>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>


static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_BINARY_SEARCH_ITERS = 200;
static constexpr float PERPLEXITY_TOL = 1e-5f;

using MPI_B = CommunicationBackend::MPI;
using NCCL_B = CommunicationBackend::NCCL;


__global__ void computeConditionalP(
    const float* __restrict__ distances,
    float* __restrict__ p_values,
    int n,
    int k,
    float target_entropy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* dist_row = distances + (int64_t)i * k;
    float* p_row = p_values + (int64_t)i * k;

    float beta_min = 1e-10f;
    float beta_max = 1e10f;
    float beta = 1.0f;

    for (int iter = 0; iter < MAX_BINARY_SEARCH_ITERS; iter++) {
        float sum_exp = 0.0f;
        for (int j = 0; j < k; j++) {
            float val = expf(-beta * dist_row[j]);
            p_row[j] = val;
            sum_exp += val;
        }
        if (sum_exp < 1e-30f) sum_exp = 1e-30f;

        float inv_sum = 1.0f / sum_exp;
        float entropy = 0.0f;
        for (int j = 0; j < k; j++) {
            p_row[j] *= inv_sum;
            if (p_row[j] > 1e-30f) {
                entropy -= p_row[j] * logf(p_row[j]);
            }
        }

        float diff = entropy - target_entropy;
        if (fabsf(diff) < PERPLEXITY_TOL) break;

        if (diff > 0.0f) {
            beta_min = beta;
            beta = (beta_max >= 1e9f) ? beta * 2.0f : (beta + beta_max) * 0.5f;
        } else {
            beta_max = beta;
            beta = (beta_min <= 1e-9f) ? beta * 0.5f : (beta + beta_min) * 0.5f;
        }
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < k; j++) {
        float val = expf(-beta * dist_row[j]);
        p_row[j] = val;
        sum_exp += val;
    }
    if (sum_exp < 1e-30f) sum_exp = 1e-30f;
    float inv_sum = 1.0f / sum_exp;
    for (int j = 0; j < k; j++) {
        p_row[j] *= inv_sum;
    }
}


__global__ void gatherRowsForSend(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int* __restrict__ indices,
    int n_indices,
    int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_indices * dim) return;
    int row = tid / dim;
    int col = tid % dim;
    dst[tid] = src[indices[row] * dim + col];
}


__global__ void fillGlobalIds(
    int* __restrict__ dst,
    const int* __restrict__ local_indices,
    int n,
    int global_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = global_offset + local_indices[tid];
}


__global__ void extractKthDistances(
    const float* __restrict__ d_all_dists,
    float* __restrict__ d_kth,
    int local_n,
    int n_neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;
    d_kth[i] = d_all_dists[i * n_neighbors + (n_neighbors - 1)];
}


__global__ void detectBoundaryPoints(
    const float* __restrict__ d_local_data,
    const float* __restrict__ d_centroids,
    const float* __restrict__ d_kth_distances,
    bool* __restrict__ d_flags,
    int local_n,
    int dim,
    int n_clusters,
    int n_ranks,
    int rank)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;

    float reach = d_kth_distances[i];
    const float* pt = d_local_data + i * dim;

    float own_dist = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = pt[d] - d_centroids[rank * dim + d];
        own_dist += diff * diff;
    }

    for (int c = 0; c < n_clusters; c++) {
        int target_rank = c % n_ranks;
        if (target_rank == rank) continue;

        float foreign_dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = pt[d] - d_centroids[c * dim + d];
            foreign_dist += diff * diff;
        }

        float boundary_margin = foreign_dist - own_dist;
        if (boundary_margin < 2.0f * reach) {
            d_flags[i * n_ranks + target_rank] = true;
        }
    }
}


__global__ void stripSelfNeighbor(
    const float* __restrict__ in_dist,
    const faiss::idx_t* __restrict__ in_idx,
    float* __restrict__ out_dist,
    faiss::idx_t* __restrict__ out_idx,
    int query_n,
    int n_neighbors)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= query_n * n_neighbors) return;
    int row = tid / n_neighbors;
    int col = tid % n_neighbors;
    int search_k = n_neighbors + 1;
    out_dist[tid] = in_dist[row * search_k + col + 1];
    out_idx[tid] = in_idx[row * search_k + col + 1];
}


struct KNNResult {
    thrust::device_vector<float> d_distances;    // [query_n * k] on GPU
    thrust::device_vector<faiss::idx_t> d_indices; // [query_n * k] on GPU
};

static KNNResult runLocalKNN(
    float* d_index_data, size_t index_n,
    float* d_query_data, size_t query_n,
    int dim, int n_neighbors,
    bool strip_self)
{
    int current_dev = 0;
    cudaGetDevice(&current_dev);

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = current_dev;
    faiss::gpu::GpuIndexFlatL2 gpu_index(&res, dim, config);

    gpu_index.add(index_n, d_index_data);

    int search_k = strip_self ? n_neighbors + 1 : n_neighbors;

    thrust::device_vector<float> d_raw_dist(query_n * search_k);
    thrust::device_vector<faiss::idx_t> d_raw_idx(query_n * search_k);

    gpu_index.search(
        query_n,
        d_query_data,
        search_k,
        thrust::raw_pointer_cast(d_raw_dist.data()),
        thrust::raw_pointer_cast(d_raw_idx.data())
    );

    KNNResult result;

    if (strip_self) {
        result.d_distances.resize(query_n * n_neighbors);
        result.d_indices.resize(query_n * n_neighbors);
        int total = query_n * n_neighbors;
        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        stripSelfNeighbor<<<grid, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(d_raw_dist.data()),
            thrust::raw_pointer_cast(d_raw_idx.data()),
            thrust::raw_pointer_cast(result.d_distances.data()),
            thrust::raw_pointer_cast(result.d_indices.data()),
            query_n, n_neighbors);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        result.d_distances = std::move(d_raw_dist);
        result.d_indices = std::move(d_raw_idx);
    }

    return result;
}


__global__ void extractFlagColumn(
    const bool* __restrict__ d_flags,
    bool* __restrict__ d_col,
    int local_n,
    int n_ranks,
    int target_rank)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;
    d_col[i] = d_flags[i * n_ranks + target_rank];
}


struct HaloGPU {
    float* d_ghost_points;            // [n_ghosts * dim] on device
    thrust::device_vector<int> d_ghost_global_ids;  // [n_ghosts] on device
    size_t n_ghosts;
};

static HaloGPU exchangeHalos(
    NCCLCommunicator& comm,
    float* d_local_data,
    size_t local_n,
    int dim,
    const float* d_centroids,
    int n_clusters,
    int rank,
    int n_ranks,
    const float* d_kth_distances,
    const std::vector<int>& rank_offsets)
{
    int my_global_offset = rank_offsets[rank];

    // Boundary detection on GPU: flag[i * n_ranks + r] = true if point i should go to rank r
    thrust::device_vector<bool> d_flags(local_n * n_ranks, false);

    {
        int grid = (local_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        detectBoundaryPoints<<<grid, BLOCK_SIZE>>>(
            d_local_data, d_centroids, d_kth_distances,
            thrust::raw_pointer_cast(d_flags.data()),
            local_n, dim, n_clusters, n_ranks, rank);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Per-rank compaction: for each target rank, stream-compact flagged point indices
    std::vector<int> send_counts_ids(n_ranks, 0);
    std::vector<int> send_counts_pts(n_ranks, 0);
    std::vector<thrust::device_vector<int>> d_per_rank_indices(n_ranks);

    thrust::device_vector<int> d_iota(local_n);
    thrust::sequence(d_iota.begin(), d_iota.end(), 0);

    thrust::device_vector<bool> d_col(local_n);
    thrust::device_vector<int> d_compact(local_n);
    int col_grid = (local_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int r = 0; r < n_ranks; r++) {
        if (r == rank) continue;

        extractFlagColumn<<<col_grid, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(d_flags.data()),
            thrust::raw_pointer_cast(d_col.data()),
            local_n, n_ranks, r);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = thrust::copy_if(
            d_iota.begin(), d_iota.end(),
            d_col.begin(),
            d_compact.begin(),
            thrust::identity<bool>());

        int count = end - d_compact.begin();
        d_per_rank_indices[r].assign(d_compact.begin(), d_compact.begin() + count);
        send_counts_ids[r] = count;
        send_counts_pts[r] = count * dim;
    }
    d_flags.clear(); d_flags.shrink_to_fit();
    d_col.clear(); d_col.shrink_to_fit();
    d_compact.clear(); d_compact.shrink_to_fit();
    d_iota.clear(); d_iota.shrink_to_fit();

    // Flatten per-rank indices into a single contiguous device buffer
    int total_send_ids = 0;
    std::vector<int> send_displs_ids(n_ranks, 0);
    for (int r = 0; r < n_ranks; r++) {
        send_displs_ids[r] = total_send_ids;
        total_send_ids += send_counts_ids[r];
    }
    std::vector<int> send_displs_pts(n_ranks, 0);
    for (int r = 0; r < n_ranks; r++) {
        send_displs_pts[r] = send_displs_ids[r] * dim;
    }
    int total_send_pts = total_send_ids * dim;

    thrust::device_vector<int> d_all_send_indices(total_send_ids);
    for (int r = 0; r < n_ranks; r++) {
        if (send_counts_ids[r] > 0) {
            thrust::copy(d_per_rank_indices[r].begin(), d_per_rank_indices[r].end(),
                         d_all_send_indices.begin() + send_displs_ids[r]);
        }
    }
    d_per_rank_indices.clear();

    // MPI count exchange
    std::vector<int> recv_counts_pts(n_ranks);
    std::vector<int> recv_counts_ids(n_ranks);
    comm.allToAll<MPI_B>(send_counts_pts.data(), 1, CommDataType::INT,
                         recv_counts_pts.data(), 1, CommDataType::INT);
    comm.allToAll<MPI_B>(send_counts_ids.data(), 1, CommDataType::INT,
                         recv_counts_ids.data(), 1, CommDataType::INT);

    std::vector<int> recv_displs_pts(n_ranks, 0);
    std::vector<int> recv_displs_ids(n_ranks, 0);
    for (int i = 1; i < n_ranks; i++) {
        recv_displs_pts[i] = recv_displs_pts[i-1] + recv_counts_pts[i-1];
        recv_displs_ids[i] = recv_displs_ids[i-1] + recv_counts_ids[i-1];
    }

    int total_recv_pts = recv_displs_pts[n_ranks-1] + recv_counts_pts[n_ranks-1];
    int total_recv_ids = recv_displs_ids[n_ranks-1] + recv_counts_ids[n_ranks-1];

    // Gather send points on GPU
    float* d_send_pts = nullptr;
    if (total_send_pts > 0) {
        CUDA_CHECK(cudaMalloc(&d_send_pts, total_send_pts * sizeof(float)));
        int grid = (total_send_ids * dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gatherRowsForSend<<<grid, BLOCK_SIZE>>>(
            d_local_data, d_send_pts,
            thrust::raw_pointer_cast(d_all_send_indices.data()),
            total_send_ids, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Build global IDs on GPU
    int* d_send_ids = nullptr;
    if (total_send_ids > 0) {
        CUDA_CHECK(cudaMalloc(&d_send_ids, total_send_ids * sizeof(int)));
        int grid = (total_send_ids + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fillGlobalIds<<<grid, BLOCK_SIZE>>>(
            d_send_ids,
            thrust::raw_pointer_cast(d_all_send_indices.data()),
            total_send_ids, my_global_offset);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    d_all_send_indices.clear(); d_all_send_indices.shrink_to_fit();

    // Allocate receive buffers on GPU
    float* d_recv_pts = nullptr;
    int* d_recv_ids = nullptr;
    if (total_recv_pts > 0) {
        CUDA_CHECK(cudaMalloc(&d_recv_pts, total_recv_pts * sizeof(float)));
    }
    if (total_recv_ids > 0) {
        CUDA_CHECK(cudaMalloc(&d_recv_ids, total_recv_ids * sizeof(int)));
    }

    // NCCL bulk transfer - GPU to GPU
    comm.allToAllV<NCCL_B>(d_send_pts, send_counts_pts.data(), send_displs_pts.data(),
                           CommDataType::FLOAT, d_recv_pts, recv_counts_pts.data(),
                           recv_displs_pts.data(), CommDataType::FLOAT);
    comm.allToAllV<NCCL_B>(d_send_ids, send_counts_ids.data(), send_displs_ids.data(),
                           CommDataType::INT, d_recv_ids, recv_counts_ids.data(),
                           recv_displs_ids.data(), CommDataType::INT);

    if (d_send_pts) cudaFree(d_send_pts);
    if (d_send_ids) cudaFree(d_send_ids);

    HaloGPU halo;
    halo.d_ghost_points = d_recv_pts;
    halo.n_ghosts = total_recv_ids;
    if (total_recv_ids > 0) {
        halo.d_ghost_global_ids.resize(total_recv_ids);
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(halo.d_ghost_global_ids.data()),
                              d_recv_ids, total_recv_ids * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaFree(d_recv_ids);
    }

    return halo;
}


struct COOEntry {
    int row;
    int col;
    float val;
};

static std::vector<COOEntry> shuffleMirrors(
    NCCLCommunicator& comm,
    const std::vector<COOEntry>& mirrors,
    int rank,
    int n_ranks,
    const std::vector<int>& rank_offsets,
    const std::vector<int>& rank_sizes)
{
    auto get_owner = [&](int global_row) -> int {
        for (int r = n_ranks - 1; r >= 0; r--) {
            if (global_row >= rank_offsets[r]) return r;
        }
        return 0;
    };

    std::vector<std::vector<COOEntry>> bins(n_ranks);
    for (const auto& e : mirrors) {
        int owner = get_owner(e.row);
        bins[owner].push_back(e);
    }

    std::vector<int> send_counts(n_ranks);
    for (int r = 0; r < n_ranks; r++) {
        send_counts[r] = bins[r].size();
    }

    std::vector<int> recv_counts(n_ranks);
    comm.allToAll<MPI_B>(send_counts.data(), 1, CommDataType::INT,
                         recv_counts.data(), 1, CommDataType::INT);

    std::vector<int> send_displs(n_ranks, 0);
    std::vector<int> recv_displs(n_ranks, 0);
    for (int i = 1; i < n_ranks; i++) {
        send_displs[i] = send_displs[i-1] + send_counts[i-1];
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    }

    int total_send = send_displs[n_ranks-1] + send_counts[n_ranks-1];
    int total_recv = recv_displs[n_ranks-1] + recv_counts[n_ranks-1];

    std::vector<int> send_rows(total_send), send_cols(total_send);
    std::vector<float> send_vals(total_send);

    for (int r = 0; r < n_ranks; r++) {
        int offset = send_displs[r];
        for (size_t j = 0; j < bins[r].size(); j++) {
            send_rows[offset + j] = bins[r][j].row;
            send_cols[offset + j] = bins[r][j].col;
            send_vals[offset + j] = bins[r][j].val;
        }
    }

    std::vector<int> recv_rows(total_recv), recv_cols(total_recv);
    std::vector<float> recv_vals(total_recv);

    comm.allToAllV<MPI_B>(send_rows.data(), send_counts.data(), send_displs.data(),
                          CommDataType::INT, recv_rows.data(), recv_counts.data(),
                          recv_displs.data(), CommDataType::INT);
    comm.allToAllV<MPI_B>(send_cols.data(), send_counts.data(), send_displs.data(),
                          CommDataType::INT, recv_cols.data(), recv_counts.data(),
                          recv_displs.data(), CommDataType::INT);
    comm.allToAllV<MPI_B>(send_vals.data(), send_counts.data(), send_displs.data(),
                          CommDataType::FLOAT, recv_vals.data(), recv_counts.data(),
                          recv_displs.data(), CommDataType::FLOAT);

    std::vector<COOEntry> received(total_recv);
    for (int i = 0; i < total_recv; i++) {
        received[i] = {recv_rows[i], recv_cols[i], recv_vals[i]};
    }
    return received;
}


SparseMatrix buildSparseP(
    NCCLCommunicator& comm,
    float* d_local_data,
    size_t local_n,
    int dim,
    int n_neighbors,
    float perplexity,
    const float* centroids,
    int n_clusters,
    cudaStream_t stream)
{
    int rank = comm.getRank();
    int n_ranks = comm.getSize();

    int my_n = (int)local_n;
    std::vector<int> all_sizes(n_ranks);
    comm.allGather<MPI_B>(&my_n, 1, CommDataType::INT,
                          all_sizes.data(), 1, CommDataType::INT);

    std::vector<int> rank_offsets(n_ranks);
    rank_offsets[0] = 0;
    for (int i = 1; i < n_ranks; i++) {
        rank_offsets[i] = rank_offsets[i-1] + all_sizes[i-1];
    }
    int N_total = rank_offsets[n_ranks-1] + all_sizes[n_ranks-1];
    int my_global_offset = rank_offsets[rank];

    if (rank == 0) std::cout << "Total points across all ranks: " << N_total << std::endl;

    // 1. LOCAL kNN (pass 1) — data stays on GPU
    if (rank == 0) std::cout << "Step 2a: Initial local kNN (k=" << n_neighbors << ")..." << std::endl;

    KNNResult local_knn = runLocalKNN(
        d_local_data, local_n,
        d_local_data, local_n,
        dim, n_neighbors, true);

    // Extract kth-distances on device
    thrust::device_vector<float> d_kth_distances(local_n);
    {
        int grid = (local_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        extractKthDistances<<<grid, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(local_knn.d_distances.data()),
            thrust::raw_pointer_cast(d_kth_distances.data()),
            local_n, n_neighbors);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Upload centroids to device
    thrust::device_vector<float> d_centroids(centroids, centroids + n_clusters * dim);

    // 2. HALO EXCHANGE — ghost points arrive on GPU
    if (rank == 0) std::cout << "Step 2b: Exchanging halo/ghost points with global IDs..." << std::endl;

    HaloGPU halo = exchangeHalos(
        comm, d_local_data, local_n, dim,
        thrust::raw_pointer_cast(d_centroids.data()), n_clusters, rank, n_ranks,
        thrust::raw_pointer_cast(d_kth_distances.data()), rank_offsets);
    d_kth_distances.clear(); d_kth_distances.shrink_to_fit();
    d_centroids.clear(); d_centroids.shrink_to_fit();

    std::cout << "Rank " << rank << ": received " << halo.n_ghosts << " ghost points" << std::endl;

    // 3. REFINED kNN (pass 2) — augmented index built on GPU
    if (halo.n_ghosts > 0) {
        if (rank == 0) std::cout << "Step 2c: Recomputing kNN with ghosts ("
                                 << local_n + halo.n_ghosts << " total indexed)..." << std::endl;
        if (rank == 1) std::cout << "Step 2c: Recomputing kNN with ghosts ("
                                 << local_n + halo.n_ghosts << " total indexed)..." << std::endl;

        size_t augmented_n = local_n + halo.n_ghosts;
        float* d_augmented;
        CUDA_CHECK(cudaMalloc(&d_augmented, augmented_n * dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_augmented, d_local_data,
                              local_n * dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_augmented + local_n * dim, halo.d_ghost_points,
                              halo.n_ghosts * dim * sizeof(float), cudaMemcpyDeviceToDevice));

        local_knn = runLocalKNN(
            d_augmented, augmented_n,
            d_local_data, local_n,
            dim, n_neighbors, true);

        cudaFree(d_augmented);
    }

    // 4. COMPUTE CONDITIONAL P(j|i) — distances already on GPU
    if (rank == 0) std::cout << "Step 2d: Computing conditional probabilities (perplexity="
                             << perplexity << ")..." << std::endl;

    thrust::device_vector<float> d_p_values(local_n * n_neighbors);

    float target_entropy = logf(perplexity);
    int grid = (local_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    computeConditionalP<<<grid, BLOCK_SIZE, 0, stream>>>(
        thrust::raw_pointer_cast(local_knn.d_distances.data()),
        thrust::raw_pointer_cast(d_p_values.data()),
        local_n,
        n_neighbors,
        target_entropy
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
    local_knn.d_distances.clear(); local_knn.d_distances.shrink_to_fit();

    // Download P values and indices for COO emission
    std::vector<float> h_p(local_n * n_neighbors);
    CUDA_CHECK(cudaMemcpy(h_p.data(), thrust::raw_pointer_cast(d_p_values.data()),
               local_n * n_neighbors * sizeof(float), cudaMemcpyDeviceToHost));
    d_p_values.clear(); d_p_values.shrink_to_fit();

    std::vector<faiss::idx_t> h_indices(local_n * n_neighbors);
    CUDA_CHECK(cudaMemcpy(h_indices.data(), thrust::raw_pointer_cast(local_knn.d_indices.data()),
               local_n * n_neighbors * sizeof(faiss::idx_t), cudaMemcpyDeviceToHost));
    local_knn.d_indices.clear(); local_knn.d_indices.shrink_to_fit();

    // Download ghost global IDs for index translation
    std::vector<int> h_ghost_ids(halo.n_ghosts);
    if (halo.n_ghosts > 0) {
        CUDA_CHECK(cudaMemcpy(h_ghost_ids.data(),
                              thrust::raw_pointer_cast(halo.d_ghost_global_ids.data()),
                              halo.n_ghosts * sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (halo.d_ghost_points) cudaFree(halo.d_ghost_points);
    halo.d_ghost_global_ids.clear(); halo.d_ghost_global_ids.shrink_to_fit();

    // 5. EMISSION — generate primary + mirror COO entries
    if (rank == 0) std::cout << "Step 2e: Emitting COO entries (primary + mirror)..." << std::endl;

    std::vector<COOEntry> local_coo;
    std::vector<COOEntry> mirror_coo;
    local_coo.reserve(local_n * n_neighbors);
    mirror_coo.reserve(local_n * n_neighbors);

    for (size_t i = 0; i < local_n; i++) {
        int i_global = my_global_offset + (int)i;
        for (int jj = 0; jj < n_neighbors; jj++) {
            faiss::idx_t j_aug = h_indices[i * n_neighbors + jj];
            if (j_aug < 0) continue;

            float pji = h_p[i * n_neighbors + jj];
            if (pji < 1e-12f) continue;

            int j_global;
            if (j_aug < (faiss::idx_t)local_n) {
                j_global = my_global_offset + (int)j_aug;
            } else {
                j_global = h_ghost_ids[j_aug - local_n];
            }

            local_coo.push_back({i_global, j_global, pji});

            if (j_aug < (faiss::idx_t)local_n) {
                local_coo.push_back({j_global, i_global, pji});
            } else {
                mirror_coo.push_back({j_global, i_global, pji});
            }
        }
    }

    h_indices.clear();
    h_p.clear();
    h_ghost_ids.clear();

    // 6. SHUFFLE — route mirrors to owner(j) via MPI alltoallv
    if (rank == 0) std::cout << "Step 2f: Shuffling mirror entries across ranks..." << std::endl;

    std::vector<COOEntry> received_mirrors = shuffleMirrors(
        comm, mirror_coo, rank, n_ranks, rank_offsets, all_sizes);

    std::cout << "Rank " << rank << ": sent " << mirror_coo.size()
              << " mirrors, received " << received_mirrors.size() << std::endl;
    mirror_coo.clear();

    // 7. MERGE local COO + received mirrors
    if (rank == 0) std::cout << "Step 2g: Merging local + received mirror entries..." << std::endl;

    local_coo.insert(local_coo.end(), received_mirrors.begin(), received_mirrors.end());
    received_mirrors.clear();

    // 8. SYMMETRIZE → P_ij = (P(j|i) + P(i|j)) / (2*N_total)
    if (rank == 0) std::cout << "Step 2h: Symmetrizing and building CSR..." << std::endl;

    std::sort(local_coo.begin(), local_coo.end(), [](const COOEntry& a, const COOEntry& b) {
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    });

    float norm = 1.0f / (2.0f * N_total);
    std::vector<int> csr_row_ptr(local_n + 1, 0);
    std::vector<int> csr_cols;
    std::vector<float> csr_vals;
    csr_cols.reserve(local_coo.size() / 2);
    csr_vals.reserve(local_coo.size() / 2);

    int prev_row = -1, prev_col = -1;
    float accum = 0.0f;

    for (size_t idx = 0; idx < local_coo.size(); idx++) {
        if (local_coo[idx].row == prev_row && local_coo[idx].col == prev_col) {
            accum += local_coo[idx].val;
        } else {
            if (prev_row >= 0) {
                float sym_val = accum * norm;
                if (sym_val > 1e-12f) {
                    int local_row = prev_row - my_global_offset;
                    if (local_row >= 0 && local_row < (int)local_n) {
                        csr_cols.push_back(prev_col);
                        csr_vals.push_back(sym_val);
                        csr_row_ptr[local_row + 1]++;
                    }
                }
            }
            prev_row = local_coo[idx].row;
            prev_col = local_coo[idx].col;
            accum = local_coo[idx].val;
        }
    }
    if (prev_row >= 0) {
        float sym_val = accum * norm;
        if (sym_val > 1e-12f) {
            int local_row = prev_row - my_global_offset;
            if (local_row >= 0 && local_row < (int)local_n) {
                csr_cols.push_back(prev_col);
                csr_vals.push_back(sym_val);
                csr_row_ptr[local_row + 1]++;
            }
        }
    }
    local_coo.clear();

    for (size_t i = 1; i <= local_n; i++) {
        csr_row_ptr[i] += csr_row_ptr[i - 1];
    }

    int64_t nnz = csr_cols.size();

    // 9. TRANSFER TO GPU + cuSPARSE DESCRIPTOR
    SparseMatrix mat;
    mat.n_rows = local_n;
    mat.n_cols = N_total;
    mat.nnz = nnz;
    mat.global_row_offset = my_global_offset;
    mat.row_offsets = thrust::device_vector<int>(csr_row_ptr.begin(), csr_row_ptr.end());
    mat.col_indices = thrust::device_vector<int>(csr_cols.begin(), csr_cols.end());
    mat.values = thrust::device_vector<float>(csr_vals.begin(), csr_vals.end());

    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat.descr,
        mat.n_rows,
        mat.n_cols,
        mat.nnz,
        thrust::raw_pointer_cast(mat.row_offsets.data()),
        thrust::raw_pointer_cast(mat.col_indices.data()),
        thrust::raw_pointer_cast(mat.values.data()),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));

    std::cout << "Rank " << rank << ": P matrix — "
              << local_n << " rows, " << nnz << " nnz ("
              << (nnz > 0 ? (float)nnz / local_n : 0.0f) << " avg/row), "
              << "global cols [0.." << N_total << ")" << std::endl;

    return mat;
}


void destroySparseP(SparseMatrix& mat) {
    if (mat.descr) {
        cusparseDestroySpMat(mat.descr);
        mat.descr = nullptr;
    }
    mat.row_offsets.clear(); mat.row_offsets.shrink_to_fit();
    mat.col_indices.clear(); mat.col_indices.shrink_to_fit();
    mat.values.clear(); mat.values.shrink_to_fit();
}
