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
        // Self-stripping on CPU (indices are small relative to point data)
        std::vector<float> h_dist(query_n * search_k);
        std::vector<faiss::idx_t> h_idx(query_n * search_k);
        CUDA_CHECK(cudaMemcpy(h_dist.data(), thrust::raw_pointer_cast(d_raw_dist.data()),
                              query_n * search_k * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_idx.data(), thrust::raw_pointer_cast(d_raw_idx.data()),
                              query_n * search_k * sizeof(faiss::idx_t), cudaMemcpyDeviceToHost));

        std::vector<float> out_dist(query_n * n_neighbors);
        std::vector<faiss::idx_t> out_idx(query_n * n_neighbors);

        for (size_t i = 0; i < query_n; i++) {
            int out = 0;
            for (int j = 0; j < search_k && out < n_neighbors; j++) {
                if (h_idx[i * search_k + j] != (faiss::idx_t)i) {
                    out_dist[i * n_neighbors + out] = h_dist[i * search_k + j];
                    out_idx[i * n_neighbors + out] = h_idx[i * search_k + j];
                    out++;
                }
            }
            while (out < n_neighbors) {
                out_dist[i * n_neighbors + out] = 0.0f;
                out_idx[i * n_neighbors + out] = i;
                out++;
            }
        }

        result.d_distances.assign(out_dist.begin(), out_dist.end());
        result.d_indices.assign(out_idx.begin(), out_idx.end());
    } else {
        result.d_distances = std::move(d_raw_dist);
        result.d_indices = std::move(d_raw_idx);
    }

    return result;
}


static float sqDistCPU(const float* a, const float* b, int dim) {
    float d = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
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
    const float* centroids,
    int n_clusters,
    int rank,
    int n_ranks,
    const std::vector<float>& kth_distances,
    const std::vector<int>& rank_offsets)
{
    // Boundary detection on CPU - needs centroid distances per point.
    // We only need local_data on host for this step.
    std::vector<float> h_local(local_n * dim);
    CUDA_CHECK(cudaMemcpy(h_local.data(), d_local_data,
                          local_n * dim * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> outgoing(n_ranks);

    for (size_t i = 0; i < local_n; i++) {
        float reach = kth_distances[i];
        const float* pt = h_local.data() + i * dim;
        float own_centroid_dist = sqDistCPU(pt, centroids + rank * dim, dim);

        for (int c = 0; c < n_clusters; c++) {
            int target_rank = c % n_ranks;
            if (target_rank == rank) continue;
            float foreign_dist = sqDistCPU(pt, centroids + c * dim, dim);
            float boundary_margin = foreign_dist - own_centroid_dist;
            if (boundary_margin < 2.0f * reach) {
                outgoing[target_rank].push_back(i);
            }
        }
    }
    h_local.clear();

    for (int r = 0; r < n_ranks; r++) {
        std::sort(outgoing[r].begin(), outgoing[r].end());
        outgoing[r].erase(std::unique(outgoing[r].begin(), outgoing[r].end()), outgoing[r].end());
    }

    int my_global_offset = rank_offsets[rank];

    // Flatten send indices and compute per-rank counts
    std::vector<int> send_counts_pts(n_ranks);
    std::vector<int> send_counts_ids(n_ranks);
    std::vector<int> all_send_indices;
    for (int r = 0; r < n_ranks; r++) {
        send_counts_pts[r] = outgoing[r].size() * dim;
        send_counts_ids[r] = outgoing[r].size();
        all_send_indices.insert(all_send_indices.end(), outgoing[r].begin(), outgoing[r].end());
    }
    int total_send_ids = all_send_indices.size();

    // MPI for lightweight count exchange
    std::vector<int> recv_counts_pts(n_ranks);
    std::vector<int> recv_counts_ids(n_ranks);
    comm.allToAll<MPI_B>(send_counts_pts.data(), 1, CommDataType::INT,
                         recv_counts_pts.data(), 1, CommDataType::INT);
    comm.allToAll<MPI_B>(send_counts_ids.data(), 1, CommDataType::INT,
                         recv_counts_ids.data(), 1, CommDataType::INT);

    std::vector<int> send_displs_pts(n_ranks, 0);
    std::vector<int> recv_displs_pts(n_ranks, 0);
    std::vector<int> send_displs_ids(n_ranks, 0);
    std::vector<int> recv_displs_ids(n_ranks, 0);
    for (int i = 1; i < n_ranks; i++) {
        send_displs_pts[i] = send_displs_pts[i-1] + send_counts_pts[i-1];
        recv_displs_pts[i] = recv_displs_pts[i-1] + recv_counts_pts[i-1];
        send_displs_ids[i] = send_displs_ids[i-1] + send_counts_ids[i-1];
        recv_displs_ids[i] = recv_displs_ids[i-1] + recv_counts_ids[i-1];
    }

    int total_send_pts = send_displs_pts[n_ranks-1] + send_counts_pts[n_ranks-1];
    int total_recv_pts = recv_displs_pts[n_ranks-1] + recv_counts_pts[n_ranks-1];
    int total_recv_ids = recv_displs_ids[n_ranks-1] + recv_counts_ids[n_ranks-1];

    // Upload gather indices to GPU, gather send points directly from device data
    thrust::device_vector<int> d_send_indices(all_send_indices.begin(), all_send_indices.end());

    float* d_send_pts = nullptr;
    if (total_send_pts > 0) {
        CUDA_CHECK(cudaMalloc(&d_send_pts, total_send_pts * sizeof(float)));
        int grid = (total_send_ids * dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gatherRowsForSend<<<grid, BLOCK_SIZE>>>(
            d_local_data, d_send_pts,
            thrust::raw_pointer_cast(d_send_indices.data()),
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
            thrust::raw_pointer_cast(d_send_indices.data()),
            total_send_ids, my_global_offset);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    d_send_indices.clear(); d_send_indices.shrink_to_fit();

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

    // Extract kth-distances to host for boundary detection
    std::vector<float> kth_distances(local_n);
    {
        std::vector<float> h_dists(local_n * n_neighbors);
        CUDA_CHECK(cudaMemcpy(h_dists.data(), thrust::raw_pointer_cast(local_knn.d_distances.data()),
                              local_n * n_neighbors * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < local_n; i++) {
            kth_distances[i] = h_dists[i * n_neighbors + (n_neighbors - 1)];
        }
    }

    // 2. HALO EXCHANGE — ghost points arrive on GPU
    if (rank == 0) std::cout << "Step 2b: Exchanging halo/ghost points with global IDs..." << std::endl;

    HaloGPU halo = exchangeHalos(
        comm, d_local_data, local_n, dim,
        centroids, n_clusters, rank, n_ranks,
        kth_distances, rank_offsets);

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
