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


// kernel: binary search for per-point sigma
__global__ void computeConditionalP(
    const float* __restrict__ distances,  // [n * k] squared L2 distances
    float* __restrict__ p_values,         // [n * k] output conditional probs
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


struct KNNResult {
    std::vector<float> distances;       // [query_n * k]
    std::vector<faiss::idx_t> indices;  // [query_n * k] indices into the indexed set
};

static KNNResult runLocalKNN(
    float* index_data, size_t index_n,
    float* query_data, size_t query_n,
    int dim, int n_neighbors,
    bool strip_self)
{
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);

    std::vector<faiss::gpu::GpuResourcesProvider*> resources(n_gpus);
    std::vector<int> devs(n_gpus);
    for (int i = 0; i < n_gpus; i++) {
        resources[i] = new faiss::gpu::StandardGpuResources();
        devs[i] = i;
    }

    faiss::IndexFlatL2 index_cpu(dim);
    faiss::gpu::GpuMultipleClonerOptions opts;
    opts.shard = true;
    faiss::Index* gpu_index = faiss::gpu::index_cpu_to_gpu_multiple(
        resources, devs, &index_cpu, &opts);

    size_t add_batch = 50000;
    for (size_t i = 0; i < index_n; i += add_batch) {
        size_t batch = std::min(add_batch, index_n - i);
        gpu_index->add(batch, index_data + i * dim);
    }

    int search_k = strip_self ? n_neighbors + 1 : n_neighbors;

    std::vector<float> raw_dist(query_n * search_k);
    std::vector<faiss::idx_t> raw_idx(query_n * search_k);

    size_t query_batch = 50000;
    for (size_t i = 0; i < query_n; i += query_batch) {
        size_t batch = std::min(query_batch, query_n - i);
        gpu_index->search(
            batch,
            query_data + i * dim,
            search_k,
            raw_dist.data() + i * search_k,
            raw_idx.data() + i * search_k
        );
    }

    delete gpu_index;
    for (auto r : resources) delete r;

    KNNResult result;
    result.distances.resize(query_n * n_neighbors);
    result.indices.resize(query_n * n_neighbors);

    if (strip_self) {
        for (size_t i = 0; i < query_n; i++) {
            int out = 0;
            for (int j = 0; j < search_k && out < n_neighbors; j++) {
                if (raw_idx[i * search_k + j] != (faiss::idx_t)i) {
                    result.distances[i * n_neighbors + out] = raw_dist[i * search_k + j];
                    result.indices[i * n_neighbors + out] = raw_idx[i * search_k + j];
                    out++;
                }
            }
            while (out < n_neighbors) {
                result.distances[i * n_neighbors + out] = 0.0f;
                result.indices[i * n_neighbors + out] = i;
                out++;
            }
        }
    } else {
        result.distances = std::move(raw_dist);
        result.indices = std::move(raw_idx);
    }

    return result;
}


// squared L2 distance between two vectors
static float sqDistCPU(const float* a, const float* b, int dim) {
    float d = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}


// HALO EXCHANGE: identify boundary points, exchange ghosts with global ID tracking
struct HaloData {
    std::vector<float> ghost_points;      // [n_ghosts * dim]
    std::vector<int> ghost_global_ids;    // global ID for each ghost point
    size_t n_ghosts;
};

static HaloData exchangeHalos(
    NCCLCommunicator& comm,
    float* local_data,
    size_t local_n,
    int dim,
    const float* centroids,
    int n_clusters,
    int rank,
    int n_ranks,
    const std::vector<float>& kth_distances,
    const std::vector<int>& rank_offsets)
{
    std::vector<std::vector<int>> outgoing(n_ranks);

    for (size_t i = 0; i < local_n; i++) {
        float reach = kth_distances[i];
        const float* pt = local_data + i * dim;
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

    // Deduplicate per-rank send lists
    for (int r = 0; r < n_ranks; r++) {
        std::sort(outgoing[r].begin(), outgoing[r].end());
        outgoing[r].erase(std::unique(outgoing[r].begin(), outgoing[r].end()), outgoing[r].end());
    }

    int my_global_offset = rank_offsets[rank];

    std::vector<int> send_counts_pts(n_ranks);
    std::vector<int> send_counts_ids(n_ranks);
    for (int r = 0; r < n_ranks; r++) {
        send_counts_pts[r] = outgoing[r].size() * dim;
        send_counts_ids[r] = outgoing[r].size();
    }

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
    int total_send_ids = send_displs_ids[n_ranks-1] + send_counts_ids[n_ranks-1];
    int total_recv_ids = recv_displs_ids[n_ranks-1] + recv_counts_ids[n_ranks-1];

    std::vector<float> send_buf_pts(total_send_pts);
    std::vector<int> send_buf_ids(total_send_ids);
    for (int r = 0; r < n_ranks; r++) {
        int offset_pts = send_displs_pts[r];
        int offset_ids = send_displs_ids[r];
        for (size_t j = 0; j < outgoing[r].size(); j++) {
            int idx = outgoing[r][j];
            std::copy(local_data + idx * dim, local_data + (idx + 1) * dim,
                      send_buf_pts.data() + offset_pts + j * dim);
            send_buf_ids[offset_ids + j] = my_global_offset + idx;
        }
    }

    std::vector<float> recv_buf_pts(total_recv_pts);
    std::vector<int> recv_buf_ids(total_recv_ids);

    comm.allToAllV<MPI_B>(send_buf_pts.data(), send_counts_pts.data(), send_displs_pts.data(),
                          CommDataType::FLOAT, recv_buf_pts.data(), recv_counts_pts.data(),
                          recv_displs_pts.data(), CommDataType::FLOAT);
    comm.allToAllV<MPI_B>(send_buf_ids.data(), send_counts_ids.data(), send_displs_ids.data(),
                          CommDataType::INT, recv_buf_ids.data(), recv_counts_ids.data(),
                          recv_displs_ids.data(), CommDataType::INT);

    HaloData halo;
    halo.ghost_points = std::move(recv_buf_pts);
    halo.ghost_global_ids = std::move(recv_buf_ids);
    halo.n_ghosts = total_recv_ids;
    return halo;
}


// COO entry with global indices for the shuffle step
struct COOEntry {
    int row;
    int col;
    float val;
};

// SHUFFLE: route mirror entries to their owning ranks
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
    float* local_data,
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

    // Compute global offsets via allgather
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

    // 1. LOCAL kNN (pass 1) → per-point reach
    if (rank == 0) std::cout << "Step 2a: Initial local kNN (k=" << n_neighbors << ")..." << std::endl;

    KNNResult local_knn = runLocalKNN(
        local_data, local_n,
        local_data, local_n,
        dim, n_neighbors, true);

    std::vector<float> kth_distances(local_n);
    for (size_t i = 0; i < local_n; i++) {
        kth_distances[i] = local_knn.distances[i * n_neighbors + (n_neighbors - 1)];
    }

    // 2. HALO EXCHANGE (points + global IDs)
    if (rank == 0) std::cout << "Step 2b: Exchanging halo/ghost points with global IDs..." << std::endl;

    HaloData halo = exchangeHalos(
        comm, local_data, local_n, dim,
        centroids, n_clusters, rank, n_ranks,
        kth_distances, rank_offsets);

    std::cout << "Rank " << rank << ": received " << halo.n_ghosts << " ghost points" << std::endl;

    // 3. REFINED kNN (pass 2) with owned + ghost
    if (halo.n_ghosts > 0) {
        if (rank == 0) std::cout << "Step 2c: Recomputing kNN with ghosts ("
                                 << local_n + halo.n_ghosts << " total indexed)..." << std::endl;
        if (rank == 1) std::cout << "Step 2c: Recomputing kNN with ghosts ("
                                 << local_n + halo.n_ghosts << " total indexed)..." << std::endl;

        size_t augmented_n = local_n + halo.n_ghosts;
        std::vector<float> augmented_data(augmented_n * dim);
        std::copy(local_data, local_data + local_n * dim, augmented_data.data());
        std::copy(halo.ghost_points.begin(), halo.ghost_points.end(),
                  augmented_data.data() + local_n * dim);

        local_knn = runLocalKNN(
            augmented_data.data(), augmented_n,
            local_data, local_n,
            dim, n_neighbors, true);
    }

    // 4. COMPUTE CONDITIONAL P(j|i) ON GPU
    if (rank == 0) std::cout << "Step 2d: Computing conditional probabilities (perplexity="
                             << perplexity << ")..." << std::endl;

    thrust::device_vector<float> d_distances(local_knn.distances.begin(), local_knn.distances.end());
    thrust::device_vector<float> d_p_values(local_n * n_neighbors);

    float target_entropy = logf(perplexity);
    int grid = (local_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    computeConditionalP<<<grid, BLOCK_SIZE, 0, stream>>>(
        thrust::raw_pointer_cast(d_distances.data()),
        thrust::raw_pointer_cast(d_p_values.data()),
        local_n,
        n_neighbors,
        target_entropy
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
    d_distances.clear(); d_distances.shrink_to_fit();

    std::vector<float> h_p(local_n * n_neighbors);
    CUDA_CHECK(cudaMemcpy(h_p.data(), thrust::raw_pointer_cast(d_p_values.data()),
               local_n * n_neighbors * sizeof(float), cudaMemcpyDeviceToHost));
    d_p_values.clear(); d_p_values.shrink_to_fit();

    // 5. EMISSION - generate primary + mirror COO entries
    if (rank == 0) std::cout << "Step 2e: Emitting COO entries (primary + mirror)..." << std::endl;

    std::vector<COOEntry> local_coo;
    std::vector<COOEntry> mirror_coo;
    local_coo.reserve(local_n * n_neighbors);
    mirror_coo.reserve(local_n * n_neighbors);

    for (size_t i = 0; i < local_n; i++) {
        int i_global = my_global_offset + (int)i;
        for (int jj = 0; jj < n_neighbors; jj++) {
            faiss::idx_t j_aug = local_knn.indices[i * n_neighbors + jj];
            if (j_aug < 0) continue;

            float pji = h_p[i * n_neighbors + jj];
            if (pji < 1e-12f) continue;

            int j_global;
            if (j_aug < (faiss::idx_t)local_n) {
                j_global = my_global_offset + (int)j_aug;
            } else {
                j_global = halo.ghost_global_ids[j_aug - local_n];
            }

            local_coo.push_back({i_global, j_global, pji});

            if (j_aug < (faiss::idx_t)local_n) {
                local_coo.push_back({j_global, i_global, pji});
            } else {
                mirror_coo.push_back({j_global, i_global, pji});
            }
        }
    }

    local_knn.distances.clear();
    local_knn.indices.clear();
    h_p.clear();
    halo.ghost_points.clear();
    halo.ghost_global_ids.clear();

    // 6. SHUFFLE - route mirrors to owner(j) via alltoallv
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

    // prefix sum
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
