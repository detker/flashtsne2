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
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/tuple.h>

#include <cusparse.h>

#include <cmath>
#include <iostream>
#include <vector>


static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_BINARY_SEARCH_ITERS = 100;
static constexpr float PERPLEXITY_TOL = 1e-5f;

using MPI_B = CommunicationBackend::MPI;
using NCCL_B = CommunicationBackend::NCCL;


__global__ void computeConditionalP(
    float* __restrict__ dist_p,
    int n,
    int stride,
    float target_entropy)
{
    int pid = blockIdx.x * blockDim.y + threadIdx.y; // one warp per row
    if (pid >= n) return;

    float *row = dist_p + (int64_t)pid * stride;

    float beta_min = 1e-10f;
    float beta_max = 1e10f;
    float beta = 1.0f;

    for (int iter = 0; iter < MAX_BINARY_SEARCH_ITERS; iter++) {
        float sum_exp = 0.0f;
        float entropy = 0.0f;
        for (int j = threadIdx.x + 1; j < stride; j += blockDim.x) {
            sum_exp += __expf(-beta * row[j]);
        }
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        }
        sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
        if (sum_exp < 1e-30f) sum_exp = 1e-30f;
        
        float inv_sum = 1.0f / sum_exp;
        float local_entropy = 0.0f;
        for (int j = threadIdx.x + 1; j < stride; j += blockDim.x) {
            float p = __expf(-beta * row[j]) * inv_sum;
            float val = p > 1e-30f ? p * __logf(p) : 0.0f;
            local_entropy += val;
        }
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            local_entropy += __shfl_down_sync(0xffffffff, local_entropy, offset);
        }
        entropy -= __shfl_sync(0xffffffff, local_entropy, 0);

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
    for (int j = threadIdx.x + 1; j < stride; j += blockDim.x) {
        sum_exp += __expf(-beta * row[j]);
    }
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
    if (sum_exp < 1e-30f) sum_exp = 1e-30f;
    float inv_sum = 1.0f / sum_exp;
    for (int j = threadIdx.x + 1; j < stride; j += blockDim.x) {
        row[j] = __expf(-beta * row[j]) * inv_sum;
    }
}


struct KNNResult {
    thrust::device_vector<float> d_distances;    // [query_n * search_k] on GPU
    thrust::device_vector<faiss::idx_t> d_indices; // [query_n * search_k] on GPU
    int search_k;
};


static KNNResult runLocalKNN(
    float* __restrict__ d_index_data, size_t index_n,
    float* __restrict__ d_query_data, size_t query_n,
    int dim, int n_neighbors)
{
    int current_dev = 0;
    cudaGetDevice(&current_dev);

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = current_dev;
    faiss::gpu::GpuIndexFlatL2 gpu_index(&res, dim, config);

    gpu_index.add(index_n, d_index_data);

    int search_k = n_neighbors + 1;

    KNNResult result;
    result.search_k = search_k;
    result.d_distances.resize(query_n * search_k);
    result.d_indices.resize(query_n * search_k);

    gpu_index.search(
        query_n,
        d_query_data,
        search_k,
        thrust::raw_pointer_cast(result.d_distances.data()),
        thrust::raw_pointer_cast(result.d_indices.data())
    );

    return result;
}


__global__ void emitCOO(
    const float* __restrict__ d_p,
    const faiss::idx_t* __restrict__ d_indices,
    int64_t* __restrict__ d_keys,
    float* __restrict__ d_vals,
    int local_n,
    int search_k,
    int global_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_neighbors = search_k - 1;
    int total_slots = local_n * n_neighbors;
    if (tid >= total_slots) return;

    int i = tid / n_neighbors;
    int jj = tid % n_neighbors;
    int src_idx = i * search_k + (jj + 1);

    int out_fwd = tid;
    int out_mir = total_slots + tid;

    faiss::idx_t j_local = d_indices[src_idx];
    float pji = d_p[src_idx];

    if (j_local < 0 || pji < 1e-12f) {
        d_keys[out_fwd] = -1;
        d_keys[out_mir] = -1;
        d_vals[out_fwd] = 0.0f;
        d_vals[out_mir] = 0.0f;
        return;
    }

    int i_global = global_offset + i;
    int j_global = global_offset + (int)j_local;

    d_keys[out_fwd] = ((int64_t)i_global << 32) | (int64_t)(unsigned int)j_global;
    d_vals[out_fwd] = pji;

    d_keys[out_mir] = ((int64_t)j_global << 32) | (int64_t)(unsigned int)i_global;
    d_vals[out_mir] = pji;
}


SparseMatrix buildSparseP(
    NCCLCommunicator& comm,
    float* d_local_data,
    size_t local_n,
    int dim,
    int n_neighbors,
    float perplexity,
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
    // [my_global_offset, my_global_offset + local_n)

    if (rank == 0) std::cout << "Total points across all ranks: " << N_total << std::endl;

    // 1. LOCAL kNN
    int device = 0;
    cudaGetDevice(&device);
    std::cout << "Step 2a: RANK: " << rank << " DEVICE: " << device << ". Local kNN (k=" << n_neighbors << ")..." << std::endl;

    KNNResult local_knn = runLocalKNN(
        d_local_data, local_n,
        d_local_data, local_n,
        dim, n_neighbors);

    int search_k = local_knn.search_k;

    // 2. COMPUTE CONDITIONAL P(j|i) - skips slot 0 (self) internally
    if (rank == 0) std::cout << "Step 2b: Computing conditional probabilities (perplexity="
                             << perplexity << ")..." << std::endl;

    float target_entropy = logf(perplexity);
    const int WARP_SIZE = 32;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    dim3 block(WARP_SIZE, warps_per_block);
    int grid = (local_n + warps_per_block - 1) / warps_per_block;

    computeConditionalP<<<grid, block, 0, stream>>>(
        thrust::raw_pointer_cast(local_knn.d_distances.data()),
        local_n,
        search_k,
        target_entropy
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 3. EMIT COO on GPU (skip slot 0 = self)
    if (rank == 0) std::cout << "Step 2c: Emitting COO entries on GPU..." << std::endl;

    int n_neighbors_actual = search_k - 1;
    int64_t total_slots = (int64_t)local_n * n_neighbors_actual;
    int64_t coo_capacity = total_slots * 2;

    thrust::device_vector<int64_t> d_keys(coo_capacity);
    thrust::device_vector<float> d_vals(coo_capacity);

    int emit_grid = (total_slots + BLOCK_SIZE - 1) / BLOCK_SIZE;
    emitCOO<<<emit_grid, BLOCK_SIZE, 0, stream>>>(
        thrust::raw_pointer_cast(local_knn.d_distances.data()),
        thrust::raw_pointer_cast(local_knn.d_indices.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_vals.data()),
        local_n, search_k, my_global_offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    local_knn.d_distances.clear(); local_knn.d_distances.shrink_to_fit();
    local_knn.d_indices.clear(); local_knn.d_indices.shrink_to_fit();

    // Remove invalid entries (key == -1)
    auto valid_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_vals.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_keys.end(), d_vals.end())),
        [] __device__ (const thrust::tuple<int64_t, float>& t) {
            return thrust::get<0>(t) < 0;
        }
    );
    int64_t n_valid = valid_end - thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_vals.begin()));
    d_keys.resize(n_valid);
    d_vals.resize(n_valid);

    // 4. SORT by key (row,col packed into int64) then reduce duplicates
    if (rank == 0) std::cout << "Step 2d: Sort + reduce on GPU..." << std::endl;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    thrust::device_vector<int64_t> d_unique_keys(n_valid);
    thrust::device_vector<float> d_reduced_vals(n_valid);

    auto reduce_end = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(),
        d_vals.begin(),
        d_unique_keys.begin(),
        d_reduced_vals.begin());

    int64_t nnz = reduce_end.first - d_unique_keys.begin();
    d_unique_keys.resize(nnz);
    d_reduced_vals.resize(nnz);
    d_keys.clear(); d_keys.shrink_to_fit();
    d_vals.clear(); d_vals.shrink_to_fit();

    // 5. Scale by 1/(2*N_total), extract local rows + global cols, filter
    if (rank == 0) std::cout << "Step 2e: Scale + extract CSR..." << std::endl;

    float norm = 1.0f / (2.0f * N_total);
    thrust::transform(d_reduced_vals.begin(), d_reduced_vals.end(),
                      d_reduced_vals.begin(),
                      [norm] __device__ (float v) { return v * norm; });

    // Filter: keep only rows owned by this rank and vals > threshold
    thrust::device_vector<int> d_coo_rows(nnz);
    thrust::device_vector<int> d_coo_cols(nnz);

    // Unpack keys into row/col
    thrust::transform(d_unique_keys.begin(), d_unique_keys.end(),
        thrust::make_zip_iterator(thrust::make_tuple(d_coo_rows.begin(), d_coo_cols.begin())),
        [my_global_offset] __device__ (int64_t key) {
            int row = (int)(key >> 32);
            int col = (int)(key & 0xFFFFFFFF);
            return thrust::make_tuple(row - my_global_offset, col);
        });
    d_unique_keys.clear(); d_unique_keys.shrink_to_fit();

    // Filter to only local rows with significant values
    auto keep_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(
            d_coo_rows.begin(), d_coo_cols.begin(), d_reduced_vals.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_coo_rows.end(), d_coo_cols.end(), d_reduced_vals.end())),
        [local_n] __device__ (const thrust::tuple<int, int, float>& t) {
            int row = thrust::get<0>(t);
            float val = thrust::get<2>(t);
            return row < 0 || row >= (int)local_n || val < 1e-12f;
        });
    int64_t nnz_final = keep_end - thrust::make_zip_iterator(thrust::make_tuple(
        d_coo_rows.begin(), d_coo_cols.begin(), d_reduced_vals.begin()));
    d_coo_rows.resize(nnz_final);
    d_coo_cols.resize(nnz_final);
    d_reduced_vals.resize(nnz_final);

    // 6. COO->CSR via cusparseXcoo2csr
    if (rank == 0) std::cout << "Step 2f: COO->CSR on GPU (cusparseXcoo2csr)..." << std::endl;

    SparseMatrix mat;
    mat.n_rows = local_n;
    mat.n_cols = N_total;
    mat.nnz = nnz_final;
    mat.global_row_offset = my_global_offset;
    mat.col_indices = std::move(d_coo_cols);
    mat.values = std::move(d_reduced_vals);
    mat.row_offsets.resize(local_n + 1);

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUSPARSE_CHECK(cusparseXcoo2csr(
        handle,
        thrust::raw_pointer_cast(d_coo_rows.data()),
        nnz_final,
        local_n,
        thrust::raw_pointer_cast(mat.row_offsets.data()),
        CUSPARSE_INDEX_BASE_ZERO
    ));
    d_coo_rows.clear(); d_coo_rows.shrink_to_fit();

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

    CUSPARSE_CHECK(cusparseDestroy(handle));

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
