#include "SymmetrizeP.hpp"
#include "error_utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cmath>


static constexpr int   BLOCK_SIZE             = 256;
static constexpr int   MAX_BINARY_SEARCH_ITERS = 100;
static constexpr float PERPLEXITY_TOL          = 1e-5f;


// 1. P(j|i) via per-row beta (precision) binary search to match target perplexity.
//    Overwrites the distance row in place: row[k] <- P(neighbor_k | i).
__global__ void conditionalP(
    float* __restrict__ dist,
    int N, int K,
    float target_entropy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float* row = dist + (int64_t)i * K;

    float beta_min = 1e-10f;
    float beta_max = 1e10f;
    float beta     = 1.0f;

    for (int iter = 0; iter < MAX_BINARY_SEARCH_ITERS; iter++) {
        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            sum_exp += expf(-beta * row[k]);
        }
        if (sum_exp < 1e-30f) sum_exp = 1e-30f;
        float inv_sum = 1.0f / sum_exp;

        float entropy = 0.0f;
        for (int k = 0; k < K; k++) {
            float p = expf(-beta * row[k]) * inv_sum;
            if (p > 1e-30f) entropy -= p * logf(p);
        }

        float diff = entropy - target_entropy;
        if (fabsf(diff) < PERPLEXITY_TOL) break;

        if (diff > 0.0f) {                 // entropy too high -> increase beta
            beta_min = beta;
            beta = (beta_max >= 1e9f) ? beta * 2.0f : 0.5f * (beta + beta_max);
        } else {                           // entropy too low -> decrease beta
            beta_max = beta;
            beta = (beta_min <= 1e-9f) ? beta * 0.5f : 0.5f * (beta + beta_min);
        }
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < K; k++) sum_exp += expf(-beta * row[k]);
    if (sum_exp < 1e-30f) sum_exp = 1e-30f;
    float inv_sum = 1.0f / sum_exp;
    for (int k = 0; k < K; k++) row[k] = expf(-beta * row[k]) * inv_sum;
}


// 2. Emit both directions. For pair (i, j=nbr, p): write (i,j,p) and (j,i,p),
//    each as a packed key = row*N + col plus the value.
__global__ void emitTriplets(
    const int*   __restrict__ nbr,
    const float* __restrict__ P,
    int64_t*     __restrict__ keys,
    float*       __restrict__ vals,
    int N, int K)
{
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)N * K) return;

    int   i = (int)(t / K);
    int   j = nbr[t];
    float p = P[t];

    int64_t base = 2 * t;
    keys[base]     = (int64_t)i * N + j;   vals[base]     = p;  // (i, j)
    keys[base + 1] = (int64_t)j * N + i;   vals[base + 1] = p;  // (j, i)
}


// key -> column index
struct KeyToCol {
    int N;
    __host__ __device__ int operator()(int64_t key) const { return (int)(key % N); }
};


CsrMatrix buildSymmetricP(
    const KnnGraph& knn,
    float           perplexity,
    cudaStream_t    stream)
{
    auto policy = thrust::cuda::par.on(stream);

    const int     N  = (int)knn.n_local;
    const int     K  = knn.k;
    const int64_t NK = (int64_t)N * K;
    const int*    d_neighbors = thrust::raw_pointer_cast(knn.indices.data());

    // 1. neighbor distances: reuse the squared-L2 distances FAISS already wrote
    //    into the k-NN graph. Copy them into a working buffer so conditionalP can
    //    overwrite it with P(j|i) without clobbering knn.distances.
    thrust::device_vector<float> d_dist(NK);
    thrust::copy(policy, knn.distances.begin(), knn.distances.begin() + NK,
                 d_dist.begin());

    // 2. conditional P(j|i) (in place over distances) ----------------------
    {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        conditionalP<<<grid, BLOCK_SIZE, 0, stream>>>(
            thrust::raw_pointer_cast(d_dist.data()), N, K, logf(perplexity));
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. emit 2*N*K triplets (both directions) -----------------------------
    thrust::device_vector<int64_t> d_keys(2 * NK);
    thrust::device_vector<float>   d_vals(2 * NK);
    {
        int grid = (int)((NK + BLOCK_SIZE - 1) / BLOCK_SIZE);
        emitTriplets<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_neighbors,
            thrust::raw_pointer_cast(d_dist.data()),
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_vals.data()), N, K);
        CUDA_CHECK(cudaGetLastError());
    }
    d_dist.clear(); d_dist.shrink_to_fit();

    // 4. sort by (row,col) key, then sum duplicates ------------------------
    //    sum of equal keys == P(j|i) + P(i|j)
    thrust::sort_by_key(policy, d_keys.begin(), d_keys.end(), d_vals.begin());

    thrust::device_vector<int64_t> d_ukeys(2 * NK);
    thrust::device_vector<float>   d_uvals(2 * NK);
    auto end = thrust::reduce_by_key(
        policy,
        d_keys.begin(), d_keys.end(),   // keys in
        d_vals.begin(),                 // values in
        d_ukeys.begin(),                // unique keys out
        d_uvals.begin());               // summed values out

    int64_t nnz = end.first - d_ukeys.begin();
    d_keys.clear(); d_keys.shrink_to_fit();
    d_vals.clear(); d_vals.shrink_to_fit();

    // 5. scale by 1/(2N) and pack CSR --------------------------------------
    CsrMatrix M;
    M.n   = N;
    M.nnz = nnz;

    // values
    M.values.resize(nnz);
    float norm = 1.0f / (2.0f * (float)N);
    thrust::transform(policy,
        d_uvals.begin(), d_uvals.begin() + nnz,
        M.values.begin(),
        [norm] __device__ (float v) { return v * norm; });

    // column indices = key % N
    M.col_indices.resize(nnz);
    thrust::transform(policy,
        d_ukeys.begin(), d_ukeys.begin() + nnz,
        M.col_indices.begin(),
        KeyToCol{N});

    // row offsets: keys are sorted ascending, so row = key / N is nondecreasing.
    // row_offsets[r] = first index whose row >= r, for r in [0, N].
    M.row_offsets.resize(N + 1);
    thrust::counting_iterator<int64_t> rows_begin(0);
    thrust::lower_bound(
        policy,
        d_ukeys.begin(), d_ukeys.begin() + nnz,           // sorted keys
        rows_begin, rows_begin + (N + 1),                 // search points r*N ...
        M.row_offsets.begin(),
        [N] __device__ (int64_t key, int64_t r) {         // key < r  <=>  row(key) < r
            return key < r * (int64_t)N;
        });

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return M;
}
