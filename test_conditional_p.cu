#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_BINARY_SEARCH_ITERS = 100;
static constexpr float PERPLEXITY_TOL = 1e-5f;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(1);                                                   \
    }                                                              \
} while(0)


__global__ void computeConditionalP_naive(
    float* __restrict__ dist_p,
    int n,
    int stride,
    float target_entropy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float* row = dist_p + (int64_t)i * stride;

    float beta_min = 1e-10f;
    float beta_max = 1e10f;
    float beta = 1.0f;

    for (int iter = 0; iter < MAX_BINARY_SEARCH_ITERS; iter++) {
        float sum_exp = 0.0f;
        float entropy = 0.0f;
        for (int j = 1; j < stride; j++) {
            float val = expf(-beta * row[j]);
            sum_exp += val;
        }
        if (sum_exp < 1e-30f) sum_exp = 1e-30f;

        float inv_sum = 1.0f / sum_exp;
        for (int j = 1; j < stride; j++) {
            float p = expf(-beta * row[j]) * inv_sum;
            if (p > 1e-30f) {
                entropy -= p * logf(p);
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
    for (int j = 1; j < stride; j++) {
        sum_exp += expf(-beta * row[j]);
    }
    if (sum_exp < 1e-30f) sum_exp = 1e-30f;
    float inv_sum = 1.0f / sum_exp;
    for (int j = 1; j < stride; j++) {
        row[j] = expf(-beta * row[j]) * inv_sum;
    }
}


// TODO: implement warp-per-row version here
// __global__ void computeConditionalP_warp(...)
__global__ void computeConditionalP_warp(
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


void launchNaive(float* d_dist, int n, int stride, float target_entropy, cudaStream_t stream) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeConditionalP_naive<<<grid, BLOCK_SIZE, 0, stream>>>(d_dist, n, stride, target_entropy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void launchWarp(float* d_dist, int n, int stride, float target_entropy, cudaStream_t stream) {
    const int BLOCK_SIZE = 256;
    const int WARP_SIZE = 32;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    dim3 block(32, warps_per_block);
    int grid = (n + warps_per_block - 1) / warps_per_block;
    computeConditionalP_warp<<<grid, block, 0, stream>>>(d_dist, n, stride, target_entropy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


int main() {
    int n = 1024;
    int n_neighbors = 90;
    int stride = n_neighbors + 1;
    float perplexity = 30.0f;
    float target_entropy = logf(perplexity);

    // Generate synthetic distances: row[0] = 0 (self), row[1..stride-1] = random positive
    std::vector<float> h_dist(n * stride);
    srand(42);
    for (int i = 0; i < n; i++) {
        h_dist[i * stride] = 0.0f;
        for (int j = 1; j < stride; j++) {
            h_dist[i * stride + j] = 0.1f + (float)rand() / RAND_MAX * 10.0f;
        }
    }

    // Run naive kernel
    thrust::device_vector<float> d_dist_naive(h_dist.begin(), h_dist.end());
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    launchNaive(thrust::raw_pointer_cast(d_dist_naive.data()), n, stride, target_entropy, stream);

    std::vector<float> h_result_naive(n * stride);
    CUDA_CHECK(cudaMemcpy(h_result_naive.data(), thrust::raw_pointer_cast(d_dist_naive.data()),
               n * stride * sizeof(float), cudaMemcpyDeviceToHost));

    // Validate: each row's probabilities (slots 1..stride-1) should sum to ~1.0
    // and the entropy should match target_entropy
    float max_sum_err = 0.0f;
    float max_entropy_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        float entropy = 0.0f;
        for (int j = 1; j < stride; j++) {
            float p = h_result_naive[i * stride + j];
            sum += p;
            if (p > 1e-30f) {
                entropy -= p * logf(p);
            }
        }
        float sum_err = fabsf(sum - 1.0f);
        float entropy_err = fabsf(entropy - target_entropy);
        if (sum_err > max_sum_err) max_sum_err = sum_err;
        if (entropy_err > max_entropy_err) max_entropy_err = entropy_err;
    }

    printf("=== Naive kernel results ===\n");
    printf("  n=%d, k=%d, perplexity=%.1f\n", n, n_neighbors, perplexity);
    printf("  Max sum error:     %.2e (should be < 1e-5)\n", max_sum_err);
    printf("  Max entropy error: %.2e (should be < 1e-4)\n", max_entropy_err);

    bool pass = (max_sum_err < 1e-5f) && (max_entropy_err < 1e-4f);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");

    thrust::device_vector<float> d_dist_warp(h_dist.begin(), h_dist.end());
    launchWarp(thrust::raw_pointer_cast(d_dist_warp.data()), n, stride, target_entropy, stream);
    
    std::vector<float> h_result_warp(n * stride);
    CUDA_CHECK(cudaMemcpy(h_result_warp.data(), thrust::raw_pointer_cast(d_dist_warp.data()),
               n * stride * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    for (int i = 0; i < n * stride; i++) {
        float diff = fabsf(h_result_naive[i] - h_result_warp[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("=== Warp kernel vs Naive ===\n");
    printf("  Max element-wise diff: %.2e (should be < 1e-5)\n", max_diff);
    printf("  %s\n", max_diff < 1e-5f ? "PASS" : "FAIL");

    CUDA_CHECK(cudaStreamDestroy(stream));
    return pass ? 0 : 1;
}
