#include "PCAInit.hpp"
#include "ICommunicator.hpp"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <cmath>
#include <vector>

__global__ void subtractRowVecKernel(float* scores, const float* mv,
                                     size_t n, int r) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * (size_t)r) return;
    int j = idx % r;
    scores[idx] -= mv[j];
}

__global__ void extract2DKernel(const float* scores, float* out,
                                size_t n, int r, float scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i * 2 + 0] = scale * scores[i * (size_t)r + 0];
    out[i * 2 + 1] = scale * scores[i * (size_t)r + 1];
}


PCAResult distributedPCA(
    NCCLCommunicator& comm,
    const thrust::device_vector<float>& local_x,
    size_t local_n,
    int dim,
    int n_components)
{
    const int D = dim;
    const int r = std::min(n_components, dim);
    const float one = 1.0f, zero = 0.0f;

    cublasHandle_t   blas;
    cusolverDnHandle_t solver;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    const float* d_X = thrust::raw_pointer_cast(local_x.data());  // A = X^T (D x n)
    // in row-major format [n, D]

    // Uninitialized: syrk (beta=0) writes the lower triangle and the upper
    // triangle is never read (syr/syevd use FILL_MODE_LOWER) - no zero-init
    float* d_S = nullptr;
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)D * D * sizeof(float)));   // reused as C, then eigenvectors
    thrust::device_vector<float> d_a(D, 0.0f);               // column sums, then mean (mu)
    thrust::device_vector<float> d_ones(local_n, 1.0f);

    // S = A A^T  (lower triangle). syrk: C(DxD) = alpha*op(A)op(A)^T, op=N, A is D x n.
    CUBLAS_CHECK(cublasSsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             D, (int)local_n, &one,
                             d_X, D, &zero,
                             d_S, D));
    // a = A * 1_n  (D-vector of column sums of X). gemv op=N, A is D x n.
    CUBLAS_CHECK(cublasSgemv(blas, CUBLAS_OP_N, D, (int)local_n, &one,
                             d_X, D,
                             thrust::raw_pointer_cast(d_ones.data()), 1, &zero,
                             thrust::raw_pointer_cast(d_a.data()), 1));
    CUDA_CHECK(cudaDeviceSynchronize());
    d_ones.clear(); d_ones.shrink_to_fit();

    comm.allReduce<CommunicationBackend::NCCL>(
        nullptr, d_S, (size_t)D * D,
        CommDataType::FLOAT, CommOp::SUM);
    comm.allReduce<CommunicationBackend::NCCL>(
        nullptr, thrust::raw_pointer_cast(d_a.data()), (size_t)D,
        CommDataType::FLOAT, CommOp::SUM);

    thrust::device_vector<unsigned long long> d_ntot(1, (unsigned long long)local_n);
    comm.allReduce<CommunicationBackend::NCCL>(
        nullptr, thrust::raw_pointer_cast(d_ntot.data()), (size_t)1,
        CommDataType::UINT64, CommOp::SUM);
    const unsigned long long total_n = d_ntot[0];

    // C = S - (1/N) a a^T (lower), then eig
    const float neg_inv_N = -1.0f / (float)total_n;
    CUBLAS_CHECK(cublasSsyr(blas, CUBLAS_FILL_MODE_LOWER, D, &neg_inv_N,
                            thrust::raw_pointer_cast(d_a.data()), 1,
                            d_S, D));
    const float inv_N = 1.0f / (float)total_n;
    CUBLAS_CHECK(cublasSscal(blas, D, &inv_N,
                             thrust::raw_pointer_cast(d_a.data()), 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    // syevd: eigenvalues ascending in d_W, eigenvectors overwrite d_S (col-major)
    thrust::device_vector<float> d_W(D);
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(
        solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, D,
        d_S, D,
        thrust::raw_pointer_cast(d_W.data()), &lwork));
    float* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(float)));

    CUSOLVER_CHECK(cusolverDnSsyevd(
        solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, D,
        d_S, D,
        thrust::raw_pointer_cast(d_W.data()),
        d_work, lwork, d_info));
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "PCA: syevd failed to converge (info=%d)\n", h_info);
        exit(1);
    }

    // build V (D x r, col-major) = top-r eigenvectors in DESCENDING order
    thrust::device_vector<float> d_V((size_t)D * r);
    const float* eigvecs = d_S;
    float* Vp = thrust::raw_pointer_cast(d_V.data());
    for (int j = 0; j < r; ++j) {
        CUDA_CHECK(cudaMemcpy(Vp + (size_t)j * D,
                              eigvecs + (size_t)(D - 1 - j) * D,
                              (size_t)D * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    // scores Z^T = V^T A, col-major r x n
    // scores Z here in row-major n x r 
    PCAResult res;
    res.scores.resize((size_t)local_n * r);
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                             r, (int)local_n, D, &one,
                             Vp, D,           // op(A) = V^T (r x D)
                             d_X, D,          // op(B) = A   (D x n)
                             &zero,
                             thrust::raw_pointer_cast(res.scores.data()), r));

    // center the scores
    thrust::device_vector<float> d_mv(r);
    CUBLAS_CHECK(cublasSgemv(blas, CUBLAS_OP_T, D, r, &one,
                             Vp, D,
                             thrust::raw_pointer_cast(d_a.data()), 1, &zero,
                             thrust::raw_pointer_cast(d_mv.data()), 1));
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        size_t total = (size_t)local_n * r;
        int block = 256;
        size_t grid = (total + block - 1) / block;
        subtractRowVecKernel<<<grid, block>>>(
            thrust::raw_pointer_cast(res.scores.data()),
            thrust::raw_pointer_cast(d_mv.data()), local_n, r);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // scale PC1, PC2 for tsne init
    const float* zp = thrust::raw_pointer_cast(res.scores.data());
    const int rr = r;
    // accumulate sum and sumsq together.
    float2 local_ss = thrust::transform_reduce(
        thrust::device,
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(local_n),
        [=] __device__ (size_t i) -> float2 { float v = zp[i * (size_t)rr]; return make_float2(v, v * v); },
        make_float2(0.0f, 0.0f),
        [] __device__ (float2 a, float2 b) -> float2 { return make_float2(a.x + b.x, a.y + b.y); });
    const float local_sum   = local_ss.x;
    const float local_sumsq = local_ss.y;

    thrust::device_vector<float> d_stats(2);
    d_stats[0] = local_sum;
    d_stats[1] = local_sumsq;
    comm.allReduce<CommunicationBackend::NCCL>(
        nullptr, thrust::raw_pointer_cast(d_stats.data()), (size_t)2,
        CommDataType::FLOAT, CommOp::SUM);
    const float g_sum   = d_stats[0];
    const float g_sumsq = d_stats[1];
    const float mean = g_sum / (float)total_n;
    const float var  = g_sumsq / (float)total_n - mean * mean;
    const float std  = std::sqrt(var > 0.0f ? var : 1.0f);
    res.init_scale = 1e-4f / (std > 0.0f ? std : 1.0f);

    res.components = std::move(d_V);
    res.r = r;
    res.local_n = local_n;

    cudaFree(d_S);
    cudaFree(d_work);
    cudaFree(d_info);
    cublasDestroy(blas);
    cusolverDnDestroy(solver);
    return res;
}


thrust::device_vector<float> extractInit2D(
    const thrust::device_vector<float>& scores,
    size_t n,
    int r,
    float scale)
{
    thrust::device_vector<float> y_init(n * 2);
    int block = 256;
    size_t grid = (n + block - 1) / block;
    extract2DKernel<<<grid, block>>>(
        thrust::raw_pointer_cast(scores.data()),
        thrust::raw_pointer_cast(y_init.data()), n, r, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    return y_init;
}
