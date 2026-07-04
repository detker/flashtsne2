#include "TsneOptimizer.hpp"
#include "ICommunicator.hpp"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <cassert>
#include <iostream>
#include <tuple>

#include "quad_tree_builder.cuh"
#include "quad_tree_traversor.cuh"  // TsneApproxCond/NodeHanlder/LeafHandler, add_vec3

namespace {

constexpr int kThreads = 256;

__global__ void deinterleaveKernel(const float* y, int n, float* xs, float* ys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    xs[i] = y[2 * i];
    ys[i] = y[2 * i + 1];
}

// Barnes-Hut repulsion. Reimplements QuadTreeGpu's traverse_impl with the
// tree's own (sorted) points separate from the visiting query points: the
// leaf loop must dereference tree points, which are not the queries once
// shards circulate between ranks. Self-interaction is not skipped either -
// on the home hop each point meets itself in a leaf with zero force and
// exactly +1 to Z, removed once globally as n_total.
__global__ void bhRepulsionKernel(
    const uint32_t* nlen, const uint32_t* f_pos, const uint32_t* length,
    const uint8_t* is_leaf, const float* x_com, const float* y_com,
    const float* px, const float* py,  // tree points (this rank, sorted)
    const float* qx, const float* qy,  // visiting query points
    float face_len, float theta, int n_query,
    float* gx, float* gy, float* gz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_query) return;

    TsneApproxCond approx_cond;
    TsneNodeHanlder node_handler;
    TsneLeafHandler leaf_handler;

    const float x = qx[tid];
    const float y = qy[tid];

    float3 res{0.0f, 0.0f, 0.0f};
    uint32_t stack[4 * QUAD_TREE_MAX_HEIGHT];
    uint8_t depth_stack[4 * QUAD_TREE_MAX_HEIGHT];
    int top = 0;
    stack[0] = 0;
    depth_stack[0] = 0;

    while (top >= 0) {
        const uint32_t node = stack[top];
        const uint8_t depth = depth_stack[top];
        top--;

        if (is_leaf[node]) {
            for (uint32_t i = 0; i < length[node]; i++) {
                const uint32_t p = f_pos[node] + i;
                add_vec3(res, leaf_handler(px[p], py[p], x, y));
            }
            continue;
        }

        if (approx_cond(x_com[node], y_com[node], x, y, face_len, depth, theta)) {
            add_vec3(res, node_handler(x_com[node], y_com[node], x, y, nlen[node]));
            continue;
        }

        for (uint32_t i = 0; i < length[node]; i++) {
            top++;
            stack[top] = f_pos[node] + i;
            depth_stack[top] = depth + 1;
        }
    }

    gx[tid] += res.x;
    gy[tid] += res.y;
    gz[tid] += res.z;
}

// F_attr_i = sum_j p_ij * (1 + ||y_i - y_j||^2)^-1 * (y_i - y_j), CSR row per
// thread (P is local: col indices in [0, n_local)).
__global__ void attractiveKernel(
    const int* row_offsets, const int* col_indices, const float* values,
    const float* y, float exaggeration, int n,
    float* ax, float* ay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float yx = y[2 * i], yy = y[2 * i + 1];
    float sx = 0.0f, sy = 0.0f;
    for (int e = row_offsets[i]; e < row_offsets[i + 1]; e++) {
        const int j = col_indices[e];
        const float dx = yx - y[2 * j];
        const float dy = yy - y[2 * j + 1];
        const float w = values[e] / (1.0f + dx * dx + dy * dy);
        sx += w * dx;
        sy += w * dy;
    }
    ax[i] = exaggeration * sx;
    ay[i] = exaggeration * sy;
}

// Repulsive grads arrive in the tree builder's spatial-sort order; ids maps
// sorted position k -> original row i, so state (y, vel, gains, attr) stays
// in P's row order across iterations.
__global__ void applyGradientKernel(
    const float* ids,
    const float* rep_x, const float* rep_y,    // sorted order
    const float* attr_x, const float* attr_y,  // original order
    float inv_Z, float eta, float momentum, float min_gain, int n,
    float* y, float* vel, float* gains)        // original order
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int i = (int)ids[k];
    const float g[2] = {
        4.0f * (attr_x[i] - rep_x[k] * inv_Z),
        4.0f * (attr_y[i] - rep_y[k] * inv_Z)
    };
    for (int d = 0; d < 2; d++) {
        const int idx = 2 * i + d;
        float gain = (g[d] * vel[idx] < 0.0f) ? gains[idx] + 0.2f
                                              : gains[idx] * 0.8f;
        if (gain < min_gain) gain = min_gain;
        const float v = momentum * vel[idx] - eta * gain * g[d];
        gains[idx] = gain;
        vel[idx] = v;
        y[idx] += v;
    }
}

__global__ void sumPointsKernel(const float* y, int n, float* sums)  // sums[2]
{
    __shared__ float sx[kThreads];
    __shared__ float sy[kThreads];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sx[threadIdx.x] = (i < n) ? y[2 * i] : 0.0f;
    sy[threadIdx.x] = (i < n) ? y[2 * i + 1] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            sx[threadIdx.x] += sx[threadIdx.x + s];
            sy[threadIdx.x] += sy[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&sums[0], sx[0]);
        atomicAdd(&sums[1], sy[0]);
    }
}

__global__ void subtractMeanKernel(float* y, int n, float mean_x, float mean_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[2 * i]     -= mean_x;
    y[2 * i + 1] -= mean_y;
}

} // namespace

thrust::device_vector<float> optimizeTsne(
    NCCLCommunicator&            comm,
    const CsrMatrix&             P,
    thrust::device_vector<float> y_init,
    size_t                       n_local,
    int64_t                      n_total,
    const TsneOptParams&         params,
    cudaStream_t                 stream)
{
    const int n = (int)n_local;
    const int size = comm.getSize();
    const int rank = comm.getRank();
    const int blocks = (n + kThreads - 1) / kThreads;
    // the sort permutation rides the builder's float mass slot; exact only
    // while local indices fit a float mantissa
    assert(n_local < (1u << 24));

    thrust::device_vector<float> y = std::move(y_init);
    thrust::device_vector<float> vel(2 * n_local, 0.0f);
    thrust::device_vector<float> gains(2 * n_local, 1.0f);
    thrust::device_vector<float> attr_x(n_local), attr_y(n_local);
    thrust::device_vector<float> d_sums(2);

    for (int iter = 0; iter < params.n_iter; iter++) {
        const float exaggeration =
            (iter < params.exaggeration_stop) ? params.early_exaggeration : 1.0f;
        const float momentum =
            (iter < params.momentum_switch) ? params.momentum_start : params.momentum_final;

        // 1. local quadtree; ids in the mass slot give back the permutation
        thrust::device_vector<float> xs(n_local), ys(n_local), ids(n_local);
        deinterleaveKernel<<<blocks, kThreads, 0, stream>>>(
            y.data().get(), n, xs.data().get(), ys.data().get());
        thrust::sequence(ids.begin(), ids.end());

        ParallelQuadtreeBuilder builder(std::move(xs), std::move(ys), std::move(ids));
        thrust::device_vector<uint32_t> nlen, f_pos, length;
        thrust::device_vector<uint8_t> is_leaf;
        thrust::device_vector<float> x_com, y_com;
        std::tie(nlen, f_pos, length, is_leaf, x_com, y_com) = builder.build_tree();
        const float face_len = builder.get_face_len();

        thrust::device_vector<float> px, py, sorted_ids;
        std::tie(px, py, sorted_ids) = builder.retrive_arguments();

        // 2. circulate shards; repulsive grads travel with their points and
        // are back home (in sorted order) after `size` exchanges
        thrust::device_vector<float> qx(px), qy(py);
        thrust::device_vector<float> rep_x(n_local, 0.0f), rep_y(n_local, 0.0f);
        float z_local = 0.0f;

        for (int hop = 0; hop < size; hop++) {
            const int n_visit = (int)qx.size();
            thrust::device_vector<float> qz(n_visit, 0.0f);
            const int visit_blocks = (n_visit + kThreads - 1) / kThreads;
            bhRepulsionKernel<<<visit_blocks, kThreads, 0, stream>>>(
                nlen.data().get(), f_pos.data().get(), length.data().get(),
                is_leaf.data().get(), x_com.data().get(), y_com.data().get(),
                px.data().get(), py.data().get(),
                qx.data().get(), qy.data().get(),
                face_len, params.theta, n_visit,
                rep_x.data().get(), rep_y.data().get(), qz.data().get());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            z_local += thrust::reduce(qz.begin(), qz.end());

            if (size > 1) {
                qx    = comm.ringExchange(std::move(qx));
                qy    = comm.ringExchange(std::move(qy));
                rep_x = comm.ringExchange(std::move(rep_x));
                rep_y = comm.ringExchange(std::move(rep_y));
            }
        }

        // 3. global Z; drop the n_total exact self terms
        float z_global = 0.0f;
        comm.allReduce<CommunicationBackend::MPI>(
            &z_local, &z_global, 1, CommDataType::FLOAT, CommOp::SUM);
        z_global -= (float)n_total;
        if (z_global < 1e-12f) z_global = 1e-12f;

        attractiveKernel<<<blocks, kThreads, 0, stream>>>(
            P.row_offsets.data().get(), P.col_indices.data().get(),
            P.values.data().get(), y.data().get(), exaggeration, n,
            attr_x.data().get(), attr_y.data().get());

        applyGradientKernel<<<blocks, kThreads, 0, stream>>>(
            sorted_ids.data().get(), rep_x.data().get(), rep_y.data().get(),
            attr_x.data().get(), attr_y.data().get(),
            1.0f / z_global, params.eta, momentum, params.min_gain, n,
            y.data().get(), vel.data().get(), gains.data().get());

        // 4. keep the embedding zero-centered globally
        thrust::fill(d_sums.begin(), d_sums.end(), 0.0f);
        sumPointsKernel<<<blocks, kThreads, 0, stream>>>(
            y.data().get(), n, d_sums.data().get());
        float h_sums[2];
        CUDA_CHECK(cudaMemcpyAsync(h_sums, d_sums.data().get(),
                                   2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float h_tot[2];
        comm.allReduce<CommunicationBackend::MPI>(
            h_sums, h_tot, 2, CommDataType::FLOAT, CommOp::SUM);
        subtractMeanKernel<<<blocks, kThreads, 0, stream>>>(
            y.data().get(), n, h_tot[0] / (float)n_total, h_tot[1] / (float)n_total);

        if (rank == 0 && (iter % 50 == 0 || iter == params.n_iter - 1)) {
            std::cout << "t-SNE iter " << iter << "/" << params.n_iter
                      << " Z=" << z_global
                      << " exaggeration=" << exaggeration
                      << " momentum=" << momentum << std::endl;
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return y;
}
