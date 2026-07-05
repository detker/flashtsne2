#include "TsneOptimizer.hpp"
#include "ICommunicator.hpp"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <iostream>
#include <tuple>
#include <utility>

#include "NcclRing.cuh"
#include "quad_tree_builder.cuh"
#include "quad_tree_traversor.cuh"

namespace {

constexpr int kThreads = 256;

__global__ void deinterleaveKernel(const float* y, int n, float* xs, float* ys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    xs[i] = y[2 * i];
    ys[i] = y[2 * i + 1];
}

/*
    One full ring circulation (moved here from main.cu). Each hop the
    resident shard traverses this rank's tree, accumulating repulsive
    forces and partial Z; grads travel with their shard, so after
    ring.size() exchanges every shard is back home with its full gradient.
*/
static std::tuple<
    thrust::device_vector<float>, // X back
    thrust::device_vector<float>, // Y back
    thrust::device_vector<float>, // X grad
    thrust::device_vector<float>, // Y grad
    float // Z
> circulate(
    const NcclRing& ring,
    QuadTreeTraversor<TsneApproxCond, TsneNodeHanlder, TsneLeafHandler>& traversor,
    thrust::device_vector<float> x,
    thrust::device_vector<float> y
){
    size_t size{x.size()};
    thrust::device_vector<float> grad_x(size, 0.0f), grad_y(size, 0.0f);
    float z{0.0f}, tmp_z{0.0f};

    for(size_t i{0}; i < (size_t)ring.size(); i++){
        traversor.load_points(std::move(x), std::move(y));

        std::tie(grad_x, grad_y, tmp_z) = traversor.traverse(
            std::move(grad_x),
            std::move(grad_y)
        );
        std::tie(x, y) = traversor.get_points();

        z += tmp_z;

        if(ring.size() > 1){
            x = ring.ring_exchange(std::move(x));
            y = ring.ring_exchange(std::move(y));
            grad_x = ring.ring_exchange(std::move(grad_x));
            grad_y = ring.ring_exchange(std::move(grad_y));
        }
    }
    /* ring.size() in-loop exchanges already bring every shard back home */

    return std::tuple<
        thrust::device_vector<float>, // X back
        thrust::device_vector<float>, // Y back
        thrust::device_vector<float>, // X grad
        thrust::device_vector<float>, // Y grad
        float // Z
    >{
        std::move(x), std::move(y),
        std::move(grad_x), std::move(grad_y),
        z
    };
}

/*
    Inputs:
    ring - NCCL ring communicator wrapper (initialized eariler via MPI)
    x - x coordinates of low dim points
    y - y coordinates of low dim points
    theta - BH accuracy/speed trade-off
    Outputs:
    x_grad - x components of gradient for each point [num_points, 1]
    y_grad - y components of gradient for each point [num_points, 1]
    x, y - returns low dimensional points back
    z - normalization factor (local part, self terms included)
    All outputs are sorted back to the original point order.
*/
static std::tuple<
    thrust::device_vector<float>,
    thrust::device_vector<float>,
    thrust::device_vector<float>,
    thrust::device_vector<float>,
    float
> step_ring(
    const NcclRing& ring,
    thrust::device_vector<float> x,
    thrust::device_vector<float> y,
    float theta
){
    thrust::device_vector<uint32_t> permutation_idx{}, nlen{}, f_pos{}, length{};
    thrust::device_vector<uint8_t> is_leaf{};
    thrust::device_vector<float> x_com{}, y_com{}, x_grad{}, y_grad{};
    float z, face_len;

    ParallelQuadtreeBuilder tree_builder;
    tree_builder.bind_arguments(std::move(x),std::move(y));

    std::tie(
        permutation_idx, nlen, f_pos, length, is_leaf, x_com, y_com
    ) = tree_builder.build_tree();
    face_len = tree_builder.get_face_len();

    std::tie(x, y) = tree_builder.retrive_arguments();

    QuadTreeTraversor<TsneApproxCond, TsneNodeHanlder, TsneLeafHandler> traversor;

    traversor.load_tree(
        std::move(nlen),
        std::move(f_pos),
        std::move(length),
        std::move(is_leaf),
        std::move(x_com),
        std::move(y_com)
    );

    /* Leaf interactions read the local tree's points - keep a copy that
       stays home while x/y circulate between ranks */
    traversor.load_tree_points(x, y);
    traversor.set_face_lenght(face_len);
    traversor.set_theta(theta);

    std::tie(
        x, y, x_grad, y_grad, z
    ) = circulate(
        ring,
        traversor,
        std::move(x),
        std::move(y)
    );

    /* thrust::stable_sort_by_key */
    thrust::sort_by_key(
        permutation_idx.begin(),
        permutation_idx.end(),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                x.begin(),
                y.begin(),
                x_grad.begin(),
                y_grad.begin()
            )
        )
    );

    return {
        std::move(x_grad),
        std::move(y_grad),
        std::move(x),
        std::move(y),
        z
    };
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

// Repulsive grads come back from step_ring already sorted to P's row order.
__global__ void applyGradientKernel(
    const float* rep_x, const float* rep_y,
    const float* attr_x, const float* attr_y,
    float inv_Z, float eta, float momentum, float min_gain, int n,
    float* y, float* vel, float* gains)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g[2] = {
        4.0f * (attr_x[i] - rep_x[i] * inv_Z),
        4.0f * (attr_y[i] - rep_y[i] * inv_Z)
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
    const NcclRing&              ring,
    const CsrMatrix&             P,
    thrust::device_vector<float> y_init,
    size_t                       n_local,
    int64_t                      n_total,
    const TsneOptParams&         params,
    cudaStream_t                 stream)
{
    const int n = (int)n_local;
    const int rank = comm.getRank();
    const int blocks = (n + kThreads - 1) / kThreads;

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

        // 1. + 2. repulsion via the QuadTree ring step: build the local
        // tree, circulate shards; grads come home in P's row order
        thrust::device_vector<float> xs(n_local), ys(n_local);
        deinterleaveKernel<<<blocks, kThreads, 0, stream>>>(
            y.data().get(), n, xs.data().get(), ys.data().get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        thrust::device_vector<float> rep_x, rep_y;
        float z_local = 0.0f;
        std::tie(rep_x, rep_y, xs, ys, z_local) = step_ring(
            ring, std::move(xs), std::move(ys), params.theta);

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
            rep_x.data().get(), rep_y.data().get(),
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
