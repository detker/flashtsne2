#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cstddef>
#include <cstdint>

#include "SymmetrizeP.hpp"

class NCCLCommunicator;

// Barnes-Hut t-SNE gradient descent (van der Maaten scheme: gains +
// momentum, early exaggeration).
//
// Per iteration:
//   1. build a local quadtree over this rank's 2D points (QuadTreeGpu)
//   2. circulate every rank's point shard around the NCCL ring; each hop
//      accumulates repulsive forces + partial Z of the visiting shard
//      against the local tree, so after `size` hops each shard returns
//      home with its full repulsive gradient
//   3. allreduce Z, compute attractive forces from the local CSR P
//   4. grad_i = 4 * (exaggeration * F_attr_i - F_rep_i / Z), apply update
struct TsneOptParams {
    int   n_iter             = 1000;
    float eta                = 200.0f;  // learning rate
    float momentum_start     = 0.5f;
    float momentum_final     = 0.8f;
    int   momentum_switch    = 250;     // iteration switching momentum
    float early_exaggeration = 12.0f;
    int   exaggeration_stop  = 250;     // iteration dropping exaggeration
    float theta              = 0.5f;    // BH accuracy/speed trade-off
    float min_gain           = 0.01f;
};

// P       : symmetric affinities from buildSymmetricP (local ids, globally
//           normalized by 1/(2*n_total))
// y_init  : [n_local * 2] interleaved low-dim init (KnnGraph::y), consumed
// returns : final embedding [n_local * 2], interleaved, same row order as P
thrust::device_vector<float> optimizeTsne(
    NCCLCommunicator&            comm,
    const CsrMatrix&             P,
    thrust::device_vector<float> y_init,
    size_t                       n_local,
    int64_t                      n_total,
    const TsneOptParams&         params = {},
    cudaStream_t                 stream = 0);
