#pragma once

#include <vector>
#include <thrust/device_vector.h>


class NCCLCommunicator;

struct KMeansResult {
    std::vector<float> centroids;              // [n_clusters * dim]
    thrust::device_vector<float> local_data;   // redistributed points for this rank
    size_t local_n;                            // number of points after redistribution
    int n_clusters;
};

KMeansResult kmeansPartition(
    NCCLCommunicator& comm,
    float *local_x,         // host ptr, this rank's initial shard
    size_t local_n,         // number of vectors in initial shard
    int dim,
    int n_clusters,
    int niter,
    int samples_per_rank    // number of samples each rank contributes to training
);
