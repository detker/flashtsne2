#include "KMeansPartition.hpp"
#include "ICommunicator.hpp"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <iostream>


KMeansResult kmeansPartition(
    NCCLCommunicator& comm,
    thrust::device_vector<float> local_cluster_data,
    size_t local_n,
    int dim,
    int n_clusters,
    int niter,
    int samples_per_rank)
{
    int rank = comm.getRank();
    int size = comm.getSize();
    int K = samples_per_rank;

    //thrust::device_vector<float> local_cluster_data(local_x, local_x + local_n * dim);

    std::mt19937 rng(1337 + rank);
    std::uniform_int_distribution<size_t> dist(0, local_n - 1);

    std::vector<int> h_sample_indices(K);
    for (int i = 0; i < K; ++i) {
        h_sample_indices[i] = dist(rng);
    }

    thrust::device_vector<int> d_sample_indices(h_sample_indices.begin(), h_sample_indices.end());
    thrust::device_vector<float> d_sample_ptr(K * dim);

    const int* d_idx = thrust::raw_pointer_cast(d_sample_indices.data());
    auto row_map = thrust::make_transform_iterator(
        thrust::counting_iterator<size_t>(0),
        [d_idx, dim] __device__ (size_t o) -> size_t {
            return (size_t)d_idx[o / dim] * dim + (o % dim);
        });
    thrust::gather(thrust::device,
                   row_map, row_map + (size_t)K * dim,
                   thrust::raw_pointer_cast(local_cluster_data.data()),
                   thrust::raw_pointer_cast(d_sample_ptr.data()));

    float* h_sample_ptr = new float[K * dim];
    cudaMemcpy(h_sample_ptr, thrust::raw_pointer_cast(d_sample_ptr.data()),
               K * dim * sizeof(float), cudaMemcpyDeviceToHost);

    float* gathered_samples = rank == 0 ? new float[K * dim * size] : nullptr;
    comm.gather<CommunicationBackend::MPI>(
        h_sample_ptr, K * dim, CommDataType::FLOAT,
        gathered_samples, K * dim, CommDataType::FLOAT, 0);

    d_sample_ptr.clear(); d_sample_ptr.shrink_to_fit();
    d_sample_indices.clear(); d_sample_indices.shrink_to_fit();
    delete[] h_sample_ptr;

    // Train KMeans on rank 0
    std::vector<float> centroids(n_clusters * dim);
    if (rank == 0) {
        faiss::ClusteringParameters cp;
        cp.niter = niter;
        cp.nredo = 5;
        cp.min_points_per_centroid = K / n_clusters;
        cp.verbose = true;
        faiss::Clustering kmeans(dim, n_clusters, cp);

        faiss::IndexFlatL2 index_cpu(dim);

        int n_gpus = 0;
        cudaGetDeviceCount(&n_gpus);
        std::vector<faiss::gpu::GpuResourcesProvider*> resources(n_gpus);
        std::vector<int> devs(n_gpus);
        for (int i = 0; i < n_gpus; i++) {
            resources[i] = new faiss::gpu::StandardGpuResources();
            devs[i] = i;
        }
        faiss::Index* gpu_index = faiss::gpu::index_cpu_to_gpu_multiple(resources, devs, &index_cpu);

        kmeans.train(K * size, gathered_samples, *gpu_index);
        std::copy(kmeans.centroids.begin(), kmeans.centroids.end(), centroids.begin());

        delete gpu_index;
        for (auto res : resources) delete res;
        delete[] gathered_samples;
    }

    comm.broadcast<CommunicationBackend::MPI>(centroids.data(), n_clusters * dim, CommDataType::FLOAT, 0);

    // assign each local point to nearest centroid
    faiss::gpu::StandardGpuResources assign_res;
    faiss::gpu::GpuIndexFlat assign_index(&assign_res, dim, faiss::MetricType::METRIC_L2);
    assign_index.add(n_clusters, centroids.data());

    thrust::device_vector<float> d_distances(local_n);
    thrust::device_vector<faiss::idx_t> d_labels(local_n);

    assign_index.search(local_n, thrust::raw_pointer_cast(local_cluster_data.data()), 1,
                        thrust::raw_pointer_cast(d_distances.data()),
                        thrust::raw_pointer_cast(d_labels.data()));
    cudaDeviceSynchronize();


    // redistribute points so each rank owns spatially nearby points
    auto distributed_points = comm.distributeData(
        thrust::raw_pointer_cast(d_labels.data()), dim, thrust::raw_pointer_cast(local_cluster_data.data()), local_n);

    d_distances.clear(); d_distances.shrink_to_fit();
    d_labels.clear(); d_labels.shrink_to_fit();
    local_cluster_data.clear(); local_cluster_data.shrink_to_fit();

    size_t cluster_n = distributed_points.size() / dim;
    std::cout << "Rank " << rank << ": received " << cluster_n
              << " points after redistribution" << std::endl;

    KMeansResult result;
    result.centroids = std::move(centroids);
    result.local_data = std::move(distributed_points);
    result.local_n = cluster_n;
    result.n_clusters = n_clusters;
    return result;
}
