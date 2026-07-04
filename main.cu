#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "ICommunicator.hpp"
#include "Dataset.hpp"
#include "PCAInit.hpp"
#include "KMeansPartition.hpp"
#include "KnnGraph.hpp"
#include "SymmetrizeP.hpp"
#include "TsneOptimizer.hpp"
#include "utils.hpp"


int main(int argc, char **argv)
{
    NCCLCommunicator *communicator = new NCCLCommunicator(argc, argv);
    communicator->initNCCL();
    int rank = communicator->getRank();
    int size = communicator->getSize();
    const Config config = parseArgs(argc, argv);
    auto dataset = new Dataset(config.data_path);
    int dim = config.dim, n_clusters = config.k, niter = config.niter;
    size_t total_vectors = dataset->get_total_vectors(dim);
    size_t vectors_per_rank = (total_vectors + size - 1) / size;
    size_t start_idx = rank * vectors_per_rank;
    size_t local_n = std::min(vectors_per_rank, total_vectors - start_idx);
    std::cout << "Rank " << rank << " - Total vectors: " << total_vectors
              << ", Local vectors: " << local_n << std::endl;
    float* local_x = dataset->get_shard_ptr(start_idx, dim);

    thrust::device_vector<float> local_cluster_data(local_x, local_x + local_n * dim);

    const int n_components = std::min(50, dim);
    PCAResult pca = distributedPCA(*communicator, local_cluster_data, local_n, dim, n_components);
    local_cluster_data.clear(); local_cluster_data.shrink_to_fit();
    const int work_dim = pca.r;
    if (rank == 0) {
        std::cout << "PCA: dim " << dim << " -> " << work_dim
                  << " components, init_scale=" << pca.init_scale << std::endl;
    }

    const int K = 0.1 * local_n + 1;
    KMeansResult km = kmeansPartition(*communicator, std::move(pca.scores), local_n, work_dim, n_clusters, niter, K, (int64_t)start_idx);

    thrust::device_vector<float> d_y_init =
        extractInit2D(km.local_data, km.local_n, work_dim, pca.init_scale);

    {
        std::vector<float> h_debug(km.local_data.size());
        thrust::copy(km.local_data.begin(), km.local_data.end(), h_debug.begin());
        std::ofstream ofs("../assignments_rank" + std::to_string(rank) + ".txt");
        for (size_t i = 0; i < km.local_n; i++) {
            ofs << h_debug[i * work_dim] << " " << h_debug[i * work_dim + 1]
                << " " << rank << "\n";
        }
    }

    {
        std::vector<float> h_init(d_y_init.size());
        thrust::copy(d_y_init.begin(), d_y_init.end(), h_init.begin());
        std::ofstream ofs("../y_init_rank" + std::to_string(rank) + ".txt");
        for (size_t i = 0; i < km.local_n; i++) {
            ofs << h_init[i * 2] << " " << h_init[i * 2 + 1] << " " << rank << "\n";
        }
    }

    if (rank == 0) {
        std::ofstream ofs("../centroids.txt");
        for (int c = 0; c < n_clusters; c++) {
            ofs << km.centroids[c * work_dim] << " " << km.centroids[c * work_dim + 1] << "\n";
        }
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float perplexity = config.perplexity;
    int n_neighbors = perplexity*3;

    KnnGraph knn = computeKnnGraph(
        *communicator,
        std::move(km.local_data),
        std::move(d_y_init),
        km.local_n, work_dim, n_neighbors,
        stream);

    int n_rows_to_print = std::min<int>(knn.n_local, 5);
    int cols_to_show = std::min(knn.k, 5);
    std::vector<int>   h_idx((size_t)n_rows_to_print * knn.k);
    std::vector<float> h_dst((size_t)n_rows_to_print * knn.k);
    cudaMemcpy(h_idx.data(), thrust::raw_pointer_cast(knn.indices.data()),
               h_idx.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dst.data(), thrust::raw_pointer_cast(knn.distances.data()),
               h_dst.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Rank " << rank << " k-NN sample (first "
              << n_rows_to_print << " rows, " << cols_to_show << "/" << knn.k
              << " neighbors, (id, dist^2)) ===" << std::endl;
    for (int r = 0; r < n_rows_to_print; r++) {
        int64_t gid = knn.global_offset + r;
        std::cout << "  local " << r << " (global " << gid << "): ";
        for (int j = 0; j < cols_to_show; j++) {
            int c = r * knn.k + j;
            std::cout << "(" << h_idx[c] << ", " << h_dst[c] << ") ";
        }
        if (knn.k > cols_to_show) std::cout << "...";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    CsrMatrix P = buildSymmetricP(knn, perplexity, stream);
    std::cout << "Rank " << rank << ": symmetric P built - " << P.n
              << " x " << P.n << ", nnz=" << P.nnz << std::endl;

    TsneOptParams opt_params;
    thrust::device_vector<float> d_y = optimizeTsne(
        *communicator, P, std::move(knn.y), knn.n_local, knn.n_total,
        opt_params, stream);

    {
        // columns: x y rank original_row_id  (id joins rows back to the
        // input .dat order, e.g. datasets/mnist_labels.txt)
        std::vector<float> h_y(d_y.size());
        thrust::copy(d_y.begin(), d_y.end(), h_y.begin());
        std::vector<int64_t> h_ids(km.ids.size());
        thrust::copy(km.ids.begin(), km.ids.end(), h_ids.begin());
        std::ofstream ofs("../y_final_rank" + std::to_string(rank) + ".txt");
        for (size_t i = 0; i < knn.n_local; i++) {
            ofs << h_y[i * 2] << " " << h_y[i * 2 + 1] << " " << rank
                << " " << h_ids[i] << "\n";
        }
    }

    cudaStreamDestroy(stream);
    delete communicator;
    delete dataset;
    return EXIT_SUCCESS;
}
