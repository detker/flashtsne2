#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>

#include "ICommunicator.hpp"
#include "Dataset.hpp"
#include "KMeansPartition.hpp"
#include "SparseP.hpp"
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

    // cudaHostRegister(local_x, local_n * dim * sizeof(float), cudaHostRegisterDefault);

    // 1. KMEANS PARTITION + REDISTRIBUTE
    
    const int K = 100000;
    KMeansResult km = kmeansPartition(*communicator, local_x, local_n, dim, n_clusters, niter, K);

    std::vector<float> h_cluster_data(km.local_data.size());
    thrust::copy(km.local_data.begin(), km.local_data.end(), h_cluster_data.begin());
    km.local_data.clear(); km.local_data.shrink_to_fit();

    size_t cluster_n = km.local_n;

    // save for debug
    std::ofstream ofs("../assignments_rank" + std::to_string(rank) + ".txt");
    for (size_t i = 0; i < km.local_n; i+=2) {
        ofs << h_cluster_data[i] << " " << h_cluster_data[i+1]
            << " " << rank << "\n";
    }

    if (rank == 0) {
        std::ofstream ofs("../centroids.txt");
        for (int c = 0; c < n_clusters; c++) {
            ofs << km.centroids[c * dim] << " " << km.centroids[c * dim + 1] << "\n";
        }
    } 

    // 2. BUILD SPARSE P_IJ ON LOCAL CLUSTER
    cudaStream_t sparse_stream;
    cudaStreamCreate(&sparse_stream);

    float perplexity = 30.0f;
    int n_neighbors = 90;

    cudaHostRegister(h_cluster_data.data(), h_cluster_data.size() * sizeof(float), cudaHostRegisterDefault);

    SparseMatrix P = buildSparseP(
        *communicator,
        h_cluster_data.data(), cluster_n,
        dim, n_neighbors, perplexity,
        km.centroids.data(), km.n_clusters,
        sparse_stream
    );

    int n_rows_to_print = std::min((int)P.n_rows, 5);
    std::vector<int> h_row_off(n_rows_to_print + 1);
    cudaMemcpy(h_row_off.data(), thrust::raw_pointer_cast(P.row_offsets.data()),
               (n_rows_to_print + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    int max_nnz = h_row_off[n_rows_to_print];
    std::vector<int> h_cols(max_nnz);
    std::vector<float> h_vals(max_nnz);
    cudaMemcpy(h_cols.data(), thrust::raw_pointer_cast(P.col_indices.data()),
               max_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals.data(), thrust::raw_pointer_cast(P.values.data()),
               max_nnz * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Rank " << rank << " P matrix sample (first "
              << n_rows_to_print << " rows) ===" << std::endl;
    for (int r = 0; r < n_rows_to_print; r++) {
        int global_row = P.global_row_offset + r;
        int start = h_row_off[r], end = h_row_off[r+1];
        std::cout << "  row " << global_row << " (" << (end - start) << " nnz): ";
        int to_show = std::min(end - start, 5);
        for (int j = 0; j < to_show; j++) {
            std::cout << "(" << h_cols[start+j] << ", " << h_vals[start+j] << ") ";
        }
        if (end - start > 5) std::cout << "...";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    cusparseHandle_t cusparse_handle;
    cusparseCreate(&cusparse_handle);
    cusparseSetStream(cusparse_handle, sparse_stream);

    cusparseDestroy(cusparse_handle);
    destroySparseP(P);
    cudaHostUnregister(h_cluster_data.data());
    cudaStreamDestroy(sparse_stream);
    cudaHostUnregister(local_x);

    delete communicator;
    delete dataset;
    return EXIT_SUCCESS;
}
