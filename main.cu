#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <utility>
#include <tuple>

#include "ICommunicator.hpp"
#include "Dataset.hpp"
#include "PCAInit.hpp"
#include "KMeansPartition.hpp"
#include "KnnGraph.hpp"
#include "SymmetrizeP.hpp"
#include "utils.hpp"

/* Tree computation libs */
#include "NcclRing.cuh"
#include "quad_tree_builder.cuh"
#include "quad_tree_traversor.cuh"

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

    for(size_t i{0}; i < ring.size(); i++){
        traversor.load_points(std::move(x), std::move(y));

        std::tie(grad_x, grad_y, tmp_z) = traversor.traverse(
            std::move(grad_x),
            std::move(grad_y)
        );
        std::tie(x, y) = traversor.get_points();

        z += tmp_z;

        x = ring.ring_exchange(std::move(x));
        y = ring.ring_exchange(std::move(y));
        grad_x = ring.ring_exchange(std::move(grad_x));
        grad_y = ring.ring_exchange(std::move(grad_y));
    }
    /* To get back original tensor we need one more hop */
    x = ring.ring_exchange(std::move(x));
    y = ring.ring_exchange(std::move(y));
    grad_x = ring.ring_exchange(std::move(grad_x));
    grad_y = ring.ring_exchange(std::move(grad_y));

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
    num_points - number of points for this rank
    Outputs:
    x_grad - x components of gradient for each point [num_points, 1]
    y_grad - y components of gradient for each point [num_points, 1]
    x, y - returns low dimensional points back
    z - normalization factor
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
    thrust::device_vector<float> y
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
    
    traversor.set_face_lenght(face_len);

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
    KMeansResult km = kmeansPartition(*communicator, std::move(pca.scores), local_n, work_dim, n_clusters, niter, K);

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

    /* Setup for ring computation */
    NcclRing ring(MPI_COMM_WORLD);
    size_t num_pts = knn.n_local;

    thrust::device_ptr<float> begin = knn.y.data();
    thrust::device_ptr<float> mid = begin + num_pts;
    thrust::device_ptr<float> end = mid + num_pts;

    thrust::device_vector<float> x(begin, mid), y(mid, end), x_grad{}, y_grad{};
    float Z;
    
    std::tie(x, y, x_grad, y_grad, Z) = step_ring(ring, std::move(x), std::move(y));

    cudaStreamDestroy(stream);
    delete communicator;
    delete dataset;
    return EXIT_SUCCESS;
}
