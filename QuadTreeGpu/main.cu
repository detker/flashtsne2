#define NCCL_DEBUG INFO

#include "quad_tree_builder.cuh"
#include "node.cuh"
#include "quad_tree_traversor.cuh"
#include "NcclRing.cuh"

#include <random>
#include <memory>
#include <chrono>

std::unique_ptr<float[]> generate_random_floats(size_t N, float min, float max)
{
    std::unique_ptr<float[]> data(new float[N]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (size_t i = 0; i < N; ++i)
    {
        data[i] = dist(gen);
    }

    return data;
}

constexpr size_t N = 1000000;

static std::tuple<
    thrust::device_vector<float>, // X back
    thrust::device_vector<float>, // Y back
    thrust::device_vector<float>, // X grad
    thrust::device_vector<float>, // Y grad 
    float // Z
> circulate(
    NcclRing& ring,
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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    NcclRing ring(MPI_COMM_WORLD);

    thrust::device_vector<float> x(1000);
    thrust::device_vector<float> y(1000);
    thrust::device_vector<uint32_t> nlen{}, f_pos{}, length{};
    thrust::device_vector<uint8_t> is_leaf{};
    thrust::device_vector<float> x_com{}, y_com{}, sink{}, x_grad{}, y_grad{};
    float z, face_len;

    ParallelQuadtreeBuilder tree_builder;
    tree_builder.bind_arguments(std::move(x), std::move(y));
    std::tie(
        nlen, f_pos, length, is_leaf, x_com, y_com
    ) = tree_builder.build_tree();
    face_len = tree_builder.get_face_len();
    std::tie(x, y, sink) = tree_builder.retrive_arguments();

    QuadTreeTraversor<TsneApproxCond, TsneNodeHanlder, TsneLeafHandler> traversor;
    traversor.set_face_lenght(face_len);

    std::tie(
        x, y, x_grad, y_grad, z
    ) = circulate(
        ring,
        traversor,
        std::move(x),
        std::move(y)
    );

    /* Apply gradient calculate Z factor */

    MPI_Finalize();
}