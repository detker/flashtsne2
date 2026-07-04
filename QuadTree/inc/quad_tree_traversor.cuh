#ifndef QUAD_TREE_TRAVERSOR
#define QUAD_TREE_TRAVERSOR

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <tuple>
#include <cstdio>

// #include "quad_tree_builder.cuh"
#include "quad_tree_config.cuh"

/*
 TODO: Add compile time selection for level profiling.
 Add table to traverse function to count how many times
 we terminate early on subtree instead of traversing deeper
*/

constexpr float theta = 0.5f;

struct TsneApproxCond{
    inline bool __device__ __host__ operator()(float x_com, float y_com, float x, float y,
        float face_len, uint8_t level, float theta){
        float x_dist = x_com - x, y_dist = y_com - y;
        float cell = face_len / (1 << level);
        float dist_sq = x_dist * x_dist + y_dist * y_dist;
        return cell * cell < theta * theta * dist_sq;
    }
};

/*
 Use float3 to pack partial Z calculation
*/
struct TsneNodeHanlder{
    inline float3 __device__ __host__ operator()(float x_com, float y_com, float x, float y, uint32_t n){
        float x_dist = x - x_com;
        float y_dist = y - y_com;
        float denom = 1.0f / (1.0f + x_dist * x_dist + y_dist * y_dist);
        float denom_sq = (float)n * denom * denom;
        return { x_dist * denom_sq, y_dist * denom_sq, (float)n * denom };
    }
};

struct TsneLeafHandler{
    inline float3 __device__ __host__ operator()(float leaf_x, float leaf_y, float x, float y){
        float x_dist = x - leaf_x;
        float y_dist = y - leaf_y;
        float denom = 1.0f / (1.0f + (x_dist * x_dist + y_dist * y_dist));
        float denom_sq = denom * denom;
        return { x_dist * denom_sq, y_dist * denom_sq, denom };
    }
};

static inline __device__ float3 add_vec3(float3 &a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

/*
 This traversor implementation will compute forces as well as
 normalization factor Z. Each thread will compute the part of
 normalization factor for given point and store partial value.
 After that we will reduce into single value. This is used instead
 of atomics to single value due to lock contention.
*/
template <typename ApproxCond, typename NodeHanlder, typename LeafHandler, bool profile = false>
static __global__ void traverse_impl(
    const uint32_t *nlen_d,
    const uint32_t *f_pos_d,
    const uint32_t *length_d,
    const uint8_t *is_leaf_d,
    const float *x_com_d,
    const float *y_com_d,
    const float *x_d,
    const float *y_d,
    const float face_length_d,
    const float theta,
    const size_t size,
    float *res_x_d,
    float *res_y_d,
    float *res_z_d,
    uint32_t *profile_table = nullptr
){
    ApproxCond approx_cond;
    NodeHanlder node_handler;
    LeafHandler leaf_handler;
                
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < size){
        float x = x_d[tid];
        float y = y_d[tid];

        /*
         TODO: Fetch top levels to shared memory
         !!! Lambdas do not support shared memory !!!
        */
        /*
         In initial 100k benchmark this kernel has almost 
         perfect stats. ~99.7% L1 hit rate, ~99.9% L2 hit 
         rate and 100% branch eff. due to no approx cond.
         With approx cond on. and spatiall sort on random
         data, traversal keep ~80% branch eff. Highest
         levels are abused. Most of data fits in L2 cache
         and we got nice stats there, however keeping
         top levels might improce L1 hit rate.
        */

        float3 res { 0.0f, 0.0f, 0.0f };
        /* Has to be at least 4 * tree_max_height */
        uint32_t stack[4 * QUAD_TREE_MAX_HEIGHT];
        uint8_t depth_stack[4 * QUAD_TREE_MAX_HEIGHT];
        uint32_t top = 0;

        top++;        
        stack[top] = 0;
        depth_stack[top] = 0;

        while (top)
        {
            uint32_t node_idx = stack[top];
            uint8_t depth = depth_stack[top];
            top--;

            if (is_leaf_d[node_idx])
            {
                /* TODO: unroll this loop based on leaf config */
                for(uint32_t i = 0; i < length_d[node_idx]; i++)
                {
                    uint32_t leaf_idx = f_pos_d[node_idx] + i;
                    /* Skip self interaction */
                    if(tid == leaf_idx) continue;
                    add_vec3(res, leaf_handler(
                        x_d[leaf_idx],
                        y_d[leaf_idx],
                        x, y
                    ));
                }

                if constexpr (profile) atomicAdd(&profile_table[depth], 1);
                continue;
            }

            if(approx_cond(
                x_com_d[node_idx],
                y_com_d[node_idx],
                x, y, 
                face_length_d,
                depth,
                theta
            )
            ){
                add_vec3(res, node_handler(
                    x_com_d[node_idx],
                    y_com_d[node_idx],
                    x, y, nlen_d[node_idx]
                ));

                if constexpr (profile) atomicAdd(&profile_table[depth], 1);
                continue;
            }

            #pragma unroll 4
            for (uint32_t i = 0; i < 4; ++i)
            {
                if(i < length_d[node_idx]){
                    top++;
                    stack[top] = f_pos_d[node_idx] + i;
                    depth_stack[top] = depth + 1;
                }
            }
        }

        res_x_d[tid] += res.x;
        res_y_d[tid] += res.y;
        res_z_d[tid] += res.z;
    }
}

template <typename ApproxCond, typename NodeHanlder, typename LeafHandler>
class QuadTreeTraversor{
public:
    QuadTreeTraversor() = default;

    inline void set_face_lenght(float face_length){
        this->face_length = face_length;
    }

    inline void load_points(
        thrust::device_vector<float> x,
        thrust::device_vector<float> y
    ){
        this->x = std::move(x);
        this->y = std::move(y);
    }

    inline std::tuple<
        thrust::device_vector<float>,
        thrust::device_vector<float>
    > get_points(void){
        return {
            std::move(x),
            std::move(y)
        };
    }

    inline void load_tree(
        thrust::device_vector<uint32_t> nlen,
        thrust::device_vector<uint32_t> f_pos,
        thrust::device_vector<uint32_t> length,
        thrust::device_vector<uint8_t> is_leaf,
        thrust::device_vector<float> x_com,
        thrust::device_vector<float> y_com
    ){
        this->nlen = std::move(nlen);
        this->f_pos = std::move(f_pos);
        this->length = std::move(length);
        this->is_leaf = std::move(is_leaf);
        this->x_com = std::move(x_com);
        this->y_com = std::move(y_com);
    }

    inline std::tuple<
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>,
        thrust::device_vector<float>,
        thrust::device_vector<float>
    > get_tree(void){
        return {
            std::move(nlen),
            std::move(f_pos),
            std::move(length),
            std::move(is_leaf),
            std::move(x_com),
            std::move(y_com)
        };
    }

    /*
     This function will traverse the tree for loaded points and store
     result for each point.
     Providing point data in spatially sorted order lowers warp
     divergence. Tree builder will already perform such sort.
    */
    inline std::tuple<
        thrust::device_vector<float>,
        thrust::device_vector<float>,
        float
    > traverse(
        thrust::device_vector<float> res_x,
        thrust::device_vector<float> res_y
    ){
        size_t size = x.size();

        thrust::device_vector<float> res_z(size, 0.0f);

        float
            *res_x_d = res_x.data().get(),
            *res_y_d = res_y.data().get(),
            *res_z_d = res_z.data().get();
        const uint32_t 
            *nlen_d = nlen.data().get(),
            *f_pos_d = f_pos.data().get(),
            *length_d = length.data().get();
        const uint8_t 
            *is_leaf_d = is_leaf.data().get();
        const float 
            *x_com_d = x_com.data().get(),
            *y_com_d = y_com.data().get(),
            *x_d = x.data().get(),
            *y_d = y.data().get();
        const float face_length_d = face_length;

        
        uint32_t *profile_table_d;
#ifdef QUAD_TREE_PROFILE
        thrust::device_vector<uint32_t> profile_table(QUAD_TREE_MAX_HEIGHT, 0);
        profile_table_d = profile_table.data().get();
#else
        profile_table_d = nullptr;
#endif

        dim3 threads(32u);
        dim3 blocks((size + threads.x - 1u) / threads.x);
        traverse_impl<ApproxCond, NodeHanlder, LeafHandler, quad_tree_profile_levels><<<blocks, threads>>>(
            nlen_d,
            f_pos_d,
            length_d,
            is_leaf_d,
            x_com_d, 
            y_com_d,
            x_d,
            y_d, 
            face_length_d,
            theta,
            size,
            res_x_d,
            res_y_d,
            res_z_d,
            profile_table_d
        );

        cudaError err = cudaDeviceSynchronize();
        if((cudaError_t)CUDA_SUCCESS != err){
            /* Failed, return empty vector */
            return {};
        }

#ifdef QUAD_TREE_PROFILE
        /* Consume profiling metrics here */
#endif
        /*
         Reduction might introduce numerical instabilities,
         in such case we could transform-reduce with double
         and cast back to float.
        */
        float Z = thrust::reduce(res_z.begin(), res_z.end());

        return std::tuple<
            thrust::device_vector<float>,
            thrust::device_vector<float>,
            float
        > {
            std::move(res_x),
            std::move(res_y),
            Z
        };
    }

private:
    float face_length;

    /* Do not modify points here */
    thrust::device_vector<float> x;
    thrust::device_vector<float> y;

    thrust::device_vector<uint32_t> nlen;
    thrust::device_vector<uint32_t> f_pos;
    thrust::device_vector<uint32_t> length;
    thrust::device_vector<uint8_t> is_leaf;
    thrust::device_vector<float> x_com;
    thrust::device_vector<float> y_com;
};

#endif