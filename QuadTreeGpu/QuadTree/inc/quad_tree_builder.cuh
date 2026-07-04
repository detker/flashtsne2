#ifndef QUAD_TREE_BUILDER
#define QUAD_TREE_BUILDER

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/replace.h>

#include <cstdlib>
#include <iostream>
#include <tuple>

#include "allocators.cuh"
#include "quad_tree_config.cuh"

class ParallelQuadtreeBuilder
{
public:
    ParallelQuadtreeBuilder() {};
    ParallelQuadtreeBuilder(thrust::device_vector<float>&& x,
                     thrust::device_vector<float>&& y,
                     thrust::device_vector<float>&& m)
        : x(x), y(y), m(m)
        { 
        };

    void bind_arguments(
        thrust::device_vector<float> x,
        thrust::device_vector<float> y
    ){
        this->x = std::move(x);
        this->y = std::move(y);
    }
    
    std::tuple<
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>,
        thrust::device_vector<float>,
        thrust::device_vector<float>> build_tree();

    std::tuple<
        thrust::device_vector<float>,
        thrust::device_vector<float>,
        thrust::device_vector<float>> retrive_arguments();

    inline float get_face_len(){
        return fmax(x_max - x_min, y_max - y_min);
    }

    /* Has to stay public for lambda accessibility for thrust */
    void compute_codes();

    // Helpers
    /* TODO: remove refrence and do explicit move / lowkey fine since const ref */
    /* Produce qudrants for level and COM summary */
    std::tuple<
        thrust::device_vector<uint64_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>,
        thrust::device_vector<float>,
        thrust::device_vector<float>> 
    generate_quadrants_for_level(
        const thrust::device_vector<uint64_t>& prev_code,
        const thrust::device_vector<uint32_t>& prev_nlen,
        const thrust::device_vector<uint32_t>& prev_start,
        const thrust::device_vector<float>& x_com,
        const thrust::device_vector<float>& y_com,
        int level
    );

    /* Remove redundant nodes  */
    std::tuple<
        thrust::device_vector<uint64_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>,
        thrust::device_vector<float>,
        thrust::device_vector<float>>
    trim_redundant_nodes(
        thrust::device_vector<uint64_t> p_key, 
        thrust::device_vector<uint32_t> nlen,
        thrust::device_vector<uint32_t> start,
        thrust::device_vector<uint8_t> clen,
        thrust::device_vector<float> x_com,
        thrust::device_vector<float> y_com
    );

    std::tuple<
        thrust::device_vector<uint32_t>,
        thrust::device_vector<float>,
        thrust::device_vector<float>>
    normalize_center_of_mass(
        thrust::device_vector<uint32_t> nlen,
        thrust::device_vector<float> x_com,
        thrust::device_vector<float> y_com
    );

    void normalize_source_data(void);
    
    std::tuple<thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>>
    fill_tree(
        thrust::device_vector<uint32_t> nlen,
        thrust::device_vector<uint32_t> start,
        thrust::device_vector<uint8_t> clen);

    static constexpr size_t get_max_height() { return H_max; };

private:
    /* Maximum of points in a single leaf */
    static constexpr size_t T = QUAD_TREE_LEAF;
    /* Maximum height of quadtree */
    static constexpr size_t H_max = QUAD_TREE_MAX_HEIGHT;
    float x_max, x_min, y_max, y_min;

    /* Input data*/
    thrust::device_vector<float> x;
    thrust::device_vector<float> y;
    thrust::device_vector<float> m;

    /*  */
    thrust::device_vector<uint64_t> code;
    /* Tree */
    thrust::device_vector<uint64_t> key;
    thrust::device_vector<uint32_t> f_pos, length;
    thrust::device_vector<uint8_t> is_leaf;
};

#endif