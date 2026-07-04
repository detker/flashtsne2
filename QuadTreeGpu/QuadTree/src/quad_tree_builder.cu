#include "quad_tree_builder.cuh"

#include <list>
#include <string>
#include <sstream>

#include <thrust/extrema.h>
#include <thrust/pair.h>

#include "node.cuh"

static void __attribute__((unused)) show_tree(thrust::device_vector<uint64_t> key, thrust::device_vector<bool> is_leaf,
    thrust::device_vector<uint32_t> f_pos, thrust::device_vector<uint32_t> length,
    thrust::device_vector<float> x, thrust::device_vector<float> y,
    thrust::device_vector<float> x_com, thrust::device_vector<float> y_com)
{
    if(key.size() > 200){
        printf("Hell nah aint printing that\n");
        return;
    }

    thrust::host_vector<uint64_t> h_key(key);
    thrust::host_vector<bool> h_is_leaf(is_leaf);
    thrust::host_vector<uint32_t> h_f_pos(f_pos), h_lenght(length);
    thrust::host_vector<float> h_x(x), h_y(y);
    thrust::host_vector<float> h_x_com(x_com), h_y_com(y_com);

    std::function<Node*(size_t)> traverse = [&](size_t index) -> Node*{
        size_t children_offset = h_f_pos[index];
        size_t child_count = h_lenght[index];

        Node *node;

        if(!h_is_leaf[index]){
            std::stringstream ss;
            ss << "Node" << " : " << h_x_com[index] << " | " << h_y_com[index];
            node = new Node(ss.str());
            for(size_t i = 0; i < child_count; i++){
                node->children.emplace_back(std::unique_ptr<Node>(traverse(children_offset + i)));
            }
        }
        else{
            std::stringstream ss;
            ss << "Leaf " << children_offset << "(" << child_count <<") " << " : " << h_x[children_offset] << " | " << h_y[children_offset];
            node = new Node(ss.str());
        }

        return node;
    };

    Node *root = traverse(0);
    root->dump();
}

template <typename T>
static thrust::device_vector<T> compress_vector(std::list<thrust::device_vector<T>> &vector_list)
{
    size_t total_len = 0;
    for (thrust::device_vector<T> &vector : vector_list)
        total_len += vector.size();

    thrust::device_vector<T> compressed(total_len);

    size_t offset = 0;
    while (!vector_list.empty())
    {
        thrust::device_vector<T> vector = vector_list.front();
        vector_list.pop_front();

        cudaMemcpy(
            compressed.data().get() + offset,
            vector.data().get(),
            sizeof(T) * vector.size(),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        offset += vector.size();
    }

    return std::move(compressed);
}

__device__ __host__ uint64_t expand_bits(uint32_t &v)
{
    uint64_t x = v & 0x00000000FFFFFFFF;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    return x;
}

std::tuple<
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint8_t>,
    thrust::device_vector<float>,
    thrust::device_vector<float>>
ParallelQuadtreeBuilder::build_tree()
{
    compute_codes();

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), m.begin()));
    thrust::stable_sort_by_key(code.begin(), code.end(), zip_begin);

    thrust::device_vector<uint64_t> p_key;
    thrust::device_vector<uint32_t> nlen;
    thrust::device_vector<uint32_t> start;
    thrust::device_vector<uint8_t> clen;
    thrust::device_vector<float> x_com;
    thrust::device_vector<float> y_com;
    {
        std::list<thrust::device_vector<uint8_t>> level_children_list;
        std::list<thrust::device_vector<uint32_t>> level_start_list;
        std::list<thrust::device_vector<uint32_t>> level_points_list;
        std::list<thrust::device_vector<uint64_t>> level_code_list;
        std::list<thrust::device_vector<float>> level_x_com_list;
        std::list<thrust::device_vector<float>> level_y_com_list;

        for (int i = H_max; i >= 0; i--)
        {
            thrust::device_vector<uint8_t> level_children;
            thrust::device_vector<uint32_t> level_start;
            thrust::device_vector<uint32_t> level_points;
            thrust::device_vector<uint64_t> level_codes;
            thrust::device_vector<float> level_x_com;
            thrust::device_vector<float> level_y_com;

            const thrust::device_vector<uint64_t> *prev_codes = level_code_list.empty() ? nullptr : &level_code_list.front();
            const thrust::device_vector<uint32_t> *prev_nlen = level_points_list.empty() ? nullptr : &level_points_list.front();
            const thrust::device_vector<uint32_t> *prev_start = level_start_list.empty() ? nullptr : &level_start_list.front();
            const thrust::device_vector<float> *prev_x_com = level_x_com_list.empty() ? nullptr : &level_x_com_list.front();
            const thrust::device_vector<float> *prev_y_com = level_y_com_list.empty() ? nullptr : &level_y_com_list.front();

            std::tie(level_codes, level_points, level_start, level_children, level_x_com, level_y_com) =
                generate_quadrants_for_level(
                    prev_codes ? *prev_codes : code,
                    prev_nlen ? *prev_nlen : thrust::device_vector<uint32_t>{},
                    prev_start ? *prev_start : thrust::device_vector<uint32_t>{},
                    prev_x_com ? *prev_x_com : thrust::device_vector<float>{},
                    prev_y_com ? *prev_y_com : thrust::device_vector<float>{},
                    i
                );

            level_children_list.push_front(std::move(level_children));
            level_start_list.push_front(std::move(level_start));
            level_points_list.push_front(std::move(level_points));
            level_code_list.push_front(std::move(level_codes));
            level_x_com_list.push_front(std::move(level_x_com));
            level_y_com_list.push_front(std::move(level_y_com));
        }

        p_key = compress_vector<uint64_t>(level_code_list);
        nlen = compress_vector<uint32_t>(level_points_list);
        start = compress_vector<uint32_t>(level_start_list);
        clen = compress_vector<uint8_t>(level_children_list);
        x_com = compress_vector<float>(level_x_com_list);
        y_com = compress_vector<float>(level_y_com_list);
    }

    std::tie(p_key, nlen, start, clen, x_com, y_com) = trim_redundant_nodes(
        std::move(p_key), 
        std::move(nlen),
        std::move(start),
        std::move(clen),
        std::move(x_com),
        std::move(y_com)
    );

    std::tie(nlen, x_com, y_com) = normalize_center_of_mass(
        std::move(nlen),
        std::move(x_com),
        std::move(y_com)
    );

    normalize_source_data();

    std::tie(nlen, f_pos, length, is_leaf) = fill_tree(
        std::move(nlen),
        std::move(start),
        std::move(clen)
    );

    return {
        std::move(nlen),
        std::move(f_pos), 
        std::move(length),
        std::move(is_leaf),
        std::move(x_com),
        std::move(y_com)
    };
}

std::tuple<
    thrust::device_vector<float>,
    thrust::device_vector<float>,
    thrust::device_vector<float>>
ParallelQuadtreeBuilder::retrive_arguments(){
    return {
        std::move(x),
        std::move(y),
        std::move(m)
    };
}

/*
    Could be better:
        1. Find min/max for x and y
        2. In single transform step
           normalize and compute code
*/
void ParallelQuadtreeBuilder::compute_codes()
{
    auto x_min_max = thrust::minmax_element(x.begin(), x.end());
    auto y_min_max = thrust::minmax_element(y.begin(), y.end());

    x_min = *x_min_max.first; x_max = *x_min_max.second;
    y_min = *y_min_max.first; y_max = *y_min_max.second;

    float x_min_d = x_min, x_max_d = x_max, y_min_d = y_min, y_max_d = y_max; 

    auto trans_x = [x_min_d, x_max_d] __device__ __host__(const float &f)
    {
        return (f - x_min_d) / (x_max_d - x_min_d);
    };
    auto trans_y = [y_min_d, y_max_d] __device__ __host__(const float &f)
    {
        return (f - y_min_d) / (y_max_d - y_min_d);
    };
    /* Maybe single-pass zip iterator transform - vectors of same shape? */
    thrust::transform(x.begin(), x.end(), x.begin(), trans_x);
    thrust::transform(y.begin(), y.end(), y.begin(), trans_y);

    code.resize(x.size());

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end()));

    // clang-format off
    thrust::transform(zip_begin, zip_end, code.begin(), 
        [] __device__ __host__ (thrust::tuple<float, float> t) {
            float a = thrust::get<0>(t);
            float b = thrust::get<1>(t);

            uint32_t ix = (uint32_t)(fmin(fmax(a, 0.0f), 1.0f) * 4294967295.0f);
            uint32_t iy = (uint32_t)(fmin(fmax(b, 0.0f), 1.0f) * 4294967295.0f);

            return expand_bits(iy) | (expand_bits(ix) << 1);
        }
    );
    // clang-format on
}

std::tuple<
    thrust::device_vector<uint64_t>,
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint8_t>,
    thrust::device_vector<float>,
    thrust::device_vector<float>> 
ParallelQuadtreeBuilder::generate_quadrants_for_level(
    const thrust::device_vector<uint64_t>& prev_code,
    const thrust::device_vector<uint32_t>& prev_nlen,
    const thrust::device_vector<uint32_t>& prev_start,
    const thrust::device_vector<float>& prev_x_com,
    const thrust::device_vector<float>& prev_y_com,
    int level)
{
    thrust::device_vector<uint64_t> quadrant_codes(prev_code.size());
    if(level == H_max){
        thrust::transform(
            prev_code.begin(),
            prev_code.end(),
            quadrant_codes.begin(),
            [level] __device__ __host__ (uint64_t c) { 
                return c >> (64 - 2 * level);
            }
        );
    }
    else{
        thrust::transform(
            prev_code.begin(),
            prev_code.end(),
            quadrant_codes.begin(),
            [] __device__ __host__ (uint64_t c) { 
                return c >> 2;
            }
        );
    }

    thrust::device_vector<uint64_t> unique_quad_codes(quadrant_codes);

    auto new_end = thrust::unique(unique_quad_codes.begin(), unique_quad_codes.end());
    uint32_t quadrant_num = thrust::distance(unique_quad_codes.begin(), new_end);

    thrust::device_vector<uint8_t> clen(quadrant_num);
    thrust::device_vector<uint32_t> nlen(quadrant_num);
    thrust::device_vector<uint32_t> start(quadrant_num);
    thrust::device_vector<float> x_com(quadrant_num);
    thrust::device_vector<float> y_com(quadrant_num);

    auto out_it = thrust::make_zip_iterator(
        thrust::make_tuple(
            clen.begin(),
            nlen.begin(),
            start.begin(),
            x_com.begin(),
            y_com.begin()
        )
    );

    auto code_comparator = [] __device__ __host__ (uint64_t c1, uint64_t c2) {return c1 == c2; };
    auto reduce_op = [level] __device__ __host__ (
        thrust::tuple<uint8_t, uint32_t, uint32_t, float, float> t1,
        thrust::tuple<uint8_t, uint32_t, uint32_t, float, float> t2
    )
    {
        return thrust::make_tuple(
            (uint8_t)(thrust::get<0>(t1) + thrust::get<0>(t2)),
            (uint32_t)(thrust::get<1>(t1) + thrust::get<1>(t2)),
            min(thrust::get<2>(t1), thrust::get<2>(t2)),
            thrust::get<3>(t1) + thrust::get<3>(t2),
            thrust::get<4>(t1) + thrust::get<4>(t2) 
        );
    };

    if(level == H_max){
        thrust::reduce_by_key(
            quadrant_codes.begin(),
            quadrant_codes.end(),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::make_constant_iterator<uint8_t>(0),
                    thrust::make_constant_iterator<uint32_t>(1),
                    thrust::make_counting_iterator<uint32_t>(0),
                    x.begin(),
                    y.begin()
                )
            ),
            thrust::make_discard_iterator(),
            out_it,
            code_comparator,
            reduce_op
        );
    }
    else{
        thrust::reduce_by_key(
            quadrant_codes.begin(),
            quadrant_codes.end(),
            thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_constant_iterator<uint8_t>(1),
                    prev_nlen.begin(),
                    prev_start.begin(),
                    prev_x_com.begin(),
                    prev_y_com.begin()
                )
            ),
            thrust::make_discard_iterator(),
            out_it,
            code_comparator,
            reduce_op
        );
    }

    unique_quad_codes.resize(quadrant_num);
    unique_quad_codes.shrink_to_fit();

    return std::make_tuple<
        thrust::device_vector<uint64_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>,
        thrust::device_vector<float>,
        thrust::device_vector<float>>
    (
        std::move(unique_quad_codes),
        std::move(nlen),
        std::move(start),
        std::move(clen),
        std::move(x_com),
        std::move(y_com)
    );
}

std::tuple<thrust::device_vector<uint64_t>,
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint8_t>,
    thrust::device_vector<float>,
    thrust::device_vector<float>>
ParallelQuadtreeBuilder::trim_redundant_nodes(
    thrust::device_vector<uint64_t> p_key,
    thrust::device_vector<uint32_t> nlen,
    thrust::device_vector<uint32_t> start,
    thrust::device_vector<uint8_t> clen,
    thrust::device_vector<float> x_com,
    thrust::device_vector<float> y_com)
{
    thrust::device_vector<uint64_t> node_child_start(clen.size()); 
    /* Value initialization is important - by default exclusive_scan happens on uint8_t and it overflows */
    auto cast = [] __device__ __host__ (uint8_t v) { return static_cast<uint64_t>(v); };
    thrust::exclusive_scan(
        thrust::make_transform_iterator(clen.begin(), cast),
        thrust::make_transform_iterator(clen.end(), cast),
        node_child_start.begin(),
        uint64_t{1}
    );

    /* Important to 0 initialize, if we have garbage incl_scan + max will fail */
    thrust::device_vector<uint32_t> parent_id(clen.size(), 0);
    uint64_t *node_child_start_d = node_child_start.data().get();
    uint32_t *parent_id_d = parent_id.data().get();
    size_t len = parent_id.size();

    thrust::for_each(
        thrust::make_counting_iterator<uint64_t>(0),
        thrust::make_counting_iterator<uint64_t>(parent_id.size()),
        [node_child_start_d, parent_id_d, len] __device__ __host__ (uint64_t i){
            /* Uncoalesed - idk how to do diffrent*/
            uint64_t child_start = node_child_start_d[i];
            if(child_start < len)
                parent_id_d[child_start] = i; 
        }
    );

    thrust::inclusive_scan(parent_id.begin(), parent_id.end(),
        parent_id.begin(), thrust::maximum());

    thrust::device_vector<uint32_t> parent_point_count(clen.size());
    uint32_t *nlen_d = nlen.data().get();
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(clen.size()),
        parent_point_count.begin(),
        [parent_id_d, nlen_d] __device__ __host__ (uint32_t i){
            return nlen_d[parent_id_d[i]];
        }
    );

    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::make_counting_iterator<uint32_t>(0),
            p_key.begin(),
            nlen.begin(),
            start.begin(),
            clen.begin(),
            x_com.begin(),
            y_com.begin()
        )
    );

    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::make_counting_iterator<uint32_t>(p_key.size()),
            p_key.end(),
            nlen.end(),
            start.end(),
            clen.end(),
            x_com.end(),
            y_com.end()
        )
    );  

    uint32_t *parent_point_count_d = parent_point_count.data().get();
    uint32_t threshold = T;
    auto end = thrust::remove_if(
        zip_begin, zip_end,
        [parent_point_count_d, threshold] __device__ __host__ (
            thrust::tuple<uint32_t, uint64_t, uint32_t, uint32_t, uint8_t, float, float> t
        ){
            uint32_t i = thrust::get<0>(t);
            /* Remove if parent already is leaf, if this is root skip */
            return (parent_point_count_d[i] <= threshold) && i != 0;
        }
    );

    auto end_tuple = end.get_iterator_tuple();

    p_key.erase(thrust::get<1>(end_tuple), p_key.end());
    nlen.erase(thrust::get<2>(end_tuple), nlen.end());
    start.erase(thrust::get<3>(end_tuple), start.end());
    clen.erase(thrust::get<4>(end_tuple), clen.end());
    x_com.erase(thrust::get<5>(end_tuple), x_com.end());
    y_com.erase(thrust::get<6>(end_tuple), y_com.end());

    /* Idk this might copy - not sure if correct to do so */
    p_key.shrink_to_fit();
    nlen.shrink_to_fit();
    start.shrink_to_fit();
    clen.shrink_to_fit();
    x_com.shrink_to_fit();
    y_com.shrink_to_fit();

    return std::make_tuple<
        thrust::device_vector<uint64_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>,
        thrust::device_vector<float>,
        thrust::device_vector<float>>
    (
        std::move(p_key),
        std::move(nlen),
        std::move(start),
        std::move(clen),
        std::move(x_com),
        std::move(y_com)        
    );
}

std::tuple<
    thrust::device_vector<uint32_t>,
    thrust::device_vector<float>,
    thrust::device_vector<float>>
ParallelQuadtreeBuilder::normalize_center_of_mass(
    thrust::device_vector<uint32_t> nlen,
    thrust::device_vector<float> x_com,
    thrust::device_vector<float> y_com)
{
    uint32_t *nlen_d = nlen.data().get();
    float *x_com_d = x_com.data().get();
    float *y_com_d = y_com.data().get();
    float x_min_d = x_min, x_max_d = x_max, y_min_d = y_min, y_max_d = y_max; 
    thrust::for_each(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(nlen.size()),
        [nlen_d, x_com_d, y_com_d, x_min_d, x_max_d, y_min_d, y_max_d]
        __device__ __host__ (uint32_t i)
        {
            x_com_d[i] = ((x_com_d[i] / (float)nlen_d[i]) * (x_max_d - x_min_d) + x_min_d);
            y_com_d[i] = ((y_com_d[i] / (float)nlen_d[i]) * (y_max_d - y_min_d) + y_min_d);
        }
    );

    return std::tuple<
        thrust::device_vector<uint32_t>,
        thrust::device_vector<float>,
        thrust::device_vector<float>>
    (
        std::move(nlen),
        std::move(x_com),
        std::move(y_com)
    );
}

void ParallelQuadtreeBuilder::normalize_source_data(void)
{
    float *x_d = x.data().get(), *y_d = y.data().get();
    float x_min_d = x_min, x_max_d = x_max, y_min_d = y_min, y_max_d = y_max; 
    thrust::for_each(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(x.size()),
        [x_d, y_d, x_min_d, x_max_d, y_min_d, y_max_d]
        __device__ __host__ (uint32_t i)
        {
            x_d[i] = x_d[i] * (x_max_d - x_min_d) + x_min_d;
            y_d[i] = y_d[i] * (y_max_d - y_min_d) + y_min_d;
        }
    );
}

std::tuple<thrust::device_vector<uint32_t>,
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint8_t>>
ParallelQuadtreeBuilder::fill_tree(
    thrust::device_vector<uint32_t> nlen,
    thrust::device_vector<uint32_t> start,
    thrust::device_vector<uint8_t> clen)
{
    thrust::device_vector<uint8_t> is_leaf(nlen.size());
    size_t threshold = T;
    uint32_t *nlen_d = nlen.data().get();
    uint8_t *clen_d = clen.data().get();
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(is_leaf.size()),
        is_leaf.begin(),
        [nlen_d, clen_d, threshold] __device__ __host__ (uint32_t i){
            /* If less than threshold or no children => node is leaf */
            return (uint8_t)(nlen_d[i] <= threshold || clen_d[i] == 0);
        }
    );

    /* set clen for leaf nodes to 0 - such that it contributes 0 to prefix sum */
    thrust::replace_if(clen.begin(), clen.end(), is_leaf.begin(),
        [] __device__ __host__ (uint8_t mask) { return mask; }, 0);

    thrust::device_vector<uint32_t> cpos(is_leaf.size()), f_pos(is_leaf.size());

    thrust::exclusive_scan(
        clen.begin(), clen.end(), cpos.begin(), uint32_t{1}
    );

    /* This block can be computed asynchronously on differnt streams */
    uint8_t *is_leaf_d = is_leaf.data().get();
    uint32_t *start_d = start.data().get();
    uint32_t *cpos_d = cpos.data().get();
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(is_leaf.size()),
        f_pos.begin(), [start_d, cpos_d, is_leaf_d] __device__ __host__ (uint32_t i) {
            return is_leaf_d[i] ? start_d[i] : cpos_d[i];
        }
    );

    /* Fill out length as last step - if non-leaf use clen, if leaf use nlen */
    thrust::device_vector<uint32_t> length(nlen.size());
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(nlen.size()),
        length.begin(),
        [is_leaf_d, clen_d, nlen_d] __device__ __host__ (uint32_t i){
            return is_leaf_d[i] ? nlen_d[i] : clen_d[i];
        }
    );

    /* Normalize center of mass */

    return std::make_tuple<
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>>
    (
        std::move(nlen), std::move(f_pos),
        std::move(length), std::move(is_leaf)
    );
}