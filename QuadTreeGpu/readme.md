Parallel-primitive based quad-tree construction on GPU and CPU using Thrust.

Refrence: https://adms-conf.org/2019-camera-ready/zhang_adms19.pdf

Note: Fix building for stanalone library for linking to pure C++ projects.

# Issues
## Memory allocation synchronization bottleneck
Using Nsight Compute and Nsight Systems tools following observations an be made:
1. Total kernel computation takes time in tens of ms dominating operation being sort which took ~4ms.
2. Execution is dominated by cudaMalloc, cudaFree and synchronization between streams.

In code there can be seen that there are many allocations/deallocations of temporary vectors which are required in the algorithm.

## Traversal

# Summary of level on finest level
If this is the lowest level we can compute center of mass
from points directly. Looking at x axis:
    x_com = SUM_i (m_i * x_i) / SUM_i (m_i)
So we got two sums to compute. This has to be done for each
quadrant. For instance quadrant k ranges from i_{begin_k} to
i_{end_k}. If we have over all scan we can see that 
    SUM(points in quadrant k) = SCAN[i_{end_k}] - SCAN[i_{begin_k}]
One issue might be numerics - i mean points are normalized to [0, 1]
range but scan for large number of points might get quite big and after
that we divide - we lose precision IDK.

If mass is used, this should be passed to inclusive sum.
```C++
auto begin = thrust::make_transform_iterator(
    thrust::make_zip_iterator(
        thrust::make_tuple(
            x.begin(),
            thrust::make_constant_iterator<float>(1.0f)
        )
    ),
    [] __device__ __host__ (thrust::tuple<float, float> t) 
    {
        float x = thrust::get<0>(t);
        float m = thrust::get<1>(t);
        return x * m;
    }
);
```

# Summary of level if we have prev
We have nlen of each child as well as COM of each child. Simply do weighted average of those.

# TODO
