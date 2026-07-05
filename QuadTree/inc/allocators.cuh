#ifndef QUAD_TREE_ALLOCATORS
#define QUAD_TREE_ALLOCATORS

#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

template <typename T>
struct DeviceGenericAllocator
{
    using value_type = T;
    using pointer = thrust::device_ptr<T>;
    using size_type = std::size_t;

    template <typename U>
    struct rebind
    {
        using other = DeviceGenericAllocator<U>;
    };

    DeviceGenericAllocator() = default;
    ~DeviceGenericAllocator() = default;

    inline pointer allocate(size_type n)
    {
        T *raw_ptr = nullptr;
        size_t size = n * sizeof(T);

        cudaError_t status = cudaMalloc(&raw_ptr, size);

        if (status != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed!");
        }

        return thrust::device_pointer_cast(raw_ptr);
    }

    inline void deallocate(pointer ptr, size_type n)
    {
        cudaFree(ptr.get());
    }
};

class GpuArena
{
public:
    GpuArena(size_t size) : total_size(size), offset(0)
    {
        cudaMalloc(&base_ptr, total_size);
    }

    GpuArena(const GpuArena &) = delete;
    GpuArena &operator=(const GpuArena &) = delete;

    GpuArena(GpuArena&& other){
        std::swap(other.base_ptr, base_ptr);
        std::swap(other.total_size, total_size);
        std::swap(other.offset, offset);
    }

    GpuArena &operator=(GpuArena&& other){
        std::swap(other.base_ptr, base_ptr);
        std::swap(other.total_size, total_size);
        std::swap(other.offset, offset);

        return *this;
    }

    ~GpuArena()
    {
        cudaFree(base_ptr);
    }

    inline void *allocate(size_t bytes)
    {
        size_t aligned_bytes = (bytes + 255) & ~255;

        if (offset + aligned_bytes > total_size)
        {
            throw std::runtime_error("Arena to small\n");
        }

        void *ptr = static_cast<char *>(base_ptr) + offset;
        offset += aligned_bytes;
        return ptr;
    }

    inline void deallocate(void *ptr, size_t bytes)
    {
        /* Deallocation does not matter - arena has to be reset manually */
    }

    inline void reset()
    {
        offset = 0;
    }

private:
    void *base_ptr = nullptr;
    size_t total_size;
    size_t offset;
};

/* 
Proof of concept areana allocator - managed to get ~20ms for 1 mln points:
- No arena version 1 mln points - ~100ms
- Arena version on some intermediate vectors 1 mln points - ~81ms
*/
template <typename T>
class DeviceArenaAllocator
{
public:
    using value_type = T;
    using pointer = thrust::device_ptr<T>;
    using size_type = std::size_t;

    template <typename U>
    struct rebind
    {
        using other = DeviceArenaAllocator<U>;
    };

    DeviceArenaAllocator(GpuArena *arena): arena(arena) {};
    ~DeviceArenaAllocator() = default;

    inline pointer allocate(size_type n)
    {
        return thrust::device_pointer_cast(arena->allocate(n * sizeof(T)));
    }

    inline void deallocate(pointer ptr, size_type n)
    {
        /* Deallocation does not matter */
    }

private:
    GpuArena *arena;
};

#endif