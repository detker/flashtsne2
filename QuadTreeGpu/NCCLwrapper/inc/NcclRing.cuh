#ifndef NCCL_RING_WRAPPER
#define NCCL_RING_WRAPPER

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <stdexcept>
#include <vector>

#define NCCL_CHECK(cmd)                                      \
    do {                                                     \
        ncclResult_t r = cmd;                                \
        if (r != ncclSuccess)                                \
            throw std::runtime_error(ncclGetErrorString(r)); \
    } while (0)

#define CUDA_CHECK(cmd)                                      \
    do {                                                     \
        cudaError_t e = cmd;                                 \
        if (e != cudaSuccess)                                \
            throw std::runtime_error(cudaGetErrorString(e)); \
    } while (0)

class NcclRing {
public:
    NcclRing(MPI_Comm mpi_comm)
        : mpi_comm_(mpi_comm)
    {
        MPI_Comm_rank(mpi_comm_, &rank_);
        MPI_Comm_size(mpi_comm_, &size_);

        /* Find local rank within global world for GPU selection */
        MPI_Comm local_comm;
        MPI_Comm_split_type(
            MPI_COMM_WORLD,
            MPI_COMM_TYPE_SHARED, 
            0,
            MPI_INFO_NULL,
            &local_comm
        );  

        int local_rank;
        MPI_Comm_rank(local_comm, &local_rank);

        /* Set GPU - each MPI spawned process will own single GPU */
        CUDA_CHECK(cudaSetDevice(local_rank));
        CUDA_CHECK(cudaStreamCreate(&stream_));

        /* Generate NCCL unique id on global rank 0 */
        ncclUniqueId id;
        if (rank_ == 0)
            NCCL_CHECK(ncclGetUniqueId(&id));

        /* Broadcast NCCL session id */
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm_);

        /* Initialize with global rank */
        NCCL_CHECK(ncclCommInitRank(&comm_, size_, id, rank_));
    }

    template <typename T>
    inline thrust::device_vector<T> ring_exchange(thrust::device_vector<T> out){
        size_t send_size = out.size(), recv_size;

        ncclGroupStart();
        ncclSend( &send_size, 1,
            ncclUint64, right(),
            comm_, stream_);

        ncclRecv(
            &recv_size, 1, 
            ncclUint64, left(),
            comm_, stream_);

        ncclGroupEnd();
        cudaStreamSynchronize(stream_);

        thrust::device_vector<T> in(recv_size);

        ncclGroupStart();
        ncclSend(
            out.data().get(), send_size,
            ncclChar, right(),
            comm_, stream_);

        ncclRecv(
            in.data().get(), recv_size, 
            ncclChar, left(),
            comm_, stream_);

        ncclGroupEnd();
        cudaStreamSynchronize(stream_);

        return std::move(in);
    }

    ~NcclRing()
    {
        if (comm_)
            ncclCommDestroy(comm_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    inline int rank() const { return rank_; }
    inline int size() const { return size_; }

    inline int left() const
    {
        return (rank_ - 1 + size_) % size_;
    }

    inline int right() const
    {
        return (rank_ + 1) % size_;
    }

    cudaStream_t stream() const { return stream_; }
    ncclComm_t comm() const { return comm_; }

private:
    MPI_Comm mpi_comm_;

    int rank_;
    int size_;

    ncclComm_t comm_{};
    cudaStream_t stream_{};
};

#endif