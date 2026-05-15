#pragma once

#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <cstdlib>
#include <limits>
#include <random>
#include <strings.h>
#include <faiss/Index.h>
#include <vector>
#include <nccl.h>
#include <thrust/device_vector.h>
#include <type_traits>
#include <cuda_runtime.h>

#include "error_utils.hpp"


enum class CommDataType { FLOAT, INT, SIZE_T, UINT64 };
enum class CommOp { SUM, MAX, MIN };
struct CommunicationBackend {
    struct MPI {};
    struct NCCL {};
};


class NCCLCommunicator {
public:
    NCCLCommunicator(int argc, char** argv);
    ~NCCLCommunicator();
    void initNCCL();

    int getRank() const;
    int getSize() const;

    template <typename T>
    void allReduce(void* sendbuff, void* recvbuff, size_t count,
                   CommDataType type, CommOp op);

    template <typename T>
    void broadcast(void* buffer, int count, CommDataType type, int root);

    template <typename T>
    void gather(const void* sendbuff, int sendcount, CommDataType sendType,
                void* recvbuff, int recvcount, CommDataType recvType, int root);

    template <typename T>
    void gatherv(const void* sendbuff, int sendcount, CommDataType sendType,
                 void* recvbuff, const int* recvcounts, const int* displs,
                 CommDataType recvType, int root);

    template <typename T>
    void allGather(const void* sendbuff, int sendcount, CommDataType sendType,
                   void* recvbuff, int recvcount, CommDataType recvType);

    template <typename T>
    void allToAll(const void* sendbuff, int sendcount, CommDataType sendType,
                  void* recvbuff, int recvcount, CommDataType recvType);

    template <typename T>
    void allToAllV(const void* sendbuff, const int* sendcounts, const int* sdispls,
                   CommDataType sendType, void* recvbuff, const int* recvcounts,
                   const int* rdispls, CommDataType recvType);

    thrust::device_vector<float> distributeData(faiss::idx_t *d_assignments, int dim,
                                                float *d_local_x, int n_local);

    static ncclDataType_t mapTypeNCCL(CommDataType t);
    static size_t typeSizeNCCL(CommDataType t);
    static ncclRedOp_t mapOpNCCL(CommOp op);
    static MPI_Datatype mapTypeMPI(CommDataType t);
    static MPI_Op mapOpMPI(CommOp op);

private:
    int _rank;
    int _size;

    int _gpus_per_node;
    int _local_gpu_id;

    ncclUniqueId _nccl_id;
    ncclComm_t _nccl_comm;
    cudaStream_t _stream;
};


template <typename T>
void NCCLCommunicator::allReduce(void *sendbuff, void *recvbuff, size_t count,
                                 CommDataType type, CommOp op) {
    if constexpr (std::is_same_v<T, CommunicationBackend::NCCL>) {
        const void *src = (sendbuff == nullptr) ? recvbuff : sendbuff;
        ncclAllReduce(src, recvbuff, count, mapTypeNCCL(type), mapOpNCCL(op),
                      _nccl_comm, _stream);
        cudaStreamSynchronize(_stream);
    } else {
        MPI_Allreduce(sendbuff == nullptr ? MPI_IN_PLACE : sendbuff, recvbuff,
                      count, mapTypeMPI(type), mapOpMPI(op), MPI_COMM_WORLD);
    }
}

template <typename T>
void NCCLCommunicator::broadcast(void *buffer, int count, CommDataType type,
                                 int root) {
    if constexpr (std::is_same_v<T, CommunicationBackend::NCCL>) {
        ncclBroadcast(buffer, buffer, count, mapTypeNCCL(type), root,
                      _nccl_comm, _stream);
        cudaStreamSynchronize(_stream);
    } else {
        MPI_Bcast(buffer, count, mapTypeMPI(type), root, MPI_COMM_WORLD);
    }
}

template <typename T>
void NCCLCommunicator::gather(const void *sendbuff, int sendcount,
                              CommDataType sendType, void *recvbuff,
                              int recvcount, CommDataType recvType, int root) {
    if constexpr (std::is_same_v<T, CommunicationBackend::NCCL>) {
        ncclGroupStart();
        ncclSend(sendbuff, sendcount, mapTypeNCCL(sendType), root,
                 _nccl_comm, _stream);
        if (_rank == root) {
            for (int i = 0; i < _size; i++) {
                void *dst = static_cast<char *>(recvbuff) +
                            i * recvcount * typeSizeNCCL(recvType);
                ncclRecv(dst, recvcount, mapTypeNCCL(recvType), i,
                         _nccl_comm, _stream);
            }
        }
        ncclGroupEnd();
        cudaStreamSynchronize(_stream);
    } else {
        MPI_Gather(sendbuff, sendcount, mapTypeMPI(sendType),
                   recvbuff, recvcount, mapTypeMPI(recvType),
                   root, MPI_COMM_WORLD);
    }
}

template <typename T>
void NCCLCommunicator::gatherv(const void *sendbuff, int sendcount,
                               CommDataType sendType, void *recvbuff,
                               const int *recvcounts, const int *displs,
                               CommDataType recvType, int root) {
    if constexpr (std::is_same_v<T, CommunicationBackend::NCCL>) {
        ncclGroupStart();
        ncclSend(sendbuff, sendcount, mapTypeNCCL(sendType), root,
                 _nccl_comm, _stream);
        if (_rank == root) {
            for (int i = 0; i < _size; i++) {
                void *dst = static_cast<char *>(recvbuff) +
                            displs[i] * typeSizeNCCL(recvType);
                ncclRecv(dst, recvcounts[i], mapTypeNCCL(recvType), i,
                         _nccl_comm, _stream);
            }
        }
        ncclGroupEnd();
        cudaStreamSynchronize(_stream);
    } else {
        MPI_Gatherv(sendbuff, sendcount, mapTypeMPI(sendType),
                    recvbuff, recvcounts, displs,
                    mapTypeMPI(recvType), root, MPI_COMM_WORLD);
    }
}

template <typename T>
void NCCLCommunicator::allGather(const void *sendbuff, int sendcount,
                                 CommDataType sendType, void *recvbuff,
                                 int recvcount, CommDataType recvType) {
    if constexpr (std::is_same_v<T, CommunicationBackend::NCCL>) {
        ncclAllGather(sendbuff, recvbuff, sendcount, mapTypeNCCL(sendType),
                      _nccl_comm, _stream);
        cudaStreamSynchronize(_stream);
    } else {
        MPI_Allgather(sendbuff, sendcount, mapTypeMPI(sendType),
                      recvbuff, recvcount, mapTypeMPI(recvType),
                      MPI_COMM_WORLD);
    }
}

template <typename T>
void NCCLCommunicator::allToAll(const void *sendbuff, int sendcount,
                                CommDataType sendType, void *recvbuff,
                                int recvcount, CommDataType recvType) {
    if constexpr (std::is_same_v<T, CommunicationBackend::NCCL>) {
        size_t send_elem = typeSizeNCCL(sendType);
        size_t recv_elem = typeSizeNCCL(recvType);

        ncclGroupStart();
        for (int i = 0; i < _size; i++) {
            const void *src =
                static_cast<const char *>(sendbuff) + i * sendcount * send_elem;
            void *dst = static_cast<char *>(recvbuff) + i * recvcount * recv_elem;
            ncclSend(src, sendcount, mapTypeNCCL(sendType), i, _nccl_comm, _stream);
            ncclRecv(dst, recvcount, mapTypeNCCL(recvType), i, _nccl_comm, _stream);
        }
        ncclGroupEnd();
        cudaStreamSynchronize(_stream);
    } else {
        MPI_Alltoall(sendbuff, sendcount, mapTypeMPI(sendType),
                     recvbuff, recvcount, mapTypeMPI(recvType), MPI_COMM_WORLD);
    }
}

template <typename T>
void NCCLCommunicator::allToAllV(const void *sendbuff, const int *sendcounts,
                                 const int *sdispls, CommDataType sendType,
                                 void *recvbuff, const int *recvcounts,
                                 const int *rdispls, CommDataType recvType) {
    if constexpr (std::is_same_v<T, CommunicationBackend::NCCL>) {
        size_t send_elem = typeSizeNCCL(sendType);
        size_t recv_elem = typeSizeNCCL(recvType);

        ncclGroupStart();
        for (int i = 0; i < _size; i++) {
            const void *src =
                static_cast<const char *>(sendbuff) + sdispls[i] * send_elem;
            void *dst = static_cast<char *>(recvbuff) + rdispls[i] * recv_elem;
            ncclSend(src, sendcounts[i], mapTypeNCCL(sendType), i, _nccl_comm, _stream);
            ncclRecv(dst, recvcounts[i], mapTypeNCCL(recvType), i, _nccl_comm, _stream);
        }
        ncclGroupEnd();
        cudaStreamSynchronize(_stream);
    } else {
        MPI_Alltoallv(sendbuff, sendcounts, sdispls, mapTypeMPI(sendType),
                      recvbuff, recvcounts, rdispls, mapTypeMPI(recvType),
                      MPI_COMM_WORLD);
    }
}
