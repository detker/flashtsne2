#include "ICommunicator.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>


__global__ void gather_rows(const float* src, float* dst, const int* indices, int n, int dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * dim) return;
    int row = tid / dim;
    int col = tid % dim;
    dst[tid] = src[indices[row] * dim + col];
}


NCCLCommunicator::NCCLCommunicator(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_size);

  cudaGetDeviceCount(&_gpus_per_node);
  _local_gpu_id = _rank % _gpus_per_node;

  cudaSetDevice(_local_gpu_id);
  cudaStreamCreate(&_stream);
  _nccl_comm = nullptr;
}


void NCCLCommunicator::initNCCL() {
  if (_nccl_comm != nullptr) return;
  cudaSetDevice(_local_gpu_id);
  if (_rank == 0) {
    ncclGetUniqueId(&_nccl_id);
  }
  MPI_Bcast(&_nccl_id, sizeof(_nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&_nccl_comm, _size, _nccl_id, _rank);
}


NCCLCommunicator::~NCCLCommunicator() {
  if (_nccl_comm) ncclCommDestroy(_nccl_comm);
  cudaStreamDestroy(_stream);
  MPI_Finalize();
}


int NCCLCommunicator::getRank() const { return _rank; }


int NCCLCommunicator::getSize() const { return _size; }


thrust::device_vector<float> NCCLCommunicator::distributeData(faiss::idx_t *d_assignments, int dim,
                                                              float *d_local_x, int n_local,
                                                              thrust::device_vector<int64_t> *row_ids) {
  cudaSetDevice(_local_gpu_id);
  auto get_target_node = [=](int cluster_id) { return cluster_id % _size; };

  std::vector<faiss::idx_t> h_assignments(n_local);
  cudaMemcpy(h_assignments.data(), d_assignments, n_local * sizeof(faiss::idx_t),
             cudaMemcpyDeviceToHost);

  std::vector<int> send_counts(_size, 0);
  for (auto &label : h_assignments) {
    send_counts[get_target_node(label)] += dim;
  }

  std::vector<int> receive_counts(_size);
  allToAll<CommunicationBackend::MPI>(
      send_counts.data(), 1, CommDataType::INT,
      receive_counts.data(), 1, CommDataType::INT);

  std::vector<int> send_displs(_size, 0);
  std::vector<int> receive_displs(_size, 0);
  for (int i = 1; i < _size; i++) {
    send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
    receive_displs[i] = receive_displs[i - 1] + receive_counts[i - 1];
  }

  int total_recv = 0;
  for (int c = 0; c < _size; c++) {
    total_recv += receive_counts[c];
  }

  thrust::device_vector<int> d_sort_keys(n_local);
  {
    std::vector<int> h_sort_keys(n_local);
    for (int i = 0; i < n_local; i++) {
      h_sort_keys[i] = get_target_node(h_assignments[i]);
    }
    thrust::copy(h_sort_keys.begin(), h_sort_keys.end(), d_sort_keys.begin());
  }

  thrust::device_vector<int> d_indices(n_local);
  thrust::sequence(d_indices.begin(), d_indices.end());
  thrust::sort_by_key(d_sort_keys.begin(), d_sort_keys.end(), d_indices.begin());

  float *d_send = nullptr;
  cudaMalloc(&d_send, (size_t)n_local * dim * sizeof(float));
  {
    int total = n_local * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    gather_rows<<<grid, block>>>(d_local_x, d_send,
                                 thrust::raw_pointer_cast(d_indices.data()), n_local, dim);
    cudaStreamSynchronize(0);
  }

  thrust::device_vector<float> d_recv(total_recv);
  allToAllV<CommunicationBackend::NCCL>(
      d_send, send_counts.data(), send_displs.data(), CommDataType::FLOAT,
      thrust::raw_pointer_cast(d_recv.data()), receive_counts.data(), receive_displs.data(), CommDataType::FLOAT);

  cudaFree(d_send);

  if (row_ids) {
    thrust::device_vector<int64_t> d_send_ids(n_local);
    thrust::gather(d_indices.begin(), d_indices.end(),
                   row_ids->begin(), d_send_ids.begin());

    std::vector<int> send_rows(_size), recv_rows(_size);
    std::vector<int> sdispl_rows(_size), rdispl_rows(_size);
    for (int i = 0; i < _size; i++) {
      send_rows[i]   = send_counts[i] / dim;
      recv_rows[i]   = receive_counts[i] / dim;
      sdispl_rows[i] = send_displs[i] / dim;
      rdispl_rows[i] = receive_displs[i] / dim;
    }

    thrust::device_vector<int64_t> d_recv_ids(total_recv / dim);
    allToAllV<CommunicationBackend::NCCL>(
        thrust::raw_pointer_cast(d_send_ids.data()), send_rows.data(), sdispl_rows.data(),
        CommDataType::UINT64,
        thrust::raw_pointer_cast(d_recv_ids.data()), recv_rows.data(), rdispl_rows.data(),
        CommDataType::UINT64);
    *row_ids = std::move(d_recv_ids);
  }

  return d_recv;
}


ncclDataType_t NCCLCommunicator::mapTypeNCCL(CommDataType t) {
  if (t == CommDataType::FLOAT)
    return ncclFloat;
  if (t == CommDataType::INT)
    return ncclInt;
  if (t == CommDataType::SIZE_T || t == CommDataType::UINT64)
    return ncclUint64;
  return static_cast<ncclDataType_t>(-1);
}


ncclRedOp_t NCCLCommunicator::mapOpNCCL(CommOp op) {
  switch (op) {
  case CommOp::SUM:
    return ncclSum;
  case CommOp::MAX:
    return ncclMax;
  case CommOp::MIN:
    return ncclMin;
  default:
    return static_cast<ncclRedOp_t>(-1);
  }
}


size_t NCCLCommunicator::typeSizeNCCL(CommDataType t) {
  switch (t) {
  case CommDataType::FLOAT:
    return sizeof(float);
  case CommDataType::INT:
    return sizeof(int);
  case CommDataType::SIZE_T:
    return sizeof(size_t);
  case CommDataType::UINT64:
    return sizeof(uint64_t);
  default:
    return 0;
  }
}


MPI_Datatype NCCLCommunicator::mapTypeMPI(CommDataType t) {
    if (t == CommDataType::FLOAT) return MPI_FLOAT;
    if (t == CommDataType::INT) return MPI_INT;
    if (t == CommDataType::SIZE_T || t == CommDataType::UINT64) return MPI_UINT64_T;
    return MPI_DATATYPE_NULL;
}


MPI_Op NCCLCommunicator::mapOpMPI(CommOp op) {
    switch (op) {
        case CommOp::SUM: return MPI_SUM;
        case CommOp::MAX: return MPI_MAX;
        case CommOp::MIN: return MPI_MIN;
        default: return MPI_OP_NULL;
    }
}


thrust::device_vector<float> NCCLCommunicator::ringExchange(thrust::device_vector<float> out) {
  if (_size == 1) return out;
  const int right = (_rank + 1) % _size;
  const int left  = (_rank - 1 + _size) % _size;

  uint64_t send_n = out.size(), recv_n = 0;
  MPI_Sendrecv(&send_n, 1, MPI_UINT64_T, right, 0,
               &recv_n, 1, MPI_UINT64_T, left, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  thrust::device_vector<float> in(recv_n);
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclSend(out.data().get(), send_n, ncclFloat, right, _nccl_comm, _stream));
  NCCL_CHECK(ncclRecv(in.data().get(), recv_n, ncclFloat, left, _nccl_comm, _stream));
  NCCL_CHECK(ncclGroupEnd());
  CUDA_CHECK(cudaStreamSynchronize(_stream));
  return in;
}
