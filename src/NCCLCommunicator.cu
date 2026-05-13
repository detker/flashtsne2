#include "ICommunicator.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>



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


thrust::device_vector<float> NCCLCommunicator::distributeData(faiss::idx_t *assignments, int dim,
                                                              float *local_x, int n_local) {
  cudaSetDevice(_local_gpu_id);
  auto get_target_node = [=](int cluster_id) { return cluster_id % _size; };

  std::vector<faiss::idx_t> h_assignments(n_local);
  cudaMemcpy(h_assignments.data(), assignments, n_local * sizeof(faiss::idx_t),
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

  std::vector<int> sorted_indices(n_local);
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
    return get_target_node(h_assignments[a]) < get_target_node(h_assignments[b]);
  });

  int cluster_n = 0;
  for (int c = 0; c < _size; c++) {
    cluster_n += receive_counts[c];
  }

  std::vector<float> h_send(n_local * dim);
  for (int i = 0; i < n_local; i++) {
    int idx = sorted_indices[i];
    std::memcpy(h_send.data() + i * dim, local_x + idx * dim, dim * sizeof(float));
  }

  std::vector<float> h_recv(cluster_n);

  allToAllV<CommunicationBackend::MPI>(
      h_send.data(), send_counts.data(), send_displs.data(), CommDataType::FLOAT,
      h_recv.data(), receive_counts.data(), receive_displs.data(), CommDataType::FLOAT);

  thrust::device_vector<float> d_recv(h_recv.begin(), h_recv.end());
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
