#include "ICommunicator.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <thrust/device_vector.h>
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
                                                              float *local_x, int n_local) {
  cudaSetDevice(_local_gpu_id);
  auto get_target_node = [=](int cluster_id) { return cluster_id % _size; };

  // Copy assignments to host for metadata computation
  std::vector<faiss::idx_t> h_assignments(n_local);
  cudaMemcpy(h_assignments.data(), d_assignments, n_local * sizeof(faiss::idx_t),
             cudaMemcpyDeviceToHost);

  // Compute per-rank send counts (metadata on CPU)
  std::vector<int> send_counts(_size, 0);
  for (auto &label : h_assignments) {
    send_counts[get_target_node(label)] += dim;
  }

  // Exchange counts via MPI (small metadata)
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

  // Compute sort keys on host, then sort on device
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

  // Copy local_x from host to device
  float *d_local_x;
  cudaMalloc(&d_local_x, (size_t)n_local * dim * sizeof(float));
  cudaMemcpy(d_local_x, local_x, (size_t)n_local * dim * sizeof(float),
             cudaMemcpyHostToDevice);

  thrust::device_vector<float> d_send(n_local * dim);
  {
    int total = n_local * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    gather_rows<<<grid, block>>>(d_local_x, thrust::raw_pointer_cast(d_send.data()),
                                 thrust::raw_pointer_cast(d_indices.data()), n_local, dim);
    cudaStreamSynchronize(0);
  }

  cudaFree(d_local_x);

  // Bulk data exchange via NCCL
  thrust::device_vector<float> d_recv(total_recv);
  allToAllV<CommunicationBackend::NCCL>(
      thrust::raw_pointer_cast(d_send.data()), send_counts.data(), send_displs.data(), CommDataType::FLOAT,
      thrust::raw_pointer_cast(d_recv.data()), receive_counts.data(), receive_displs.data(), CommDataType::FLOAT);

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
