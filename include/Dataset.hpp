#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "error_utils.hpp"


class Dataset {
    int fd;
    float* data_ptr;
    size_t total_bytes;
    size_t mapped_size;

public:
    Dataset(const char* filename) {
        fd = open(filename, O_RDONLY);
        if (fd == -1) ERR("cannot open file"); 

        struct stat sb;
        if (fstat(fd, &sb) == -1) ERR("fstat failed");
        total_bytes = sb.st_size;

        data_ptr = (float*)mmap(NULL, total_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data_ptr == MAP_FAILED) ERR("mmap failed");
        else {
            madvise(data_ptr, total_bytes, MADV_SEQUENTIAL | MADV_WILLNEED);
        }
    }

    float* get_shard_ptr(size_t start_idx, int d) {
        return data_ptr + (start_idx * d);
    }

    size_t get_total_vectors(int d) const {
        return total_bytes / (sizeof(float) * d);
    }

    ~Dataset() {
        if (data_ptr != MAP_FAILED) munmap(data_ptr, total_bytes);
        if (fd != -1) close(fd);
    }
};
