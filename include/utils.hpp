#pragma once

#include <iostream>
#include "error_utils.hpp"


typedef struct {
    const char *data_path;
    const int dim;
    const int k;
    const int niter;
} Config; 


inline Config parseArgs(int argc, char **argv)
{
    if (argc < 4){
        std::cout << "args: " << argc << std::endl;
        usage(argv[0]);
    }    

    Config config = {
        .data_path = argv[1],
        .dim = std::stoi(argv[2]),
        .k = std::stoi(argv[3]),
        .niter = std::stoi(argv[4])
    };
    return config;
}
