#pragma once
#include "cstdio"

void check_error_function(cudaError_t status, int line, const char *file) {
    if (status != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(status), file, line);
//        exit(EXIT_FAILURE);
    }
}

#define CHECK_ERROR(status) (check_error_function(status, __LINE__, __FILE__))

