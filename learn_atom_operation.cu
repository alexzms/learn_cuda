/*
 * This program will compute histogram using atom operation
 */

#include "./includes/book.h"
#include "iostream"
#include "chrono"
#include "check_error.cuh"
#include "memory"

#define DATA_SIZE (100*1024*1024)

// Time to generate: 43.5902 ms
__global__ void calculate_histo_kernel(const unsigned char *data, unsigned int *histo) {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    // computation area moves forward stride at each timestep
    unsigned int stride = blockDim.x * gridDim.x;
    while (offset < DATA_SIZE) {
        atomicAdd(&(histo[data[offset]]), 1);
        offset += stride;
    }
}

// Time to generate: 15.5934 ms
__global__ void shared_calculate_histo_kernel(const unsigned char *data, unsigned int *histo) {
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;                          // make sure that in every block, first thing we do is init all 0
    __syncthreads();                                // in temp, so that later calculation will not be wrong
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    // computation area moves forward stride at each timestep
    unsigned int stride = blockDim.x * gridDim.x;
    while (offset < DATA_SIZE) {
        atomicAdd(&(temp[data[offset]]), 1);
        offset += stride;
    }
    __syncthreads();                                // all computation in shared memory is done, now every thread takes
    atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);     // the responsibility of one histo, move it!
}

int main() {
    std::unique_ptr<unsigned char[]> data{(unsigned char*) big_random_block(DATA_SIZE)};
    // unsigned char has 256 possibilities
    unsigned int h_histo_gt[256];
    for (auto & i : h_histo_gt) {
        i = 0;
    }
    auto start_cpu_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i != DATA_SIZE; ++i) {
        h_histo_gt[data[i]] += 1;
    }
    auto end_cpu_time = std::chrono::high_resolution_clock::now();
    unsigned int sum_test_gt = 0;
    for (auto i : h_histo_gt) {
        sum_test_gt += i;
    }
    // GT(CPU) is correct! Time: 313 ms
    if (sum_test_gt == DATA_SIZE) {
        std::cout << "GT(CPU) is correct! Time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu_time - start_cpu_time).count()
                << " ms" << std::endl;
    }

    cudaEvent_t start, stop;
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start, 0));

    unsigned char *d_data;
    unsigned int *d_histo;
    std::unique_ptr<unsigned int[]> h_histo{(unsigned int*)malloc(256 * sizeof(int))};
    CHECK_ERROR(cudaMalloc((void**)&d_data, DATA_SIZE));
    CHECK_ERROR(cudaMalloc((void**)&d_histo, 256 * sizeof(int)));
    CHECK_ERROR(cudaMemcpy(d_data, data.get(), DATA_SIZE, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemset(d_histo, 0, 256 * sizeof(int)));

    cudaDeviceProp prop{};
    CHECK_ERROR(cudaGetDeviceProperties(&prop, 0));
    int blocks = 2 * prop.multiProcessorCount;
    std::cout << "blocks = " << blocks << std::endl;
    // computation area: blocks * 256, this needs to loop over all data
    shared_calculate_histo_kernel<<<blocks, 256>>>(d_data, d_histo);

    CHECK_ERROR(cudaMemcpy(h_histo.get(), d_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaEventRecord(stop, 0));
    CHECK_ERROR(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
    std::cout << "Time to generate: " << elapsed_time << " ms" << std::endl;
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    bool any_error = false;
    // check result with ground truth
    for (int i = 0; i != 256; ++i) {
        if (h_histo[i] != h_histo_gt[i]) {
            std::cout << "Error: " << i << " " << h_histo[i] << " " << h_histo_gt[i] << std::endl;
            any_error = true;
        }
    }
    if (!any_error) {
        std::cout << "Test passed!" << std::endl;
    }

    std::cout << "Test finished!" << std::endl;
    cudaFree(d_data);
    cudaFree(d_histo);
    // all the ptr is handled by intelligent ptr
    return 0;
}