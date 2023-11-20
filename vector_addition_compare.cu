/*
 * This program is rather easy, it just adds up two vectors, and check for the error between ground truth by cpu
 * and results by cuda:gpu.
 */


#include <iostream>
#include "includes/book.h"
#include <chrono>

#define LENGTH 10000000

unsigned int max_block = 0;
unsigned int max_thread = 0;

int chapter5_any_length();
int chapter5_block_threads();
int chapter5_only_threads();

__global__ void add_vector_any_length(const int *a, const int *b, int *c) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < LENGTH) {
        c[tid] = a[tid] + b[tid];
        // input is a blockDim.x * gridDim.x computation area
        // and every time, step ahead by a whole computation area
        // total parallel threads: blockDim.x * gridDim.x, each compute LENGTH/(blockDim.x * gridDim.x)
        //  addition operations
        tid += blockDim.x * gridDim.x;
    }
}

/*
 * blockDim就是block中threads的形状，例如如果一个block有4x4=16个threads，那么blockDim=(4,4)
 */
__global__ void add_vector_block_threads(const int *a, const int *b, int *c) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < LENGTH) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void add_vector_only_threads(const int *a, const int *b, int *c) {
    unsigned int tid = threadIdx.x;
    if (tid < LENGTH) {
        c[tid] = a[tid] + b[tid];
    }
}


int main() {
    int device_count;
    HANDLE_ERROR(cudaGetDeviceCount(&device_count));
    std::cout << "Device count: " << device_count << std::endl;
    for (int i = 0; i != device_count; ++i) {
        std::cout << "Fetching device properties for device " << i << std::endl;
        cudaDeviceProp device_prop{};
        HANDLE_ERROR(cudaGetDeviceProperties(&device_prop, i));
        std::cout << "Device name: " << device_prop.name << std::endl;
        std::cout << "Total global memory: " << device_prop.totalGlobalMem << std::endl;
        // max blocks and max threads per block
        max_block = device_prop.maxGridSize[0];
        max_thread = device_prop.maxThreadsPerBlock;
        std::cout << "Max blocks per grid: " << device_prop.maxGridSize[0] << std::endl;
        std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
    }
//    std::cout << "-----Calling chapter5_only_threads()-----" << std::endl;
    int result_only_threads = 0; // only_threads only supports LENGTH<=1024
    std::cout << "-----Calling chapter5_any_length()-----" << std::endl;
    int result_any_length = chapter5_any_length();
    std::cout << "-----Calling chapter5_block_threads()-----" << std::endl;
    int result_block_threads = chapter5_block_threads();

    return result_only_threads + result_block_threads + result_any_length;
}


/*
 * Chapter 5 of CUDA by Example
 * Vector addition with any length, implemented by the idea of "grid-stride loop"
 * Typical result(Length == 10000000):
 * The result is valid
 *  CPU time: 13654 us
 *  GPU memory time: 126437 us(mem1 copy) + 7736 us(calc) + 13259 us(mem2 copy) = 147434 us
 * The calc time can be fine-tuned by changing the grid_length and block_length, and
 * By unsigned int grid_length = 65536; unsigned int block_length = 1024; we can run faster
 * than chapter5_block_threads() with the same length.
 */
int chapter5_any_length() {
    // define computation area
    unsigned int grid_length = 65536;
    unsigned int block_length = 1024;
    bool is_valid = true;
    int *a = (int*)malloc(LENGTH * sizeof(int));
    int *b = (int*)malloc(LENGTH * sizeof(int));
    int *c_validation = (int*)malloc(LENGTH * sizeof(int));
    int *c = (int*)malloc(LENGTH * sizeof(int));
    int *dev_a, *dev_b, *dev_c;
    // init the values
    for(int i = 0; i != LENGTH; ++i) {
        a[i] = 2 * i + i % 5;
        b[i] = 3 * i - i % 13;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i != LENGTH; ++i) {
        c_validation[i] = a[i] + b[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    auto start_mem_gpu = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMemset(dev_c, 0, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_a, a, LENGTH * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, LENGTH * sizeof(int), cudaMemcpyHostToDevice));
    auto end_mem_gpu = std::chrono::high_resolution_clock::now();
    auto start_calc_gpu = std::chrono::high_resolution_clock::now();
    // (LENGTH + threads_per_block) / threads_per_block, is (int)ceil((float)LENGTH / threads_per_block)
    add_vector_any_length<<<grid_length, block_length>>>(dev_a, dev_b, dev_c);
    HANDLE_ERROR(cudaDeviceSynchronize());
    auto end_calc_gpu = std::chrono::high_resolution_clock::now();
    auto start_mem2_gpu = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMemcpy(c, dev_c, LENGTH * sizeof(int), cudaMemcpyDeviceToHost));
    auto end_mem2_gpu = std::chrono::high_resolution_clock::now();

    // validate
    for(int i = 0; i != LENGTH; ++i) {
        if (c[i] != c_validation[i]) {
            is_valid = false;
            break;
        }
    }
    std::cout << "The result is " << (is_valid ? "valid" : "invalid") << std::endl;
    std::cout << "CPU time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count()
              << " us" << std::endl;
    std::cout << "GPU memory time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_mem_gpu - start_mem_gpu).count()
              << " us(mem1 copy) + "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_calc_gpu - start_calc_gpu).count()
              << " us(calc) + "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_mem2_gpu - start_mem2_gpu).count()
              << " us(mem2 copy) = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_mem2_gpu - start_mem_gpu).count()
              << " us" << std::endl;

    free(a); free(b); free(c); free(c_validation);
    HANDLE_ERROR(cudaFree(dev_a)); HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return is_valid ? 0 : -1;
}

int chapter5_block_threads() {
    const int threads_per_block = 256;
    bool is_valid = true;
    int *a = (int*)malloc(LENGTH * sizeof(int));
    int *b = (int*)malloc(LENGTH * sizeof(int));
    int *c_validation = (int*)malloc(LENGTH * sizeof(int));
    int *c = (int*)malloc(LENGTH * sizeof(int));
    int *dev_a, *dev_b, *dev_c;
    // init the values
    for(int i = 0; i != LENGTH; ++i) {
        a[i] = 2 * i + i % 5;
        b[i] = 3 * i - i % 13;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i != LENGTH; ++i) {
        c_validation[i] = a[i] + b[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    auto start_mem_gpu = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMemset(dev_c, 0, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_a, a, LENGTH * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, LENGTH * sizeof(int), cudaMemcpyHostToDevice));
    auto end_mem_gpu = std::chrono::high_resolution_clock::now();
    auto start_calc_gpu = std::chrono::high_resolution_clock::now();
    // (LENGTH + threads_per_block) / threads_per_block, is (int)ceil((float)LENGTH / threads_per_block)
    add_vector_block_threads
        <<<(LENGTH + threads_per_block) / threads_per_block, threads_per_block>>>
        (dev_a, dev_b, dev_c);
    HANDLE_ERROR(cudaDeviceSynchronize());
    auto end_calc_gpu = std::chrono::high_resolution_clock::now();
    auto start_mem2_gpu = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMemcpy(c, dev_c, LENGTH * sizeof(int), cudaMemcpyDeviceToHost));
    auto end_mem2_gpu = std::chrono::high_resolution_clock::now();

    // validate
    for(int i = 0; i != LENGTH; ++i) {
        if (c[i] != c_validation[i]) {
            is_valid = false;
            break;
        }
    }
    std::cout << "The result is " << (is_valid ? "valid" : "invalid") << std::endl;
    std::cout << "CPU time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count()
              << " us" << std::endl;
    std::cout << "GPU memory time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_mem_gpu - start_mem_gpu).count()
              << " us(mem1 copy) + "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_calc_gpu - start_calc_gpu).count()
              << " us(calc) + "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_mem2_gpu - start_mem2_gpu).count()
              << " us(mem2 copy) = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_mem2_gpu - start_mem_gpu).count()
              << " us" << std::endl;

    free(a); free(b); free(c); free(c_validation);
    HANDLE_ERROR(cudaFree(dev_a)); HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return is_valid ? 0 : -1;
}



/*
 * Chapter 5 of CUDA by Example
 * Vector addition with only one block and multiple threads
 * Typical result(Length <= 1024):
 *  The result is valid
 *  CPU time: 5 us
 *  GPU memory time: 144563 us(mem1 copy) + 257 us(calc) + 73 us(mem2 copy) = 144894 us
 */
int chapter5_only_threads() {
    if (LENGTH > max_thread) {
        std::cout << "Warning: the length of the vector is larger than the max threads per block.\n"
        << "The results might be invalid" << std::endl;
    }
    bool is_valid = true;
    int *a = (int*)malloc(LENGTH * sizeof(int));
    int *b = (int*)malloc(LENGTH * sizeof(int));
    int *c_validation = (int*)malloc(LENGTH * sizeof(int));
    int *c = (int*)malloc(LENGTH * sizeof(int));
    int *dev_a, *dev_b, *dev_c;
    // init the values
    for(int i = 0; i != LENGTH; ++i) {
        a[i] = 2 * i + i % 5;
        b[i] = 3 * i - i % 13;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i != LENGTH; ++i) {
        c_validation[i] = a[i] + b[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    auto start_mem_gpu = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMemset(dev_c, 0, LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_a, a, LENGTH * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, LENGTH * sizeof(int), cudaMemcpyHostToDevice));
    auto end_mem_gpu = std::chrono::high_resolution_clock::now();
    auto start_calc_gpu = std::chrono::high_resolution_clock::now();
    add_vector_only_threads<<<1, LENGTH>>>(dev_a, dev_b, dev_c);
    HANDLE_ERROR(cudaDeviceSynchronize());
    auto end_calc_gpu = std::chrono::high_resolution_clock::now();
    auto start_mem2_gpu = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMemcpy(c, dev_c, LENGTH * sizeof(int), cudaMemcpyDeviceToHost));
    auto end_mem2_gpu = std::chrono::high_resolution_clock::now();

    // validate
    for(int i = 0; i != LENGTH; ++i) {
        if (c[i] != c_validation[i]) {
            is_valid = false;
            break;
        }
    }
    std::cout << "The result is " << (is_valid ? "valid" : "invalid") << std::endl;
    std::cout << "CPU time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count()
        << " us" << std::endl;
    std::cout << "GPU memory time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_mem_gpu - start_mem_gpu).count()
        << " us(mem1 copy) + "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_calc_gpu - start_calc_gpu).count()
        << " us(calc) + "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_mem2_gpu - start_mem2_gpu).count()
        << " us(mem2 copy) = "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_mem2_gpu - start_mem_gpu).count()
        << " us" << std::endl;

    free(a); free(b); free(c); free(c_validation);
    HANDLE_ERROR(cudaFree(dev_a)); HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return is_valid ? 0 : -1;
}