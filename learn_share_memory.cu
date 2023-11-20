/*
 * This program is about computing dot_product in the way using __shared__ memory to sum up the results in a single
 * block, and this can speed up the summing process by roughly 10x times
 * Also, this program renders some green balls using __shared__ memory
 */

#include <cmath>
#include <cstdio>
#include "includes/check_error.cuh"
#include "cpu_bitmap.h"

#define M_PI 3.14159265358979323846

const unsigned int N = 1024;
// these two value is only used for dot_product
const unsigned int threads_per_block = 256;
const unsigned int blocks_per_grid = ((N - 1) / threads_per_block) + 1;

__global__ void dot_product(const float *a,const float *b, float *c) {
    __shared__ float cache[threads_per_block];
    unsigned int thread_id = threadIdx.x;
    unsigned int tid = thread_id + blockIdx.x * blockDim.x;
    float temp = 0.0f;
    while (tid < N) {
        temp += a[tid] * b[tid];
        // move forward by a computation area
        tid += gridDim.x * blockDim.x;
    }
    cache[thread_id] = temp;
    // this is to ensure all threads has finished writing in cache
    __syncthreads();
    unsigned int reduction_step = blockDim.x / 2;
    while (reduction_step != 0) {
        if (thread_id < reduction_step) {
            cache[thread_id] += cache[thread_id + reduction_step];
        }
        // this is to prevent the addition by reduction is not sync finished
        __syncthreads();
        reduction_step /= 2;
    }
    // just let the first thread to write to the final result(which is the sum of the block)
    if (thread_id == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void compute_bitmap(unsigned char *bitmap) {
    __shared__ float cache[16][16];
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int offset = y * gridDim.x * blockDim.x + x;
    const float period = 128.0f;
    cache[threadIdx.y][threadIdx.x] = 255 * (sinf((float)((float)x * 2.0f * M_PI / period)) + 1.0f) *
                                       (sinf((float)((float)y * 2.0f * M_PI / period)) + 1.0f) / 4.0f;
    // if we don't add this line, the result will be wrong
    __syncthreads();
    bitmap[offset * 4 + 0] = 0;
    bitmap[offset * 4 + 1] = (unsigned char)cache[threadIdx.y][threadIdx.x];
    bitmap[offset * 4 + 2] = 0;
    bitmap[offset * 4 + 3] = 255;
}

bool check_correct(float *cal, float *ref) {
    for (int i = 0; i != blocks_per_grid; ++i) {
        if (fabs(cal[i] - ref[i]) > 1e-5 * ((float)N / blocks_per_grid)) {
            printf("Error: cal[%d] = %f, ref[%d] = %f\n", i, cal[i], i, ref[i]);
            printf("The error is %f\n", fabs(cal[i] - ref[i]));
            printf("The estimate error per thread is %f\n",
                   fabs(cal[i] - ref[i]) / ((float)N / blocks_per_grid));
            return false;
        }
    }
    return true;
}

int correct_dot_product() {
    float *h_a, *h_b, *h_partial_c, *ref_partial_c;
    float *d_a, *d_b, *d_partial_c;
    h_a = new float[N];
    h_b = new float[N];
    h_partial_c = new float[blocks_per_grid];
    ref_partial_c = new float[blocks_per_grid];
    for (int i = 0; i != N; ++i) {
        h_a[i] = (float)(i % 3) * 4.5f;
        h_b[i] = (float)(i % 2) * 0.6f;
    }
    for (int j = 0; j != blocks_per_grid; ++j) {
        // calculate the reference result
        float temp = 0.0f;
        for (int i = 0; i != threads_per_block; ++i) {
            temp += h_a[i + j * threads_per_block] * h_b[i + j * threads_per_block];
        }
        ref_partial_c[j] = temp;
    }
    CHECK_ERROR(cudaMalloc((void**)&d_a, sizeof(float) * N));
    CHECK_ERROR(cudaMalloc((void**)&d_b, sizeof(float) * N));
    CHECK_ERROR(cudaMalloc((void**)&d_partial_c, sizeof(float) * blocks_per_grid));
    CHECK_ERROR(cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice));

    dot_product<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_partial_c);
    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());
    CHECK_ERROR(cudaMemcpy(h_partial_c, d_partial_c, sizeof(float) * blocks_per_grid, cudaMemcpyDeviceToHost));

    bool correct = check_correct(h_partial_c, ref_partial_c);
    if (!correct) {
        printf("The result is not correct!\n");
        printf("To be mentioned, the precision of gpu calculating float is not as good as cpu\n");
        printf("If a[i] or b[i] is too big(based on experiment, usually >10 is too big)"
               "the result will be wrong\n");
    }
    float sum = 0.0f;
    for (int i = 0; i != blocks_per_grid; ++i) {
        sum += h_partial_c[i];
    }
    printf("The dot product of the two vectors is %f\n", sum);
    CHECK_ERROR(cudaFree(d_a));
    CHECK_ERROR(cudaFree(d_b));
    CHECK_ERROR(cudaFree(d_partial_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_partial_c;
    delete[] ref_partial_c;
    return 0;
}

int sin_bitmap() {
    CPUBitmap h_bitmap(N, N);
    unsigned char *d_bitmap;
    CHECK_ERROR(cudaMalloc((void**)&d_bitmap, sizeof(unsigned char) * N * N * 4));

    dim3 grid(N / 16, N / 16);
    dim3 block(16, 16);
    compute_bitmap<<<grid, block>>>(d_bitmap);
    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());
    CHECK_ERROR(cudaMemcpy(h_bitmap.get_ptr(), d_bitmap, sizeof(unsigned char) * N * N * 4, cudaMemcpyDeviceToHost));
    h_bitmap.display_and_exit();

    return 0;
}

int main() {
//    sin_bitmap();
    correct_dot_product();
    return 0;
}