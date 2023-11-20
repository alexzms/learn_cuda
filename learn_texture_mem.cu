/*
 * This program should've use texture memory to speed up the data fetching, but texture reference is deprecated!!!
 * No more work on this... ABANDONED
 */

#include <cuda.h>
#include "cuda_texture_types.h"
#include "iostream"
#include "random"
#include <cstdio>
#include "./includes/book.h"
#include "./includes/cpu_anim.h"
#include "check_error.cuh"
#include <cstdio>

#define DIM 1024
#define FLOW_SPEED 0.25f
#define ANIMATION_SPEED 90
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f


struct DataBlock {
    unsigned char *d_output_bitmap;
    float *d_iptr;
    float *d_optr;
    float *d_sptr;
    CPUAnimBitmap *h_bitmap;

    cudaEvent_t start, stop;
    float total_time;
    unsigned int frames_count;
};

__global__ void maintain_source_constant_temperature_kernel(float *iptr, const float *sptr) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    unsigned int offset = x + y * gridDim.x * blockDim.x;

    // if sptr[i] is not zero, set iptr[i] to sptr[i] to maintain the source temperature
    // we can compare to 0 safely because value of sptr will be written as 0, this won't cause problem
    if (sptr[offset] != 0) {
        iptr[offset] = sptr[offset];
    }
}

// instead of directly update the iptr, we will output result to optr
// because if we do it on iptr, the result value will depend on the order of execution of code, which is not ideal
__global__ void heat_flow_kernel(float *optr, const float *iptr) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    unsigned int offset = x + y * gridDim.x * blockDim.x;

    unsigned int left_offset = offset - 1;
    unsigned int right_offset = offset + 1;
    // prevent overflow, the reason we still keep this is because later, that term will cancel out
    // like when x == 0, left == 0, ptr[left] = ptr[x], so T_new = ... T_left + ... - 4T_old, they cancel out
    if (x == 0) {
        left_offset += 1;
    } else if (x == DIM - 1) {
        right_offset -= 1;
    }
    unsigned int up_offset = offset - DIM;
    unsigned int down_offset = offset + DIM;
    if (y == 0) {
        up_offset += DIM;
    } else if (y == DIM - 1) {
        down_offset -= DIM;
    }
    optr[offset] = iptr[offset]
         + FLOW_SPEED * (iptr[left_offset]+iptr[right_offset]+iptr[up_offset]+iptr[down_offset] - 4 * iptr[offset]);
}

__global__ void float_to_color_kernel(unsigned char *obitmap, const float *iptr) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    unsigned int offset = x + y * gridDim.x * blockDim.x;

    float value = iptr[offset];
    if (value > MAX_TEMP) {
        value = MAX_TEMP;
    }
    if (value < MIN_TEMP) {
        value = MIN_TEMP;
    }
    // when it's MAX_TEMP, it's red, when it's MIN_TEMP, it's blue, we will do a linear interpolation
    // to get the color value
    float color_value = (value - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
    obitmap[offset * 4 + 0] = (unsigned char)(255 * color_value);
    obitmap[offset * 4 + 1] = 0;
    obitmap[offset * 4 + 2] = (unsigned char)(255 * (1 - color_value));
    obitmap[offset * 4 + 3] = 255;
}

void anim_gpu(DataBlock *data_block, int tick) {
    CHECK_ERROR(cudaEventRecord(data_block->start, nullptr));
    dim3 grid(DIM/16, DIM/16);
    dim3 block(16, 16);
    CPUAnimBitmap *h_bitmap = data_block->h_bitmap;
    for (int i = 0; i != ANIMATION_SPEED; ++i) {
        maintain_source_constant_temperature_kernel<<<grid, block>>>
                                                        (data_block->d_iptr, data_block->d_sptr);
        heat_flow_kernel<<<grid, block>>>(data_block->d_optr, data_block->d_iptr);
        // swap
//        float *d_temp = data_block->d_iptr;
//        data_block->d_iptr = data_block->d_optr;
//        data_block->d_optr = d_temp;
        std::swap(data_block->d_iptr, data_block->d_optr);
    }
    float_to_color<<<grid, block>>>
                                                (data_block->d_output_bitmap, data_block->d_iptr);
    CHECK_ERROR(cudaMemcpy(h_bitmap->get_ptr(), data_block->d_output_bitmap,
                           h_bitmap->image_size(), cudaMemcpyDeviceToHost));
    float elapsed_time = 0.0f;
    CHECK_ERROR(cudaEventRecord(data_block->stop, nullptr));
    CHECK_ERROR(cudaEventSynchronize(data_block->stop));
    CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, data_block->start, data_block->stop));
    data_block->total_time += elapsed_time;
    data_block->frames_count += 1;
    printf("[INFO] Tick: %u, FPS(current): %3.1f, FPS(average): %3.1f\n",
           tick, 1000.0f / elapsed_time, 1000.0f / (data_block->total_time / data_block->frames_count));
}

void anim_exit_gpu(DataBlock *data_block) {
    CHECK_ERROR(cudaFree(data_block->d_iptr));
    CHECK_ERROR(cudaFree(data_block->d_optr));
    CHECK_ERROR(cudaFree(data_block->d_sptr));
    CHECK_ERROR(cudaEventDestroy(data_block->start));
    CHECK_ERROR(cudaEventDestroy(data_block->stop));
}

int main() {
    DataBlock data_block{};
    CPUAnimBitmap bitmap(DIM, DIM, &data_block);
    data_block.h_bitmap = &bitmap;
    data_block.total_time = 0.0f;
    data_block.frames_count = 0;

    CHECK_ERROR(cudaEventCreate(&data_block.start));
    CHECK_ERROR(cudaEventCreate(&data_block.stop));
    CHECK_ERROR(cudaMalloc((void**)&data_block.d_output_bitmap, bitmap.image_size()));
    unsigned int float_bitmap_size = DIM * DIM * sizeof(float);
    CHECK_ERROR(cudaMalloc((void**)&data_block.d_iptr, float_bitmap_size));
    CHECK_ERROR(cudaMalloc((void**)&data_block.d_optr, float_bitmap_size));
    CHECK_ERROR(cudaMalloc((void**)&data_block.d_sptr, float_bitmap_size));

    auto *h_sptr = new float[DIM * DIM];
    for (int i = 0; i != DIM * DIM; ++i) {
        h_sptr[i] = 0.0f;
        unsigned int x = i % DIM;
        unsigned int y = i / DIM;
        if ((x > 300) && (x < DIM-300) && (y > 310) && (y < 601)) {
            h_sptr[i] = MAX_TEMP;
        }
    }
    h_sptr[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    h_sptr[DIM * 700 + 100] = MIN_TEMP;
    h_sptr[DIM * 300 + 300] = MIN_TEMP;
    h_sptr[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y != 900; ++y) {
        for (int x = 400; x != DIM-400; ++x) {
            h_sptr[x + y * DIM] = MIN_TEMP;
        }
    }

    for (int y = 100; y != 200; ++y) {
        for (int x = 400; x != DIM-400; ++x) {
            h_sptr[x + y * DIM] = MIN_TEMP;
        }
    }
    CHECK_ERROR(cudaMemcpy(data_block.d_sptr, h_sptr, float_bitmap_size, cudaMemcpyHostToDevice));
    for (int y = 800; y != DIM - 100; ++y) {
        for (int x = 100; x != 200; ++x) {
            h_sptr[x + y * DIM] = MAX_TEMP;
        }
    }
    CHECK_ERROR(cudaMemcpy(data_block.d_iptr, h_sptr, float_bitmap_size, cudaMemcpyHostToDevice));
    delete[] h_sptr;

    bitmap.anim_and_exit((void (*)(void *, int))anim_gpu, (void (*)(void *))anim_exit_gpu);
    CHECK_ERROR(cudaFree(data_block.d_output_bitmap));
    CHECK_ERROR(cudaFree(data_block.d_iptr));
    CHECK_ERROR(cudaFree(data_block.d_optr));
    CHECK_ERROR(cudaFree(data_block.d_sptr));
    return 0;
}