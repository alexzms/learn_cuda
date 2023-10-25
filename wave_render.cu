#include <iostream>
#include "includes/book.h"
#include "includes/cpu_anim.h"
#include "includes/cpu_bitmap.h"
#include <chrono>

#define DIM 1024

struct DataBlock {
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

void clean_up(DataBlock *d) {
    cudaFree(d->dev_bitmap);
}
/*
 * blockDim就是block中threads的形状，例如如果一个block有4x4=16个threads，那么blockDim=(4,4)
 * 同理可得gridDim
 */
__global__ void calculate_frame(unsigned char *bitmap, const int ticks) {
    // 获取到了这个thread对应原图的精确坐标了
    unsigned int thread_offset_x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int thread_offset_y = threadIdx.y + blockIdx.y * blockDim.y;
    // 计算坐标(x,y)对应到1维数组的index
    unsigned int offset = thread_offset_y * blockDim.x * gridDim.x + thread_offset_x;

    // 以屏幕中心为原点，进行数值计算
    float fx = (float)thread_offset_x - (float)DIM/2;
    float fy = (float)thread_offset_y - (float)DIM/2;
    float dist = sqrt((fx - 200) * (fx - 200) + (fy) * (fy));
    float dist2 = sqrt((fx + 200) * (fx + 200) + (fy) * (fy));
    float dist3 = sqrt((fx) * (fx) + (fy - 170) * (fy - 170));
    float dist4 = sqrt((fx) * (fx) + (fy + 170) * (fy + 170));
    // 计算灰度值，其中的数学表达式在这个语境下没有什么值得研究的意义，和CUDA没什么关系
    auto grey = (unsigned char)(
            255 * (
                    cos(dist / 10.0f - (float)ticks / 7.0f)
                    +
                    cos(dist2 / 10.0f - (float)ticks / 7.0f)
                    +
                    cos(dist3 / 10.0f - (float)ticks / 7.0f)
                    +
                    cos(dist4 / 10.0f - (float)ticks / 7.0f)
            ) / 4);
    bitmap[offset * 4 + 0] = grey;
    bitmap[offset * 4 + 1] = grey;
    bitmap[offset * 4 + 2] = grey;
    bitmap[offset * 4 + 3] = 255;
}

/*
 * Using CPU to calculate this can only maintain around 5fps
 * While using CUDA, GTX4060, it can be 100+ times better
 */
__host__ void calculate_frame_cpu(unsigned char *bitmap, const int ticks, const unsigned int thread_offset_x,
                                  const unsigned int thread_offset_y) {
    unsigned int offset = thread_offset_y * DIM + thread_offset_x;
    float fx = (float)thread_offset_x - (float)DIM/2;
    float fy = (float)thread_offset_y - (float)DIM/2;
    float dist = sqrt((fx - 200) * (fx - 200) + (fy) * (fy));
    float dist2 = sqrt((fx + 200) * (fx + 200) + (fy) * (fy));
    float dist3 = sqrt((fx) * (fx) + (fy - 170) * (fy - 170));
    float dist4 = sqrt((fx) * (fx) + (fy + 170) * (fy + 170));
    // 计算灰度值，其中的数学表达式在这个语境下没有什么值得研究的意义，和CUDA没什么关系
    auto grey = (unsigned char)(
                        255 * (
                        cos(dist / 10.0f - (float)ticks / 7.0f)
                        +
                        cos(dist2 / 10.0f - (float)ticks / 7.0f)
                        +
                        cos(dist3 / 10.0f - (float)ticks / 7.0f)
                        +
                        cos(dist4 / 10.0f - (float)ticks / 7.0f)
                ) / 4);
    bitmap[offset * 4 + 0] = grey;
    bitmap[offset * 4 + 1] = grey;
    bitmap[offset * 4 + 2] = grey;
    bitmap[offset * 4 + 3] = 255;
}

void generate_frame(DataBlock *d, const int ticks) {
    dim3 grid(DIM/16, DIM/16);
    dim3 threads(16, 16);
    calculate_frame<<<grid, threads>>>(d->dev_bitmap, ticks);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
                            d->bitmap->image_size(), cudaMemcpyDeviceToHost));
//    for (int i = 0; i != DIM; ++i) {
//        for (int j = 0; j != DIM; ++j) {
//            calculate_frame_cpu(d->bitmap->get_ptr(), ticks, j, i);
//        }
//    }
}

int main() {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, data.bitmap->image_size()));

    data.bitmap->anim_and_exit((void (*)(void*, int)) generate_frame, (void (*)(void*)) clean_up);
}
