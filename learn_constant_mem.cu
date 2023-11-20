/*
 * This program is about rendering some colorful spheres using raytracing, while comparing whether the use of
 * __constant__ memory will speed up the process.
 */

#include <cmath>
#include <cstdio>
#include "random"
#include "cuda.h"
#include "includes/check_error.cuh"
#include "includes/cpu_bitmap.h"

#define INF 2e10f
#define DIM 1024
#define SPHERE_NUM 20

float RND(float range) {
    static std::random_device rd;
    static std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.0f, range);
    return dist(mt);
}


struct Sphere {
    float r,g,b;
    float radius;
    float x,y,z;
    // (ox, oy) is pixel on the screen, light ray will be casted from (ox, oy, 0), direction is (0, 0, 1)
    __device__ float hit(float ox, float oy, float *edge_coefficient) const {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            // dz / radius will be the cosine of the angle between the normal vector and the light ray
            // which indicates the light intensity
            *edge_coefficient = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

__global__ void ray_tracing_no_constant_mem(unsigned char *bitmap, Sphere *d_spheres) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int offset = x + y * gridDim.x * blockDim.x;

    auto ox = (float)x - (float)DIM/2;
    auto oy = (float)y - (float)DIM/2;

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i != SPHERE_NUM; ++i) {
        float n;
        float t = d_spheres[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fsacle = n;
            r = d_spheres[i].r * fsacle;
            g = d_spheres[i].g * fsacle;
            b = d_spheres[i].b * fsacle;
        }
    }

    bitmap[offset*4 + 0] = (int)(r * 255);
    bitmap[offset*4 + 1] = (int)(g * 255);
    bitmap[offset*4 + 2] = (int)(b * 255);
    bitmap[offset*4 + 3] = 255;
}

float no_constant_mem() {
    cudaEvent_t start, stop;
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start, 0));

    CPUBitmap bitmap{DIM, DIM};
    unsigned char* d_bitmap; Sphere *d_spheres;
    auto *h_spheres = new Sphere[SPHERE_NUM];

    for (int i = 0; i != SPHERE_NUM; ++i) {
        h_spheres[i].r = RND(1.0f);
        h_spheres[i].g = RND(1.0f);
        h_spheres[i].b = RND(1.0f);
        h_spheres[i].x = RND(DIM) - DIM/2;
        h_spheres[i].y = RND(DIM) - DIM/2;
        h_spheres[i].z = RND(DIM) - DIM/2;
        h_spheres[i].radius = RND(500.0f) + 80.0f;
    }

    CHECK_ERROR(cudaMalloc((void**)&d_bitmap, bitmap.image_size()));
    CHECK_ERROR(cudaMalloc((void**)&d_spheres, sizeof(Sphere) * SPHERE_NUM));
    CHECK_ERROR(cudaMemcpy(d_spheres, h_spheres, sizeof(Sphere) * SPHERE_NUM,
                           cudaMemcpyHostToDevice));
    delete[] h_spheres;
    dim3 grid(DIM/16, DIM/16);
    dim3 block(16, 16);
    // kernel
    ray_tracing_no_constant_mem<<<grid, block>>>(d_bitmap, d_spheres);
    CHECK_ERROR(cudaMemcpy(bitmap.get_ptr(), d_bitmap, bitmap.image_size(),
                           cudaMemcpyDeviceToHost));
//    h_bitmap.display_and_exit();
    bitmap.save_as_bmp("./no_constant_mem.bmp");
    cudaFree(d_bitmap);
    cudaFree(d_spheres);
    CHECK_ERROR(cudaEventRecord(stop, 0));
    CHECK_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate without constant mem: %3.1f ms\n", elapsedTime);
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}

__constant__ Sphere d_spheres[SPHERE_NUM];

__global__ void ray_tracing(unsigned char *bitmap) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int offset = x + y * gridDim.x * blockDim.x;

    auto ox = (float)x - (float)DIM/2;
    auto oy = (float)y - (float)DIM/2;

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (auto & d_sphere : d_spheres) {
        float n;
        float t = d_sphere.hit(ox, oy, &n);
        if (t > maxz) {
            float fsacle = n;
            r = d_sphere.r * fsacle;
            g = d_sphere.g * fsacle;
            b = d_sphere.b * fsacle;
        }
    }

    bitmap[offset*4 + 0] = (int)(r * 255);
    bitmap[offset*4 + 1] = (int)(g * 255);
    bitmap[offset*4 + 2] = (int)(b * 255);
    bitmap[offset*4 + 3] = 255;
}

float constant_mem() {
    cudaEvent_t start, stop;
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start, 0));

    CPUBitmap bitmap{DIM, DIM};
    unsigned char* d_bitmap;
    auto *h_spheres = new Sphere[SPHERE_NUM];

    for (int i = 0; i != SPHERE_NUM; ++i) {
        h_spheres[i].r = RND(1.0f);
        h_spheres[i].g = RND(1.0f);
        h_spheres[i].b = RND(1.0f);
        h_spheres[i].x = RND(DIM) - DIM/2;
        h_spheres[i].y = RND(DIM) - DIM/2;
        h_spheres[i].z = RND(DIM) - DIM/2;
        h_spheres[i].radius = RND(500.0f) + 80.0f;
    }

    CHECK_ERROR(cudaMalloc((void**)&d_bitmap, bitmap.image_size()));
    // no need to allocate memory for __constant__ d_spheres
//    CHECK_ERROR(cudaMalloc((void**)&d_spheres, sizeof(Sphere) * SPHERE_NUM));
    // copy data to __constant__ d_spheres, note the new syntax
    CHECK_ERROR(cudaMemcpyToSymbol(d_spheres, h_spheres, sizeof(Sphere) * SPHERE_NUM,
                           0, cudaMemcpyHostToDevice));
    delete[] h_spheres;
    dim3 grid(DIM/16, DIM/16);
    dim3 block(16, 16);
    // kernel
    ray_tracing<<<grid, block>>>(d_bitmap);
    CHECK_ERROR(cudaMemcpy(bitmap.get_ptr(), d_bitmap, bitmap.image_size(),
                           cudaMemcpyDeviceToHost));
//    h_bitmap.display_and_exit();
    bitmap.save_as_bmp("constant_mem.bmp");
    cudaFree(d_bitmap);
//    cudaFree(d_spheres);
    CHECK_ERROR(cudaEventRecord(stop, 0));
    CHECK_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate with constant mem: %3.1f ms\n", elapsedTime);
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}



/*
 * Conclusion:
 * Why we use constant memory? because although constant memory is off-chip, when we use constant memory, the constant
 * will be broadcasted to half warp(16 threads) in the same SM, meaning half of the threads in the same SM will access
 * the same constant memory with almost no latency. While that already saved about 94% of the time, the GPU will also
 * actively cache the constant memory in the L1 cache, which will further reduce the latency.
 *
 *
 * But when the DIM and SPHERE_NUM is very large, the constant memory will be too small to hold all the data,
 * so we have to use global memory, but the broadcasting will still be used! That will be a serial process, which means
 * the threads in the same half warp will access the global memory one by one, but if we just ignore the constant memory,
 * the threads will access the global memory in parallel. So we can see performance drop when DIM and SPHERE_NUM is
 * very large.
 *
 */
int main() {
    int run_times = 1;
    float no_constant_mem_time = 0.0f;
    float constant_mem_time = 0.0f;
    for (int i = 0; i != run_times; ++i) {
        no_constant_mem_time += no_constant_mem();
        constant_mem_time += constant_mem();
    }
    /*
     * Typical result(DIM=1024):
     * Average time to generate without constant mem: 6.1 ms
     * Average time to generate with constant mem: 5.5 ms
     *
     * Typical result(DIM=10240):
     * Average time to generate without constant mem: 1675.9 ms
     * Average time to generate with constant mem: 1966.0 ms
     */
    printf("Average time to generate without constant mem: %3.1f ms\n", no_constant_mem_time / (float)run_times);
    printf("Average time to generate with constant mem: %3.1f ms\n", constant_mem_time / (float)run_times);
    return 0;
}