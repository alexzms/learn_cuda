#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include <iostream>
#include "includes/book.h"
#include <chrono>
#include "Shader.h"
#include <cuda_gl_interop.h>

int chapter1();
int chapter2_vector_addition();
int chapter2_julia_set();
float scale = 1.5f; float shift_x = 0.0f; float shift_y = 0.0f; float delta_time = 0.0f;
bool first_mouse = true; float last_x; float last_y; float last_frame = 0.0f; bool need_redraw = true;

const int DIM = 2048;

void process_input(GLFWwindow* window);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void scroll_callback(GLFWwindow* window, double unknown_offset, double scroll_offset);
void mouse_callback(GLFWwindow* window, double width_pos, double height_pos);

struct complex_number {
    float r;
    float i;
    // supports both host init and device init
    __host__ __device__ complex_number(float r_, float i_): r(r_), i(i_) {}
    // [[nodiscard]] means that if the return value is not used, there will be a compiler error
    [[nodiscard]]__device__ float magnitude() const { return r * r + i * i; }
    __device__ complex_number operator*(const complex_number &rhs) const {
        return {r * rhs.r - i * rhs.i, r * rhs.i + i * rhs.r};
    }
    __device__ complex_number operator+(const complex_number &rhs) const {
        return {r + rhs.r, i + rhs.i};
    }
};

complex_number C(-0.8f, 0.156f);

__device__ float julia(const int x, const int y, const int DIM, const complex_number C,
                     float scale_dev, float shift_dev_x, float shift_dev_y) {
    // we want it to scale at the center of (shift_dev_x, shift_dev_y)
    float jx = scale_dev * (float)(DIM / 2 - x + shift_dev_x / scale_dev);
    float jy = scale_dev * (float)(DIM / 2 - y + shift_dev_y / scale_dev);

    jx /= DIM / 2;
    jy /= DIM / 2;

    complex_number Z(jx, jy);

    // max iter = 200 (hardcoded)
    for (int i = 0; i != 200; ++i) {
        Z = Z * Z + C;
        if (Z.magnitude() > 4.0f) {
            return (float)i;
        }
    }
    return 0;
}

__global__ void add_integer(int a, int b, int* c) {
    *c = a + b;
}

__global__ void add_vector(const int* a, const int* b, int* c, const int length) {
    int tid = blockIdx.x;
    if (tid < length) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void compute_julia(unsigned char* ptr, complex_number C,
                              float scale_dev, float shift_dev_x, float shift_dev_y) {
    int i = blockIdx.x;
    int j = blockIdx.y;

    int offset = i + j * gridDim.x;
    int julia_value = julia(i, j, gridDim.x, C, scale_dev, shift_dev_x, shift_dev_y);
    if (julia_value > 50) {
        julia_value = 50;
    }
    // 3, 4, 5: silver
    ptr[offset * 4 + 0] = 3 * julia_value;
    ptr[offset * 4 + 1] = 4 * julia_value;
    ptr[offset * 4 + 2] = 5 * julia_value;
    ptr[offset * 4 + 3] = 255 * julia_value > 0 ? 255 : 0;
}

__global__ void compute_julia_interop(uchar4* ptr, complex_number C, float scale_dev, float shift_dev_x, float shift_dev_y) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + y * gridDim.x * blockDim.x;

    float julia_value = julia(x, y, gridDim.x * blockDim.x, C, scale_dev, shift_dev_x, shift_dev_y);

//    if (julia_value > 50.0f) {
//        julia_value = 50.0f;
//    }
    // 3, 4, 5: silver
    ptr[offset] = make_uchar4(3.01 * julia_value, 3.07 * julia_value, 3.10 * julia_value,
                              255 * (julia_value > 0 ? 1 : 0));
}

int main() {
    chapter2_julia_set();
    return 0;
}

int chapter2_julia_set() {
    const int image_size = DIM * DIM * 4;
    auto* bitmap_sysmem = (unsigned char*)malloc(image_size);
//    unsigned char* dev_map;
    // zero-copy memory
//    HANDLE_ERROR(cudaHostAlloc((void**)&zero_copy_map, image_size, cudaHostAllocMapped));
//    HANDLE_ERROR(cudaHostGetDevicePointer((void**)&dev_map, zero_copy_map, 0));

    // WOW: use a glfw to visualize the result, it's so cool!!!!!!!!!!!!!!!!!!!!!!
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    GLFWmonitor* glfWmonitor = glfwGetPrimaryMonitor();
    if (glfWmonitor == nullptr) {
        std::cout << "Failed to get primary monitor.." << std::endl;
        glfwTerminate();
        return -1;
    }
    const GLFWvidmode *glfWvidmode = glfwGetVideoMode(glfWmonitor);
    unsigned int RENDER_WIDTH = glfWvidmode->width;
    unsigned int RENDER_HEIGHT = glfWvidmode->height;

    GLFWwindow* window = glfwCreateWindow(RENDER_WIDTH, RENDER_HEIGHT,
                                          "Colors", glfWmonitor, nullptr);
    if (window == nullptr) {
        std::cout << "Failed to create glf windows.." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cout << "Failed to init GLAD loader.." << std::endl;
        glfwTerminate();
        return -1;
    }
    glViewport(0, 0, RENDER_WIDTH, RENDER_HEIGHT);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

    // a rectangle in front of the camera
    float vertices[] = {
            // positions          // texture coords
            1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // top right
            1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // bottom right
            -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // bottom left
            -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // top left
    };

    unsigned int indices[] = {
            0, 1, 3, // first triangle
            1, 2, 3  // second triangle
    };

    Shader shader("./shaders/display_texture.vert", "./shaders/display_texture.frag");
    shader.use();
    unsigned int VBO, VAO, EBO;
    // generate vertex array object
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    // generate vertex buffer object
    glGenBuffers(1, &VBO);
    // generate element buffer object
    glGenBuffers(1, &EBO);
    // bind the vertex array object
    glBindVertexArray(VAO);
    // copy vertices array into vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // copy indices array into element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    // configure vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          (void*) nullptr);
    glEnableVertexAttribArray(0); // pos
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          (void*) (3 * sizeof(float)));
    glEnableVertexAttribArray(1); // texture coords

    // texture
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // set the texture wrapping/filtering options (on the currently bound texture object)
    // set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // use zero-copy memory as the texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, DIM, DIM, 0, GL_RGBA, GL_UNSIGNED_BYTE, bitmap_sysmem);

    // 注册 OpenGL 缓冲区到 CUDA 图形资源
    GLuint bufferObj;
    cudaGraphicsResource *resource;
    glGenBuffers(1, &bufferObj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferObj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, DIM * DIM * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

    uchar4* devPtr;
    size_t  size;
    // 映射资源
    HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

    dim3 blocks(DIM/32, DIM/32);
    dim3 threads(32,32);
    last_frame = (float)glfwGetTime();
    // render loop
    while (!glfwWindowShouldClose(window)) {
        auto current_frame = (float)glfwGetTime();
        delta_time = current_frame - last_frame;
        last_frame = current_frame;
//        std::cout << "FPS: " << 1.0f / delta_time << " Estimated PCIE bandwidth: "
//                  << 3.0f * (float)image_size / delta_time / 1024 / 1024 / 1024 << " GB/s" << std::endl;
        // input
        process_input(window);
        // render
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // state-setting function
        glClear(GL_COLOR_BUFFER_BIT); // state-using function

        // memset the zero-copy memory
        compute_julia_interop<<<blocks, threads>>>(devPtr, C, scale, shift_x, shift_y);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // draw the rectangle
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferObj);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        shader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));
    HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
    // glfw: terminate, clearing all previously allocated GLFW resources.
    free(bitmap_sysmem);
    glfwTerminate();
    return 0;
}

/*
 * This chapter is also about the basic usage of cuda, we parallelized the vector addition.
 * And observed the performance.
 */
int chapter2_vector_addition() {
    const unsigned int ARRAY_LENGTH = 65535;
    bool identical = true;
    int *a = (int*)malloc(ARRAY_LENGTH * sizeof(int));
    int *b = (int*)malloc(ARRAY_LENGTH * sizeof(int));
    int *c = (int*)malloc(ARRAY_LENGTH * sizeof(int));
    int *cpu_temp_c = (int*)malloc(ARRAY_LENGTH * sizeof(int));
    int *dev_a, *dev_b, *dev_c;
    // first we assign values to a[] and b[]
    for (int i = 0; i != ARRAY_LENGTH; ++i) {
        a[i] = i;
        b[i] = -2 * i + i % 5;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i != ARRAY_LENGTH; ++i) {
        cpu_temp_c[i] = a[i] + b[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();


    auto start_host_to_device = std::chrono::high_resolution_clock::now();
    // allocate on gpu
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, ARRAY_LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, ARRAY_LENGTH * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, ARRAY_LENGTH * sizeof(int)));
    // copy to gpu
    HANDLE_ERROR(cudaMemcpy(dev_a, a, ARRAY_LENGTH * sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, ARRAY_LENGTH * sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_c, c, ARRAY_LENGTH * sizeof(int),
                            cudaMemcpyHostToDevice));
    auto end_host_to_device = std::chrono::high_resolution_clock::now();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    add_vector<<<ARRAY_LENGTH, 1>>>
            (dev_a, dev_b, dev_c, ARRAY_LENGTH);
    HANDLE_ERROR(cudaDeviceSynchronize());
    auto end_gpu = std::chrono::high_resolution_clock::now();

    auto start_device_to_host = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMemcpy(c, dev_c, ARRAY_LENGTH * sizeof(int),
                            cudaMemcpyDeviceToHost));
    auto end_device_to_host = std::chrono::high_resolution_clock::now();



    // check if the result is correct
    for (int i = 0; i != ARRAY_LENGTH; ++i) {
        if (c[i] != cpu_temp_c[i]) {
            identical = false;
            break;
        }
    }
    if (identical) {
        std::cout << "The result is correct" << std::endl;
    } else {
        std::cout << "The result is incorrect" << std::endl;
    }

    std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::microseconds>
            (end_cpu - start_cpu).count() << " us" << std::endl;
    std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::microseconds>
            (end_gpu - start_gpu).count() << " us" << "+"
              << std::chrono::duration_cast<std::chrono::microseconds>
                      (end_host_to_device - start_host_to_device).count() << " us(host->device memcpy)"
              << "+" << std::chrono::duration_cast<std::chrono::microseconds>
                      (end_device_to_host - start_device_to_host).count() << " us(device->host memcpy)"
              << std::endl;
    std::cout << "Hint: the copy time is really large, and gpu time is also not fascinating."
                 "We need to optimize this deeper." << std::endl;

    // free memory
    free(a);
    free(b);
    free(c);
    free(cpu_temp_c);
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    return 0;
}

/*
 * This chapter is about the basic usage of cuda, including:
 * 1. cudaGetDeviceCount
 * 2. cudaGetDeviceProperties
 * 3. cudaMalloc
 * 4. cudaFree
 * 5. cudaMemcpy
 * 6. kernel function
 */
int chapter1() {
    int device_count;
    HANDLE_ERROR(cudaGetDeviceCount(&device_count));
    std::cout << "device count: " << device_count << std::endl;
    for (int i = 0; i != device_count; ++i) {
        std::cout << "Fetching device properties for device " << i << std::endl;
        cudaDeviceProp device_prop{};
        HANDLE_ERROR(cudaGetDeviceProperties(&device_prop, i));
        std::cout << "Device name: " << device_prop.name << std::endl;
        std::cout << "Compute capability: " << device_prop.major << "." << device_prop.minor << std::endl;
        std::cout << "Total global memory: " << device_prop.totalGlobalMem << std::endl;
        std::cout << "Shared memory per block: " << device_prop.sharedMemPerBlock << std::endl;
        std::cout << "Registers per block: " << device_prop.regsPerBlock << std::endl;
        std::cout << "Warp size: " << device_prop.warpSize << std::endl;
        std::cout << "Memory pitch: " << device_prop.memPitch << std::endl;
        std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads dimensions: " << device_prop.maxThreadsDim[0] << " " << device_prop.maxThreadsDim[1] << " " << device_prop.maxThreadsDim[2] << std::endl;
        std::cout << "Max grid size: " << device_prop.maxGridSize[0] << " " << device_prop.maxGridSize[1] << " " << device_prop.maxGridSize[2] << std::endl;
        std::cout << "Clock rate: " << device_prop.clockRate << std::endl;
        std::cout << "Total constant memory: " << device_prop.totalConstMem << std::endl;
        std::cout << "Texture alignment: " << device_prop.textureAlignment << std::endl;
        std::cout << "Concurrent copy and execution: " << device_prop.deviceOverlap << std::endl;
        std::cout << "Number of multiprocessors: " << device_prop.multiProcessorCount << std::endl;
        std::cout << "Kernel execution timeout enabled: " << device_prop.kernelExecTimeoutEnabled << std::endl;
        std::cout << "Integrated: " << device_prop.integrated << std::endl;
        std::cout << "Can map host memory: " << device_prop.canMapHostMemory << std::endl;
        std::cout << "Supports page-locked memory: " << device_prop.canMapHostMemory << std::endl;
        std::cout << "Compute mode: " << device_prop.computeMode << std::endl;
        std::cout << "Maximum 1D texture size: " << device_prop.maxTexture1D << std::endl;
        std::cout << "Maximum 2D texture size: " << device_prop.maxTexture2D[0] << " " << device_prop.maxTexture2D[1] << std::endl;
        std::cout << "Maximum 3D texture size: " << device_prop.maxTexture3D[0] << " " << device_prop.maxTexture3D[1] << " " << device_prop.maxTexture3D[2] << std::endl;
        std::cout << "Maximum 1D layered texture dimensions: " << device_prop.maxTexture1DLayered[0] << " " << device_prop.maxTexture1DLayered[1] << std::endl;
        std::cout << "Maximum 2D layered texture dimensions: " << device_prop.maxTexture2DLayered[0] << " " << device_prop.maxTexture2DLayered[1] << " " << device_prop.maxTexture2DLayered[2] << std::endl;
        std::cout << "Surface alignment: " << device_prop.surfaceAlignment << std::endl;
        std::cout << "Concurrent kernels: " << device_prop.concurrentKernels << std::endl;
        std::cout << "ECC enabled: " << device_prop.ECCEnabled << std::endl;
        std::cout << "PCI bus ID: " << device_prop.pciBusID << std::endl;
        std::cout << "PCI device ID: " << device_prop.pciDeviceID << std::endl;
        std::cout << "PCI domain ID: " << device_prop.pciDomainID << std::endl;
        std::cout << "TCC driver: " << device_prop.tccDriver << std::endl;
        std::cout << "Async engine count: " << device_prop.asyncEngineCount << std::endl;
        std::cout << "Unified addressing: " << device_prop.unifiedAddressing << std::endl;
        std::cout << "Memory clock rate: " << device_prop.memoryClockRate << std::endl;
        std::cout << "Memory bus width: " << device_prop.memoryBusWidth << std::endl;
        std::cout << "L2 cache size: " << device_prop.l2CacheSize << std::endl;
        std::cout << "Max threads per multiprocessor: " << device_prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Stream priorities: " << device_prop.streamPrioritiesSupported << std::endl;
        std::cout << "Global L1 cache supported: " << device_prop.globalL1CacheSupported << std::endl;
        std::cout << "Local L1 cache supported: " << device_prop.localL1CacheSupported << std::endl;
        std::cout << "Shared memory per multiprocessor: " << device_prop.sharedMemPerMultiprocessor << std::endl;
        std::cout << "Registers per multiprocessor: " << device_prop.regsPerMultiprocessor << std::endl;
        std::cout << "Managed memory: " << device_prop.managedMemory << std::endl;
        std::cout << "Is multi-GPU board: " << device_prop.isMultiGpuBoard << std::endl;
        std::cout << "Multi-GPU board group ID: " << device_prop.multiGpuBoardGroupID << std::endl;
        std::cout << "Host native atomic supported: " << device_prop.hostNativeAtomicSupported << std::endl;
        std::cout << "Single to double precision perf ratio: " << device_prop.singleToDoublePrecisionPerfRatio << std::endl;
        std::cout << "Pageable memory access: " << device_prop.pageableMemoryAccess << std::endl;
        std::cout << "Concurrent managed access: " << device_prop.concurrentManagedAccess << std::endl;
        std::cout << "Compute preemption supported: " << device_prop.computePreemptionSupported << std::endl;
        std::cout << "Can use host pointer for registered memory: " << device_prop.canUseHostPointerForRegisteredMem << std::endl;
        std::cout << "Cooperative launch: " << device_prop.cooperativeLaunch << std::endl;
        std::cout << "Cooperative multi-device launch: " << device_prop.cooperativeMultiDeviceLaunch << std::endl;
        std::cout << "Pageable memory access uses host page tables: " << device_prop.pageableMemoryAccessUsesHostPageTables << std::endl;
        std::cout << "Direct managed memory access from host: " << device_prop.directManagedMemAccessFromHost << std::endl;
        std::cout << "Max blocks per multiprocessor: " << device_prop.maxBlocksPerMultiProcessor << std::endl;
    }
    int device_id;
    HANDLE_ERROR(cudaGetDevice(&device_id));
    std::cout << "Current device id: " << device_id << std::endl;
    cudaDeviceProp ideal_device_prop{};
    memset(&ideal_device_prop, 0, sizeof(cudaDeviceProp));
    ideal_device_prop.major = 1;
    ideal_device_prop.minor = 3;
    HANDLE_ERROR(cudaChooseDevice(&device_id, &ideal_device_prop));
    std::cout << "Ideal device id: " << device_id << std::endl;
    HANDLE_ERROR(cudaSetDevice(device_id));
    std::cout << "Setting current device id to " << device_id << std::endl;

    int* result = (int*)malloc(sizeof(int));
    int* dev_result;
    // HANDLE_ERROR: simple implementation of error catch
    // cudaMalloc: Just like malloc, but the parameter is pointer of pointer
    HANDLE_ERROR(cudaMalloc((void**)&dev_result, sizeof(int)));

    add_integer<<<1, 1>>>(3, 4, dev_result);
    // the following code is not allowed, because modifying cuda memory out of add_integer function
    // will result in memory error
    // *dev_result = 5;

    // copy the cuda memory from device gpu to host system dram
    HANDLE_ERROR(
            cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost)
    );

    std::cout << "Hello, World! result=" << *result << std::endl;
    cudaFree(dev_result);
    free(result);
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    need_redraw = true;
    glViewport(0, 0, width, height);
}

void process_input(GLFWwindow* window) {
    // glfwGetKey: get the state of the key
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    // wasd
    float pixel_per_input = 30 * scale;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        shift_y -= pixel_per_input;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        shift_y += pixel_per_input;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        shift_x += pixel_per_input;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        shift_x -= pixel_per_input;
    }
    // q, e
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        scale *= 1.0f + 0.7f * delta_time;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        scale /= 1.0f + 0.7f * delta_time;
    }
    // z to increase real part of C, x to decrease real part of C
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        C.r += 0.0001f;
    }
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
        C.r -= 0.0001f;
    }
    // c to increase imaginary part of C, v to decrease imaginary part of C
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        C.i += 0.0001f;
    }
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
        C.i -= 0.0001f;
    }
}

void scroll_callback(GLFWwindow* window, double unknown_offset, double scroll_offset) {
    need_redraw = true;
    const float coefficient = 0.1f * scale;
    scale -= coefficient * (float)scroll_offset;
    if (scale <= 0.0f) {
        scale = 0.00001f;
    }
//    std::cout << "scale: " << scale << std::endl;
}

void mouse_callback(GLFWwindow* window, double width_pos, double height_pos) {
    if (first_mouse) {
        last_x = width_pos;
        last_y = height_pos;
        first_mouse = false;
    }
    float x_offset = width_pos - last_x;
    float y_offset = last_y - height_pos;
    last_x = width_pos;
    last_y = height_pos;
    const float sensitivity = delta_time * 100.0f * scale;
    x_offset *= sensitivity;
    y_offset *= sensitivity;
    shift_x -= x_offset;
    shift_y -= y_offset;
//    std::cout << "shift_x: " << shift_x << " shift_y: " << shift_y << std::endl;
}