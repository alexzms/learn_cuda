cmake_minimum_required(VERSION 3.26)
project(learn_cuda_chap_1_4 LANGUAGES CXX CUDA)
project(vector_addition_compare LANGUAGES CXX CUDA)
project(wave_render LANGUAGES CXX CUDA)
project(learn_share_memory LANGUAGES CXX CUDA)
project(learn_constant_mem LANGUAGES CXX CUDA)
project(learn_texture_mem LANGUAGES CXX CUDA)
project(learn_atom_operation LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_80,code=sm_80)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86)

set_source_files_properties(
        ./lib/glad.c
        includes/book.h
        includes/cpu_anim.h
        includes/cpu_bitmap.h
        includes/gl_helper.h
        includes/gpu_anim.h
        includes/Shader.h
        PROPERTIES
        LANGUAGE CXX
)

set_source_files_properties(
        learn_atom_operation.cu
        learn_texture_mem.cu
        learn_cuda_chap_1_4.cu
        vector_addition_compare.cu
        wave_render.cu
        learn_share_memory.cu
        learn_constant_mem
        includes/check_error.cuh
        PROPERTIES
        LANGUAGE CUDA
)

add_executable(learn_share_memory
        learn_share_memory.cu
        ./lib/glad.c
        )

add_executable(wave_render
        wave_render.cu
        ./lib/glad.c

)

add_executable(learn_cuda_chap_1_4
        learn_cuda_chap_1_4.cu
        ./lib/glad.c

)

add_executable(vector_addition_compare
        vector_addition_compare.cu
        ./lib/glad.c
)

add_executable(learn_constant_mem
        learn_constant_mem.cu
        ./lib/glad.c
)

add_executable(learn_texture_mem
        learn_texture_mem.cu
        ./lib/glad.c
)

add_executable(learn_atom_operation
        learn_atom_operation.cu
        ./lib/glad.c
)

include_directories(E:/ComputerGraphics/libraries/OpenGL/includes)
include_directories(E:/ComputerGraphics/libraries/Utilities/includes)
include_directories(includes)
link_directories(E:/ComputerGraphics/libraries/OpenGL/lib)
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64")

target_link_libraries(learn_cuda_chap_1_4 glfw3 glut64 opengl32.lib )
target_link_libraries(vector_addition_compare glfw3 glut64 opengl32.lib )
target_link_libraries(wave_render glfw3 glut64 opengl32.lib )
target_link_libraries(learn_share_memory glfw3 glut64 opengl32.lib )
target_link_libraries(learn_constant_mem glfw3 glut64 opengl32.lib )
target_link_libraries(learn_texture_mem glfw3 glut64 opengl32.lib )
target_link_libraries(learn_atom_operation glfw3 glut64 opengl32.lib )

target_link_libraries(wave_render
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)
target_link_libraries(learn_cuda_chap_1_4
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)
target_link_libraries(vector_addition_compare
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)
target_link_libraries(learn_share_memory
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)

target_link_libraries(learn_constant_mem
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)

target_link_libraries(learn_texture_mem
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)

target_link_libraries(learn_atom_operation
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)


set_target_properties(learn_cuda_chap_1_4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(vector_addition_compare PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(wave_render PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(learn_share_memory PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(learn_constant_mem PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(learn_texture_mem PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(learn_atom_operation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
