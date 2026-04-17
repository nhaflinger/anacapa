# ---------------------------------------------------------------------------
# Pure-CUDA GPU backend (no OptiX)
#
# Shade.cu is compiled as a regular CUDA source alongside the host-side .cu
# files.  Separable compilation lets CudaPathIntegrator.cu call the shade()
# __global__ kernel defined in Shade.cu across translation units.
# ---------------------------------------------------------------------------

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "nvcc not found — install nvidia-cuda-toolkit")
endif()

# ---------------------------------------------------------------------------
# Target GPU architecture
# Default sm_75 (Turing / RTX 2000–4000 / A-series).  Override: cmake -DCUDA_ARCH=86
# ---------------------------------------------------------------------------
set(CUDA_ARCH "75" CACHE STRING "CUDA SM architecture number (e.g. 75, 86, 89)")
message(STATUS "CUDA SM target: sm_${CUDA_ARCH}")

# ---------------------------------------------------------------------------
# Host-side and device-side CUDA sources added to anacapa_lib
# ---------------------------------------------------------------------------
target_sources(anacapa_lib PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/CudaContext.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/CudaAccelStructure.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/CudaPathIntegrator.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/Shade.cu"
)

# Separable compilation: allows __global__ in Shade.cu to be called from
# CudaPathIntegrator.cu after device linking.
set_property(TARGET anacapa_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(anacapa_lib PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Link CUDA runtime
target_link_libraries(anacapa_lib PUBLIC CUDA::cudart)

# Architecture flags applied to all CUDA sources in anacapa_lib
set_property(TARGET anacapa_lib PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH}")

# Suppress nvcc/GCC compatibility warnings
target_compile_options(anacapa_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --allow-unsupported-compiler
        -Xcompiler=-Wno-pedantic
        -Xcompiler=-D_PSTL_GLUE_ALGORITHM_DEFS_H
    >
)
