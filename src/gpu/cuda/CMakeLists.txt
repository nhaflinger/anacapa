# ---------------------------------------------------------------------------
# CUDA GPU backend (Phase 5, transitioning to CUDA + OptiX in Phase 6).
#
# Today: Shade.cu is a megakernel compiled alongside the host-side .cu files;
# CudaPathIntegrator launches it via <<< grid, block >>>.  Separable compilation
# lets CudaPathIntegrator.cu call the shade() __global__ kernel across TUs.
#
# Phase 6 (in progress): Shade.cu will be split into OptiX raygen / closesthit
# / miss programs compiled to PTX (see anacapa_compile_ptx() helper below) and
# loaded at runtime by an OptiX pipeline.  The software BVH built in
# CudaAccelStructure.cu will be replaced by per-mesh OptiX GAS + a top-level
# IAS, with motion-aware GASes for parity with the Metal primitive_motion path.
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
# Host-side CUDA sources added to anacapa_lib.  shaders/Shade.cu is NOT in
# this list — it's compiled to PTX (below) and loaded at runtime by the
# OptiX pipeline.
# ---------------------------------------------------------------------------
target_sources(anacapa_lib PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/CudaContext.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/CudaAccelStructure.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/CudaPathIntegrator.cu"
)

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

# ---------------------------------------------------------------------------
# OptiX wiring (Phase 6)
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_OPTIX)
    target_include_directories(anacapa_lib PRIVATE "${OptiX_INSTALL_DIR}/include")
    target_compile_definitions(anacapa_lib PRIVATE ANACAPA_ENABLE_OPTIX)

    # nvrtc — runtime compilation of MaterialX-generated CUDA source into
    # OptiX direct-callable PTX modules (see CudaPathIntegrator.cu's
    # buildMaterialXCallables(), mirrors the Metal backend's
    # newLibraryWithSource: runtime compilation). Needs the same target
    # architecture as the offline-compiled PTX above.
    if(ANACAPA_ENABLE_MATERIALX)
        target_link_libraries(anacapa_lib PUBLIC CUDA::nvrtc)
        target_compile_definitions(anacapa_lib PRIVATE ANACAPA_CUDA_ARCH="${CUDA_ARCH}")
    endif()

    # Where the integrator looks for compiled PTX shaders at runtime.  PTX
    # produced by anacapa_compile_ptx() is written into this directory.
    set(ANACAPA_PTX_DIR "${CMAKE_BINARY_DIR}/ptx" CACHE INTERNAL "PTX output dir")
    file(MAKE_DIRECTORY "${ANACAPA_PTX_DIR}")
    target_compile_definitions(anacapa_lib PRIVATE
        ANACAPA_PTX_DIR=\"${ANACAPA_PTX_DIR}\")

    # ---------------------------------------------------------------------------
    # anacapa_compile_ptx(<ptx_name> <source.cu> [extra_include_dir...])
    #
    # Compiles a CUDA source to ${ANACAPA_PTX_DIR}/<ptx_name>.ptx using the
    # CMake-detected nvcc.  The PTX target is added as a dependency of
    # anacapa_lib so it builds whenever the lib does.
    #
    # Used by Step 4 of the OptiX migration to compile OptiX raygen / CH / miss
    # programs.  Not yet invoked anywhere — wired up here so Step 4 only has
    # to add the call.
    # ---------------------------------------------------------------------------
    function(anacapa_compile_ptx ptx_name src_cu)
        set(ptx_out "${ANACAPA_PTX_DIR}/${ptx_name}.ptx")
        set(extra_includes ${ARGN})
        set(_inc_flags "")
        foreach(_inc IN LISTS extra_includes)
            list(APPEND _inc_flags "-I${_inc}")
        endforeach()
        add_custom_command(
            OUTPUT  "${ptx_out}"
            COMMAND ${CMAKE_CUDA_COMPILER}
                    -ptx
                    --use_fast_math
                    -std=c++17
                    -arch=sm_${CUDA_ARCH}
                    -I${CMAKE_CURRENT_SOURCE_DIR}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/shaders
                    -I${OptiX_INSTALL_DIR}/include
                    ${_inc_flags}
                    -o "${ptx_out}"
                    "${src_cu}"
            DEPENDS "${src_cu}"
            COMMENT "Compiling PTX: ${ptx_name}.ptx"
            VERBATIM
        )
        add_custom_target(ptx_${ptx_name} ALL DEPENDS "${ptx_out}")
        add_dependencies(anacapa_lib ptx_${ptx_name})
    endfunction()

    # OptiX raygen / closesthit / miss programs
    anacapa_compile_ptx(Shade "${CMAKE_CURRENT_SOURCE_DIR}/shaders/Shade.cu")
endif()
