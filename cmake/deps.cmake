include(FetchContent)

# ---------------------------------------------------------------------------
# Dependency philosophy: zero compiled third-party dependencies in the core
# renderer. Only header-only libs are fetched here. OpenImageIO is the one
# system dependency (EXR/texture I/O) and is expected to be installed via
# the system package manager (Homebrew on macOS, apt on Linux).
#
# Threading: custom ThreadPool (std::thread)  — see src/render/ThreadPool.h
# BVH:       custom SAH BVH                   — see src/accel/BVH.h
# GPU:       MetalBackend (Phase 5, macOS)    — see src/backends/metal/
#            CUDABackend  (Phase 5, Linux/Win) — see src/backends/cuda/
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# spdlog — header-only structured logging (bundles fmt)
# ---------------------------------------------------------------------------
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1
    GIT_SHALLOW    TRUE
)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL  OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)

# ---------------------------------------------------------------------------
# CLI11 — single-header command-line parser
# ---------------------------------------------------------------------------
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(CLI11)

# ---------------------------------------------------------------------------
# Google Test — unit testing framework
# ---------------------------------------------------------------------------
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK             OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# OpenImageIO — EXR output and texture loading
# Expected to be installed via Homebrew: brew install openimageio
# ---------------------------------------------------------------------------
find_package(OpenImageIO REQUIRED)

# ---------------------------------------------------------------------------
# OpenUSD + MaterialX (optional — Phase 3)
# Expected: brew install usd  (or build from source)
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_USD)
    find_package(pxr      REQUIRED)
    find_package(MaterialX REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Open Shading Language (optional — Phase 4)
# MaterialX node implementations are often written in OSL; the two work together.
# Expected: brew install open-shading-language
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_OSL)
    find_package(OSL REQUIRED COMPONENTS oslexec oslcomp)
    target_link_libraries(anacapa_lib PUBLIC OSL::oslexec OSL::oslcomp)
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_OSL)
endif()

# ---------------------------------------------------------------------------
# Metal (optional — Phase 5, macOS only)
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "ANACAPA_ENABLE_METAL requires macOS/Apple Silicon")
    endif()
    find_library(METAL_LIBRARY    Metal    REQUIRED)
    find_library(FOUNDATION_LIB   Foundation REQUIRED)
    find_library(QUARTZCORE_LIB   QuartzCore REQUIRED)
endif()
