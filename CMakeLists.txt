cmake_minimum_required(VERSION 3.25)
project(anacapa CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ANACAPA_ENABLE_USD    "Build with OpenUSD support (Phase 3)"                OFF)
option(ANACAPA_ENABLE_OSL    "Build with Open Shading Language support (Phase 4)"  OFF)
option(ANACAPA_ENABLE_METAL  "Build Metal GPU backend (Phase 5, macOS)"            OFF)
option(ANACAPA_ENABLE_CUDA   "Build CUDA/OptiX GPU backend (Phase 5)"              OFF)
option(ANACAPA_ENABLE_ASAN   "Build with AddressSanitizer"                         OFF)

# ---------------------------------------------------------------------------
# Dependencies (all header-only except OpenImageIO)
# ---------------------------------------------------------------------------
include(cmake/deps.cmake)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
file(GLOB_RECURSE ANACAPA_SOURCES
    src/core/*.cpp
    src/accel/*.cpp
    src/shading/*.cpp
    src/integrator/*.cpp
    src/sampling/*.cpp
    src/film/*.cpp
    src/render/*.cpp
    src/scene/*.cpp
)

add_library(anacapa_lib STATIC ${ANACAPA_SOURCES})

target_include_directories(anacapa_lib
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(anacapa_lib
    PUBLIC
        OpenImageIO::OpenImageIO
        spdlog::spdlog
)

# compiler options applied after target exists
include(cmake/compiler_options.cmake)

if(ANACAPA_ENABLE_USD)
    target_link_libraries(anacapa_lib PUBLIC
        pxr::usd pxr::usdGeom pxr::usdLux pxr::usdShade
        MaterialXCore MaterialXFormat MaterialXGenShader
    )
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_USD)
endif()

if(ANACAPA_ENABLE_METAL)
    target_link_libraries(anacapa_lib PUBLIC
        ${METAL_LIBRARY} ${FOUNDATION_LIB} ${QUARTZCORE_LIB}
    )
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_METAL)
    add_subdirectory(src/backends/metal)
endif()

if(ANACAPA_ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_CUDA)
    add_subdirectory(src/backends/cuda)
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(anacapa src/main.cpp)
target_link_libraries(anacapa PRIVATE anacapa_lib CLI11::CLI11)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
