cmake_minimum_required(VERSION 3.22)
project(anacapa CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ANACAPA_ENABLE_USD    "Build with OpenUSD support (Phase 3)"                ON)
option(ANACAPA_ENABLE_OSL    "Build with Open Shading Language support (Phase 4)"  OFF)
option(ANACAPA_ENABLE_METAL  "Build Metal GPU backend (Phase 5, macOS)"            OFF)
option(ANACAPA_ENABLE_CUDA   "Build pure-CUDA GPU backend (Phase 5)"              OFF)
option(ANACAPA_ENABLE_ASAN   "Build with AddressSanitizer"                         OFF)
option(ANACAPA_ENABLE_OIDN   "Build with Intel OpenImageDenoise support"           ON)
option(ANACAPA_ENABLE_VIEWER "Build SDL2+ImGui progressive render viewer"          OFF)

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
        nlohmann_json::nlohmann_json
)

# compiler options applied after target exists
include(cmake/compiler_options.cmake)

if(ANACAPA_ENABLE_USD)
    target_include_directories(anacapa_lib PUBLIC "${USD_ROOT}/include")
    target_link_directories(anacapa_lib PUBLIC "${USD_ROOT}/lib")
    target_link_libraries(anacapa_lib PUBLIC
        usd_usd usd_usdGeom usd_usdLux usd_usdShade usd_usdRender
        usd_tf usd_gf usd_vt usd_sdf usd_ar usd_plug
    )
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_USD)
endif()

if(ANACAPA_ENABLE_OSL)
    target_include_directories(anacapa_lib PUBLIC "${OSL_INCLUDE_DIR}")
    target_link_directories(anacapa_lib PUBLIC "${OSL_LIB_DIR}")
    target_link_libraries(anacapa_lib PUBLIC
        "${OSL_LIB_DIR}/liboslexec.dylib"
        "${OSL_LIB_DIR}/liboslcomp.dylib"
    )
    set_target_properties(anacapa_lib PROPERTIES
        BUILD_RPATH "${OSL_LIB_DIR}"
        INSTALL_RPATH "${OSL_LIB_DIR}"
    )
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_OSL)

    # Copy _mx_stdlib.h (shared OSL boilerplate) to the build directory so
    # oslCompileShader can add it to the include path at runtime.
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/shading/_mx_stdlib.h"
        "${CMAKE_BINARY_DIR}/_mx_stdlib.h"
        COPYONLY
    )
    target_compile_definitions(anacapa_lib PUBLIC
        ANACAPA_MX_STDLIB_DIR="${CMAKE_BINARY_DIR}"
    )
endif()

if(ANACAPA_ENABLE_METAL)
    target_link_libraries(anacapa_lib PUBLIC
        ${METAL_LIBRARY} ${FOUNDATION_LIB} ${QUARTZCORE_LIB}
    )
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_METAL)
    add_subdirectory(src/gpu/metal)
endif()

if(ANACAPA_ENABLE_OIDN)
    target_link_libraries(anacapa_lib PUBLIC OpenImageDenoise)
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_OIDN)
endif()

if(ANACAPA_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Use shared cudart globally — avoids nvlink trying to open the stub librt.a
    # that modern glibc ships (librt functions are merged into libc).
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_ARCH "75" CACHE STRING "CUDA SM architecture (e.g. 75, 86)")
    set_target_properties(anacapa_lib PROPERTIES
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_CUDA)
    add_subdirectory(src/gpu/cuda)
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(anacapa src/main.cpp)
target_link_libraries(anacapa PRIVATE anacapa_lib CLI11::CLI11)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_OIDN)
    add_executable(denoise src/tools/denoise.cpp)
    target_link_libraries(denoise PRIVATE anacapa_lib CLI11::CLI11 OpenImageDenoise)
endif()

if(ANACAPA_ENABLE_VIEWER)
    # Compile Dear ImGui sources directly (it has no CMake build system)
    FetchContent_GetProperties(imgui SOURCE_DIR IMGUI_SOURCE_DIR)
    set(IMGUI_SOURCES
        ${IMGUI_SOURCE_DIR}/imgui.cpp
        ${IMGUI_SOURCE_DIR}/imgui_draw.cpp
        ${IMGUI_SOURCE_DIR}/imgui_tables.cpp
        ${IMGUI_SOURCE_DIR}/imgui_widgets.cpp
        ${IMGUI_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
        ${IMGUI_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    add_executable(viewer src/tools/viewer.cpp ${IMGUI_SOURCES})
    FetchContent_GetProperties(stb  SOURCE_DIR STB_SOURCE_DIR)
    target_include_directories(viewer PRIVATE
        ${IMGUI_SOURCE_DIR}
        ${IMGUI_SOURCE_DIR}/backends
        ${STB_SOURCE_DIR}
    )
    target_link_libraries(viewer PRIVATE
        SDL2::SDL2-static
        glad
        CLI11::CLI11
    )
    # OpenGL
    if(APPLE)
        target_link_libraries(viewer PRIVATE "-framework OpenGL")
    else()
        find_package(OpenGL REQUIRED)
        target_link_libraries(viewer PRIVATE OpenGL::GL)
    endif()
    # Silence warnings in ImGui's own sources
    set_source_files_properties(${IMGUI_SOURCES}
        PROPERTIES COMPILE_FLAGS "-w")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
