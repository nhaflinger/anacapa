cmake_minimum_required(VERSION 3.22)
project(anacapa CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ANACAPA_ENABLE_USD     "Build with OpenUSD support (Phase 3)"                ON)
option(ANACAPA_ENABLE_OSL     "Build with Open Shading Language support (Phase 4)"  OFF)
option(ANACAPA_ENABLE_ALEMBIC "Build with Alembic hair/curve loader"                ON)
option(ANACAPA_ENABLE_METAL   "Build Metal GPU backend (Phase 5, macOS)"            OFF)
option(ANACAPA_ENABLE_CUDA    "Build CUDA GPU backend (Phase 5)"                   OFF)
option(ANACAPA_ENABLE_OPTIX   "Build CUDA backend with OptiX hardware RT (Phase 6)" ON)
option(ANACAPA_ENABLE_ASAN    "Build with AddressSanitizer"                         OFF)
option(ANACAPA_ENABLE_OIDN    "Build with Intel OpenImageDenoise support"           ON)
option(ANACAPA_ENABLE_VIEWER  "Build SDL2+ImGui progressive render viewer"          OFF)
option(ANACAPA_ENABLE_MATERIALX "Build MaterialX GPU codegen (procedural shaders on GPU)" OFF)

# MaterialX GPU codegen is only useful when a GPU backend exists to run the
# generated shaders on — auto-enable unless the user explicitly opted out.
if((ANACAPA_ENABLE_METAL OR ANACAPA_ENABLE_CUDA) AND NOT ANACAPA_ENABLE_MATERIALX)
    message(STATUS "Auto-enabling ANACAPA_ENABLE_MATERIALX (a GPU backend is ON)")
    set(ANACAPA_ENABLE_MATERIALX ON CACHE BOOL "Build MaterialX GPU codegen (procedural shaders on GPU)" FORCE)
endif()

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
    src/sky/*.cpp
)

add_library(anacapa_lib STATIC ${ANACAPA_SOURCES})

target_include_directories(anacapa_lib
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(anacapa_lib
    PUBLIC
        OpenImageIO::OpenImageIO
        spdlog::spdlog_header_only
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
    if(APPLE)
        target_link_libraries(anacapa_lib PUBLIC
            "${OSL_LIB_DIR}/liboslexec.dylib"
            "${OSL_LIB_DIR}/liboslcomp.dylib"
        )
    else()
        target_link_libraries(anacapa_lib PUBLIC
            "${OSL_LIB_DIR}/liboslexec.so"
            "${OSL_LIB_DIR}/liboslcomp.so"
        )
    endif()
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

if(ANACAPA_ENABLE_MATERIALX)
    target_link_libraries(anacapa_lib PUBLIC
        MaterialXCore
        MaterialXFormat
        MaterialXGenShader
        MaterialXGenGlsl
        MaterialXGenMsl
    )
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_MATERIALX)

    # MaterialX's stdlib .mtlx documents (nodedefs, nodegraphs, target defs)
    # must be loadable at runtime by MaterialXCodegen.  Prefer MATERIALX_STDLIB_DIR
    # (set by MaterialXConfig.cmake when we take the USD-imported path); else fall
    # back to the FetchContent'd source tree's libraries/ directory.
    if(DEFINED MATERIALX_STDLIB_DIR)
        set(_ANACAPA_MATERIALX_LIB "${MATERIALX_STDLIB_DIR}")
    else()
        FetchContent_GetProperties(materialx SOURCE_DIR MATERIALX_SOURCE_DIR)
        set(_ANACAPA_MATERIALX_LIB "${MATERIALX_SOURCE_DIR}/libraries")
    endif()
    target_compile_definitions(anacapa_lib PUBLIC
        ANACAPA_MATERIALX_LIBRARIES_DIR="${_ANACAPA_MATERIALX_LIB}"
    )
endif()

if(ANACAPA_ENABLE_ALEMBIC)
    target_link_libraries(anacapa_lib PUBLIC Alembic::Alembic)
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_ALEMBIC)
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
    # OIIO 2.5+ vecparam.h uses C++20-only unevaluated lambdas, and its
    # ustring.h uses C++20 constexpr helpers.  nvcc 12+ supports C++20; on
    # nvcc 11 we fall back to C++17 and lose CUDA support for the newer OIIO.
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Use shared cudart globally — avoids nvlink trying to open the stub librt.a
    # that modern glibc ships (librt functions are merged into libc).
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_ARCH "75" CACHE STRING "CUDA SM architecture (e.g. 75, 86)")
    set_target_properties(anacapa_lib PROPERTIES
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(anacapa_lib PUBLIC ANACAPA_ENABLE_CUDA)
    # OIIO 2.5+ uses constexpr helpers marked __host__ that inline into
    # __device__ contexts (via ustring.h → ustringhash) — nvcc needs
    # --expt-relaxed-constexpr to accept the cross-call.
    target_compile_options(anacapa_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    # Skip GCC 11's serializeintrin.h when compiling with older nvcc that
    # doesn't recognise __builtin_ia32_serialize (Intel SERIALIZE, GCC 11+).
    # Defining the header guard makes it a no-op — no code in our TUs uses
    # _serialize() directly.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
        target_compile_definitions(anacapa_lib PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:_SERIALIZE_H_INCLUDED>)
    endif()
    add_subdirectory(src/gpu/cuda)
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(anacapa src/main.cpp src/export/SceneExporter.cpp)
target_include_directories(anacapa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(anacapa PRIVATE anacapa_lib CLI11::CLI11)
if(ANACAPA_ENABLE_CUDA)
    set_target_properties(anacapa PROPERTIES CUDA_RUNTIME_LIBRARY "Shared")
endif()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_OIDN)
    add_executable(denoise src/tools/denoise.cpp)
    target_link_libraries(denoise PRIVATE anacapa_lib CLI11::CLI11 OpenImageDenoise)
endif()

if(ANACAPA_ENABLE_ALEMBIC)
    add_executable(hair_export src/tools/hair_export.cpp)
    target_link_libraries(hair_export PRIVATE Alembic::Alembic nlohmann_json::nlohmann_json)
    target_compile_definitions(hair_export PRIVATE ANACAPA_ENABLE_ALEMBIC)
    target_include_directories(hair_export PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

if(ANACAPA_ENABLE_OSL)
    # Build-time OSL shader compiler (wraps anacapa's oslCompileShader API).
    add_executable(osl_compile src/tools/osl_compile.cpp)
    target_link_libraries(osl_compile PRIVATE anacapa_lib)
    target_include_directories(osl_compile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # Compile hand-authored shaders from src/shaders/ → scenes/materials/.
    # The output .oso is committed to the repo so OSL-disabled builds can still
    # reference the shader by name without needing to recompile.
    set(_shaders_src "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders")
    set(_shaders_dst "${CMAKE_CURRENT_SOURCE_DIR}/scenes/materials")

    set(_marschner_osl "${_shaders_src}/anacapa_marschner_hair.osl")
    set(_marschner_oso "${_shaders_dst}/anacapa_marschner_hair.oso")

    add_custom_command(
        OUTPUT  "${_marschner_oso}"
        COMMAND osl_compile "${_marschner_osl}" "${_marschner_oso}" "${_shaders_dst}"
        DEPENDS osl_compile "${_marschner_osl}"
        COMMENT "Compiling anacapa_marschner_hair.osl"
    )
    add_custom_target(anacapa_shaders ALL DEPENDS "${_marschner_oso}")
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
    FetchContent_GetProperties(stb               SOURCE_DIR STB_SOURCE_DIR)
    FetchContent_GetProperties(imgui_filebrowser SOURCE_DIR IMGUI_FILEBROWSER_SOURCE_DIR)
    target_include_directories(viewer PRIVATE
        ${IMGUI_SOURCE_DIR}
        ${IMGUI_SOURCE_DIR}/backends
        ${STB_SOURCE_DIR}
        ${IMGUI_FILEBROWSER_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(viewer PRIVATE
        SDL2::SDL2-static
        glad
        CLI11::CLI11
        OpenImageIO::OpenImageIO
    )
    # OpenGL
    if(APPLE)
        target_link_libraries(viewer PRIVATE "-framework OpenGL")
    else()
        find_package(OpenGL REQUIRED)
        target_link_libraries(viewer PRIVATE OpenGL::GL)
    endif()
    # OpenColorIO (optional — viewer falls back to ACES polynomial without it)
    find_package(OpenColorIO QUIET)
    if(OpenColorIO_FOUND)
        target_link_libraries(viewer PRIVATE OpenColorIO::OpenColorIO)
        target_compile_definitions(viewer PRIVATE ANACAPA_HAVE_OCIO)
        message(STATUS "Viewer: OpenColorIO ${OpenColorIO_VERSION} found — OCIO color management enabled")
    else()
        message(STATUS "Viewer: OpenColorIO not found — falling back to built-in ACES polynomial")
    endif()
    # Silence warnings in ImGui's own sources
    set_source_files_properties(${IMGUI_SOURCES}
        PROPERTIES COMPILE_FLAGS "-w")
endif()

# ---------------------------------------------------------------------------
# Blender addon — zip tools/anacapa_renderer/ for install via Blender UI
# Excludes __pycache__ and .pyc files.  Rebuilds whenever any .py file changes.
# ---------------------------------------------------------------------------
set(_addon_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/tools/anacapa_renderer")
set(_addon_zip     "${CMAKE_BINARY_DIR}/bin/anacapa_renderer.zip")
set(_addon_py_files
    "${_addon_src_dir}/__init__.py"
    "${_addon_src_dir}/engine.py"
    "${_addon_src_dir}/export.py"
    "${_addon_src_dir}/operators.py"
    "${_addon_src_dir}/properties.py"
    "${_addon_src_dir}/ui.py"
    "${_addon_src_dir}/blender_prep_for_usd_export.py"
    "${_addon_src_dir}/anacapa_scene_export.py"
    "${_addon_src_dir}/osl/anacapa_lib.h"
)
add_custom_command(
    OUTPUT  "${_addon_zip}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E rm -f "${_addon_zip}"
    COMMAND python3
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package_addon.py"
        "${_addon_zip}"
        ${_addon_py_files}
    DEPENDS ${_addon_py_files}
    COMMENT "Packaging Blender addon → bin/anacapa_renderer.zip"
)
add_custom_target(anacapa_blender DEPENDS "${_addon_zip}")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
