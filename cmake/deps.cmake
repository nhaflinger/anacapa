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
#
# spdlog 1.15 bundles fmt 11; OIIO 2.5.x bundles fmt 10 in its detail/
# header tree.  When both land in the same translation unit (any file that
# includes both spdlog and OIIO — MaterialXCodegen, USDLoader, etc.) the two
# incompatible fmt namespaces collide.  Newer OIIO (2.6+) bundles fmt 11 and
# matches spdlog 1.15; older installs (e.g. our Linux graphics stack at
# ~/oiio 2.5.16, which bundles fmt exactly 10.0.0) don't.
#
# On systems where OIIO bundles fmt 10, we FetchContent standalone fmt 10.0.0
# and tell spdlog to use it externally (SPDLOG_FMT_EXTERNAL) — that way both
# spdlog and OIIO see the same v10 symbol set instead of two slightly
# different v10 headers stepping on each other (int_checker, buffer_appender,
# and other internals moved between fmt 10.0 and 10.2).
# ---------------------------------------------------------------------------
set(_anacapa_use_external_fmt OFF)
if(NOT APPLE AND EXISTS "$ENV{HOME}/oiio/include/OpenImageIO/detail/fmt/core.h")
    file(READ "$ENV{HOME}/oiio/include/OpenImageIO/detail/fmt/core.h" _oiio_fmt_core
         LIMIT 4096)
    if(_oiio_fmt_core MATCHES "FMT_VERSION 100000")
        set(_anacapa_use_external_fmt ON)
        message(STATUS "fmt pinned to 10.0.0 externally (matches OIIO 2.5.x bundled fmt 10.0.0)")
    endif()
endif()

if(_anacapa_use_external_fmt)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        10.0.0
        GIT_SHALLOW    TRUE
    )
    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
    set(FMT_TEST    OFF CACHE BOOL "" FORCE)
    set(FMT_DOC     OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fmt)
endif()

# spdlog 1.15 needs fmt >= 11 (uses fmt/base.h).  When we pin external fmt
# to 10.0.0 for OIIO compatibility, drop spdlog to 1.14.1 (last release that
# supports fmt 10).  Mac / newer stacks keep 1.15.3.
if(_anacapa_use_external_fmt)
    set(_anacapa_spdlog_tag "v1.14.1")
else()
    set(_anacapa_spdlog_tag "v1.15.3")
endif()
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        ${_anacapa_spdlog_tag}
    GIT_SHALLOW    TRUE
)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
if(_anacapa_use_external_fmt)
    set(SPDLOG_FMT_EXTERNAL ON  CACHE BOOL "" FORCE)
else()
    set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
endif()
FetchContent_MakeAvailable(spdlog)

# ---------------------------------------------------------------------------
# nlohmann/json — single-header JSON parser (header-only)
# Used by USDLoader to read MaterialX JSON sidecars produced by the prep script.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

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
# OpenUSD (optional — Phase 4)
# Default search path: ~/usd  (built by build_scripts/build_usd.py)
# Override with: cmake -DUSD_ROOT=/path/to/usd
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_USD)
    set(USD_ROOT "$ENV{HOME}/usd" CACHE PATH "OpenUSD install root")
    find_package(pxr REQUIRED
        PATHS "${USD_ROOT}" "${USD_ROOT}/lib/cmake/pxr"
        NO_DEFAULT_PATH)
    message(STATUS "OpenUSD ${PXR_VERSION} found at ${USD_ROOT}")
endif()

# ---------------------------------------------------------------------------
# Open Shading Language (optional — Phase 4)
#
# Strategy: link against Blender's bundled liboslexec/liboslcomp dylibs
# (no system package needed), and fetch the OSL 1.14.7 headers-only from
# GitHub so we can compile against them.
#
# Override paths:
#   OSL_INCLUDE_DIR  — path to OSL headers (src/include inside a checkout)
#   OSL_LIB_DIR      — directory containing liboslexec.dylib / .so
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_OSL)
    # --- Headers: fetch OSL 1.14.7 source for its include/ directory ---------
    if(NOT OSL_INCLUDE_DIR)
        FetchContent_Declare(
            osl_headers
            GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/OpenShadingLanguage.git
            GIT_TAG        v1.14.7.0
            GIT_SHALLOW    TRUE
            # Only need the include directory — skip everything else.
            GIT_SUBMODULES ""
        )
        # We never call FetchContent_MakeAvailable (no build needed) — just
        # populate so we know the source directory.
        FetchContent_GetProperties(osl_headers)
        if(NOT osl_headers_POPULATED)
            FetchContent_Populate(osl_headers)
        endif()
        set(OSL_INCLUDE_DIR "${osl_headers_SOURCE_DIR}/src/include"
            CACHE PATH "Path to OSL headers" FORCE)
        # Copy pre-generated headers (oslconfig.h, oslversion.h) that OSL's own
        # cmake would normally produce — avoids requiring a full OSL build.
        file(COPY "${CMAKE_SOURCE_DIR}/cmake/osl_generated/OSL/"
             DESTINATION "${OSL_INCLUDE_DIR}/OSL")
    endif()
    message(STATUS "OSL headers: ${OSL_INCLUDE_DIR}")

    # --- Libraries: prefer explicit OSL_LIB_DIR, then Blender (Steam/App) ----
    if(NOT OSL_LIB_DIR)
        # Steam install
        set(_blender_steam
            "$ENV{HOME}/Library/Application Support/Steam/steamapps/common/Blender/Blender.app/Contents/Resources/lib")
        # Non-Steam .app install
        set(_blender_app
            "/Applications/Blender.app/Contents/Resources/lib")
        if(EXISTS "${_blender_steam}/liboslexec.dylib")
            set(OSL_LIB_DIR "${_blender_steam}" CACHE PATH "OSL library directory" FORCE)
        elseif(EXISTS "${_blender_app}/liboslexec.dylib")
            set(OSL_LIB_DIR "${_blender_app}" CACHE PATH "OSL library directory" FORCE)
        else()
            message(FATAL_ERROR
                "ANACAPA_ENABLE_OSL: cannot find liboslexec.dylib.\n"
                "Set -DOSL_LIB_DIR=/path/to/dir/containing/liboslexec")
        endif()
    endif()
    message(STATUS "OSL libs: ${OSL_LIB_DIR}")

    # target_* commands are applied in CMakeLists.txt after anacapa_lib is defined
endif()

# ---------------------------------------------------------------------------
# MaterialX — GPU shader codegen (MSL for Metal, GLSL for CUDA post-processing)
#
# This is an explicit exception to the "header-only only" dependency
# philosophy above (same exception already made for OSL/OIIO): MaterialX has
# no Homebrew formula and its C++ core/generators aren't available anywhere
# on a typical dev machine (Blender bundles a Python-only build with no
# linkable headers/libs) — so unlike OSL, there's no precompiled shortcut and
# it must be built from source.
#
# Only the pieces we actually use are enabled: core doc model, format I/O,
# the shader-generation framework, and the MSL/GLSL generator backends. No
# Python bindings, viewer, tests, docs, or OSL/MDL generators (Blender's own
# bundled MaterialX already covers .osl generation for the CPU path).
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_MATERIALX)
    # When USD is built with MaterialX support, pxrConfig.cmake already ran
    # `find_package(MaterialX REQUIRED)` above (in the pxr find_package call),
    # importing MaterialXCore/Format/etc. as targets from USD's bundled
    # install.  Trying to FetchContent our own copy of MaterialX in that case
    # collides on the target names (add_library sees the imported target and
    # errors out).  Detect the imported case and skip FetchContent — the
    # already-imported targets are what our code will link against.
    #
    # The trade-off is that USD's bundled MaterialX is often an older release
    # than v1.39.3 (e.g. USD 24.08 bundles 1.38.10).  If MaterialXCodegen /
    # MxGlslToCuda use v1.39-only API this will surface as a compile error
    # rather than a link/runtime surprise.
    if(TARGET MaterialXCore)
        message(STATUS "MaterialX already imported by USD — skipping FetchContent")
    else()
        FetchContent_Declare(
            materialx
            GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/MaterialX.git
            GIT_TAG        v1.39.3
            GIT_SHALLOW    TRUE
        )
        set(MATERIALX_BUILD_PYTHON  OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_VIEWER  OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_DOCS    OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_RENDER  OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_OSL OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_MDL OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_GLSL ON  CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_MSL  ON  CACHE BOOL "" FORCE)
        set(MATERIALX_INSTALL_INCLUDE_PATH "" CACHE STRING "" FORCE)
        set(MATERIALX_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(materialx)
    endif()
endif()

# ---------------------------------------------------------------------------
# Intel OpenImageDenoise (optional — denoising phase)
# Expected: brew install open-image-denoise
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_OIDN)
    find_package(OpenImageDenoise REQUIRED
        PATHS /opt/homebrew/lib/cmake /opt/homebrew/lib/cmake/OpenImageDenoise-2.4.1
        PATH_SUFFIXES OpenImageDenoise)
endif()

# ---------------------------------------------------------------------------
# SDL2 + Dear ImGui (optional — render viewer)
# Both fetched via FetchContent; no system install needed.
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_VIEWER)
    # stb_image — single-header image loader
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(stb)

    # glad — OpenGL function loader (GL 3.3 core, no extensions)
    FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG        v0.1.36
        GIT_SHALLOW    TRUE
    )
    set(GLAD_PROFILE  "core"  CACHE STRING "" FORCE)
    set(GLAD_API      "gl=3.3" CACHE STRING "" FORCE)
    set(GLAD_GENERATOR "c"    CACHE STRING "" FORCE)
    FetchContent_MakeAvailable(glad)

    # SDL2
    FetchContent_Declare(
        SDL2
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-2.30.3
        GIT_SHALLOW    TRUE
    )
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON  CACHE BOOL "" FORCE)
    set(SDL_TEST   OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SDL2)

    # Dear ImGui — no CMakeLists; we compile its sources directly in the viewer target
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(imgui)

    # imgui-filebrowser — pure ImGui cross-platform file browser (single header)
    FetchContent_Declare(
        imgui_filebrowser
        GIT_REPOSITORY https://github.com/AirGuanZ/imgui-filebrowser.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(imgui_filebrowser)
endif()

# ---------------------------------------------------------------------------
# Alembic (optional — hair/curve loader)
# Expected: brew install alembic
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_ALEMBIC)
    find_package(Alembic REQUIRED
        PATHS "/opt/homebrew/Cellar/alembic/1.8.11/lib/cmake/Alembic"
              "/opt/homebrew/lib/cmake/Alembic"
              "/usr/local/lib/cmake/Alembic"
        NO_DEFAULT_PATH)
    message(STATUS "Alembic ${Alembic_VERSION} found")
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

# ---------------------------------------------------------------------------
# OptiX (optional — Phase 6, requires CUDA)
#
# Header-only SDK; the ray-tracing engine ships inside the NVIDIA driver.
# Default search path: ~/optix  (downloaded from developer.nvidia.com/optix)
# Override with: cmake -DOptiX_INSTALL_DIR=/path/to/optix-sdk
# ---------------------------------------------------------------------------
if(ANACAPA_ENABLE_CUDA AND ANACAPA_ENABLE_OPTIX)
    set(OptiX_INSTALL_DIR "$ENV{HOME}/optix" CACHE PATH "NVIDIA OptiX SDK root")

    # If the SDK was extracted with its versioned subdirectory intact (the
    # default of the .sh installer), descend into it.
    if(NOT EXISTS "${OptiX_INSTALL_DIR}/include/optix.h")
        file(GLOB _optix_subdir
             RELATIVE "${OptiX_INSTALL_DIR}"
             "${OptiX_INSTALL_DIR}/NVIDIA-OptiX-SDK-*")
        foreach(_sub IN LISTS _optix_subdir)
            if(EXISTS "${OptiX_INSTALL_DIR}/${_sub}/include/optix.h")
                set(OptiX_INSTALL_DIR "${OptiX_INSTALL_DIR}/${_sub}"
                    CACHE PATH "NVIDIA OptiX SDK root" FORCE)
                break()
            endif()
        endforeach()
    endif()

    if(NOT EXISTS "${OptiX_INSTALL_DIR}/include/optix.h")
        message(FATAL_ERROR
            "ANACAPA_ENABLE_OPTIX: cannot find optix.h under "
            "${OptiX_INSTALL_DIR}/include.\n"
            "Either install the OptiX SDK to ~/optix, or pass "
            "-DOptiX_INSTALL_DIR=/path/to/sdk, or disable with "
            "-DANACAPA_ENABLE_OPTIX=OFF.")
    endif()
    # Extract version from optix.h (OPTIX_VERSION = major*10000 + minor*100 + micro)
    file(STRINGS "${OptiX_INSTALL_DIR}/include/optix.h" _optix_version_line
         REGEX "^#define OPTIX_VERSION +[0-9]+")
    if(_optix_version_line MATCHES "OPTIX_VERSION +([0-9]+)")
        set(_optix_v ${CMAKE_MATCH_1})
        math(EXPR _optix_major "${_optix_v} / 10000")
        math(EXPR _optix_minor "(${_optix_v} / 100) % 100")
        message(STATUS "OptiX SDK ${_optix_major}.${_optix_minor} found at ${OptiX_INSTALL_DIR}")
    else()
        message(STATUS "OptiX SDK found at ${OptiX_INSTALL_DIR}")
    endif()
endif()
