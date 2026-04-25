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
