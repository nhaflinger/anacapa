# ---------------------------------------------------------------------------
# Metal GPU backend
#
# Compiles .metal shaders → .air → anacapa.metallib at build time.
# Adds MetalContext, MetalBuffer, MetalPathIntegrator to anacapa_lib.
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.25)

# ---------------------------------------------------------------------------
# Locate xcrun — prefers full Xcode (has 'metal') over Command Line Tools
# ---------------------------------------------------------------------------
find_program(XCRUN xcrun HINTS
    /Applications/Xcode.app/Contents/Developer/usr/bin
    /usr/bin
    REQUIRED)

# Resolve the actual metal / metallib executables so custom_command paths
# are absolute and immune to PATH changes at build time.
execute_process(
    COMMAND ${XCRUN} --sdk macosx --find metal
    OUTPUT_VARIABLE METAL_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

execute_process(
    COMMAND ${XCRUN} --sdk macosx --find metallib
    OUTPUT_VARIABLE METALLIB_LINKER
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

if(NOT METAL_COMPILER OR NOT METALLIB_LINKER)
    # Fallback: search inside Xcode.app directly
    find_program(METAL_COMPILER metal
        HINTS /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
        REQUIRED)
    find_program(METALLIB_LINKER metallib
        HINTS /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
        REQUIRED)
endif()

message(STATUS "Metal compiler : ${METAL_COMPILER}")
message(STATUS "metallib linker: ${METALLIB_LINKER}")

# ---------------------------------------------------------------------------
# Shader compilation: .metal → .air → .metallib
# ---------------------------------------------------------------------------
set(METAL_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(METALLIB_OUTPUT  "${CMAKE_BINARY_DIR}/anacapa.metallib")

# Collect all .metal sources
file(GLOB METAL_SOURCES "${METAL_SHADER_DIR}/*.metal")

# Compile each .metal → .air
set(AIR_FILES "")
foreach(METAL_SRC ${METAL_SOURCES})
    get_filename_component(STEM ${METAL_SRC} NAME_WE)
    set(AIR_OUT "${CMAKE_BINARY_DIR}/${STEM}.air")

    add_custom_command(
        OUTPUT  ${AIR_OUT}
        COMMAND ${METAL_COMPILER}
                    -std=metal3.0
                    -I "${METAL_SHADER_DIR}"
                    -c "${METAL_SRC}"
                    -o "${AIR_OUT}"
        DEPENDS ${METAL_SRC} "${METAL_SHADER_DIR}/SharedTypes.h"
        COMMENT "Compiling Metal shader: ${STEM}.metal"
    )
    list(APPEND AIR_FILES ${AIR_OUT})
endforeach()

# Link all .air files → single .metallib
add_custom_command(
    OUTPUT  ${METALLIB_OUTPUT}
    COMMAND ${METALLIB_LINKER} ${AIR_FILES} -o ${METALLIB_OUTPUT}
    DEPENDS ${AIR_FILES}
    COMMENT "Linking anacapa.metallib"
)

add_custom_target(anacapa_metallib ALL DEPENDS ${METALLIB_OUTPUT})

# ---------------------------------------------------------------------------
# Make the metallib path available as a compile definition so
# MetalContext::create() knows where to find it at runtime.
# ---------------------------------------------------------------------------
target_compile_definitions(anacapa_lib PUBLIC
    ANACAPA_METALLIB_PATH="${METALLIB_OUTPUT}"
)

# ---------------------------------------------------------------------------
# Objective-C++ sources for the Metal backend
# ---------------------------------------------------------------------------
target_sources(anacapa_lib PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/MetalContext.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/MetalBuffer.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/MetalAccelStructure.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/MetalPathIntegrator.mm"
)

# .mm files need ObjC ARC
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/MetalContext.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/MetalBuffer.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/MetalAccelStructure.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/MetalPathIntegrator.mm"
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

# Include path so .mm files can find their own headers without prefix
target_include_directories(anacapa_lib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Ensure metallib is built before the library
add_dependencies(anacapa_lib anacapa_metallib)
