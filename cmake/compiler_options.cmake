include(CheckCXXCompilerFlag)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(ANACAPA_ARCH_ARM ON)
else()
    set(ANACAPA_ARCH_X86 ON)
endif()

target_compile_options(anacapa_lib PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -fno-math-errno
        -fno-trapping-math
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /fp:fast
    >
)

# Architecture-specific SIMD
if(ANACAPA_ARCH_ARM)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        target_compile_options(anacapa_lib PRIVATE -march=native)
    endif()
elseif(ANACAPA_ARCH_X86)
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
    if(HAS_AVX2)
        target_compile_options(anacapa_lib PRIVATE -mavx2 -mfma)
    else()
        check_cxx_compiler_flag("-msse4.2" HAS_SSE42)
        if(HAS_SSE42)
            target_compile_options(anacapa_lib PRIVATE -msse4.2)
        endif()
    endif()
endif()

# Sanitizers
if(ANACAPA_ENABLE_ASAN)
    target_compile_options(anacapa_lib PUBLIC -fsanitize=address,undefined)
    target_link_options(anacapa_lib    PUBLIC -fsanitize=address,undefined)
endif()

# Optimization tiers
target_compile_options(anacapa_lib PRIVATE
    $<$<CONFIG:Release>:        -O3 -DNDEBUG>
    $<$<CONFIG:Debug>:          -O0 -g -fno-inline>
    $<$<CONFIG:RelWithDebInfo>: -O2 -g -DNDEBUG>
)
