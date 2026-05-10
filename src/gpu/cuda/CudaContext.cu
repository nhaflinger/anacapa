#ifdef ANACAPA_ENABLE_CUDA

#include "CudaContext.h"

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef ANACAPA_ENABLE_OPTIX
#include <optix.h>
#include <optix_stubs.h>
// Defines the global g_optixFunctionTable. Exactly one TU may include this.
#include <optix_function_table_definition.h>
#endif

#include <cstdio>

#define CUDA_CHECK(call) do {                                               \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "[error] CUDA %d at %s:%d: %s\n",                  \
            int(_e), __FILE__, __LINE__, cudaGetErrorString(_e));           \
    }                                                                       \
} while(0)

#ifdef ANACAPA_ENABLE_OPTIX
#define OPTIX_CHECK(call) do {                                              \
    OptixResult _r = (call);                                                \
    if (_r != OPTIX_SUCCESS) {                                              \
        fprintf(stderr, "[error] OptiX %d (%s) at %s:%d\n",                 \
            int(_r), optixGetErrorName(_r), __FILE__, __LINE__);            \
    }                                                                       \
} while(0)
#endif

namespace anacapa {

#ifdef ANACAPA_ENABLE_OPTIX
// OptiX log levels: 1=fatal, 2=error, 3=warning, 4=print/info.
static void optixLogCallback(unsigned int level, const char* tag,
                             const char* message, void* /*cbdata*/) {
    const char* ll = (level == 1) ? "fatal"
                   : (level == 2) ? "error"
                   : (level == 3) ? "warn"
                                  : "info";
    fprintf(stderr, "[OptiX %s] %s: %s\n",
            ll, tag ? tag : "", message ? message : "");
}
#endif

struct CudaContext::Impl {
    CUstream    stream = nullptr;
    std::string deviceName;
    bool        valid  = false;
#ifdef ANACAPA_ENABLE_OPTIX
    OptixDeviceContext optix = nullptr;
#endif
};

CudaContext::CudaContext() : m_impl(std::make_unique<Impl>()) {}
CudaContext::~CudaContext() {
#ifdef ANACAPA_ENABLE_OPTIX
    if (m_impl->optix) optixDeviceContextDestroy(m_impl->optix);
#endif
    if (m_impl->stream) cudaStreamDestroy(m_impl->stream);
}

bool CudaContext::isAvailable() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count > 0;
}

std::unique_ptr<CudaContext> CudaContext::create() {
    auto ctx = std::unique_ptr<CudaContext>(new CudaContext());

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        fprintf(stderr, "[error] CudaContext: no CUDA devices found\n");
        return nullptr;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    ctx->m_impl->deviceName = prop.name;

    CUDA_CHECK(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&ctx->m_impl->stream)));

#ifdef ANACAPA_ENABLE_OPTIX
    OptixResult oxInit = optixInit();
    if (oxInit != OPTIX_SUCCESS) {
        fprintf(stderr, "[error] CudaContext: optixInit failed (%d, %s)\n",
                int(oxInit), optixGetErrorName(oxInit));
        return nullptr;
    }
    OptixDeviceContextOptions opts{};
    opts.logCallbackFunction = &optixLogCallback;
    opts.logCallbackLevel    = 4;  // info and below
    // Validation mode adds runtime checks (SBT, AS sanity, etc.) at the cost
    // of throughput.  Turn on for development; off in release.
#ifndef NDEBUG
    opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OptixResult oxCtx = optixDeviceContextCreate(/*cuCtx=*/0, &opts,
                                                 &ctx->m_impl->optix);
    if (oxCtx != OPTIX_SUCCESS) {
        fprintf(stderr, "[error] CudaContext: optixDeviceContextCreate failed "
                        "(%d, %s)\n", int(oxCtx), optixGetErrorName(oxCtx));
        return nullptr;
    }
    printf("[info]  CudaContext: OptiX device context created (SDK %d.%d.%d)\n",
           OPTIX_VERSION / 10000,
           (OPTIX_VERSION / 100) % 100,
           OPTIX_VERSION % 100);
#endif

    ctx->m_impl->valid = true;
    printf("[info]  CudaContext: initialized on '%s'\n", ctx->m_impl->deviceName.c_str());
    return ctx;
}

bool        CudaContext::isValid()      const { return m_impl->valid; }
std::string CudaContext::name()         const { return m_impl->deviceName; }
void*       CudaContext::cuStream()     const { return m_impl->stream; }
void*       CudaContext::optixContext() const {
#ifdef ANACAPA_ENABLE_OPTIX
    return m_impl->optix;
#else
    return nullptr;
#endif
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_CUDA
