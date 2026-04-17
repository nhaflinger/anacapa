#ifdef ANACAPA_ENABLE_CUDA

#include "CudaContext.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_CHECK(call) do {                                               \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "[error] CUDA %d at %s:%d: %s\n",                  \
            int(_e), __FILE__, __LINE__, cudaGetErrorString(_e));           \
    }                                                                       \
} while(0)

namespace anacapa {

struct CudaContext::Impl {
    CUstream    stream = nullptr;
    std::string deviceName;
    bool        valid  = false;
};

CudaContext::CudaContext() : m_impl(std::make_unique<Impl>()) {}
CudaContext::~CudaContext() {
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

    ctx->m_impl->valid = true;
    printf("[info]  CudaContext: initialized on '%s'\n", ctx->m_impl->deviceName.c_str());
    return ctx;
}

bool        CudaContext::isValid()  const { return m_impl->valid; }
std::string CudaContext::name()     const { return m_impl->deviceName; }
void*       CudaContext::cuStream() const { return m_impl->stream; }

} // namespace anacapa

#endif // ANACAPA_ENABLE_CUDA
