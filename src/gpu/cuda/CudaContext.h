#pragma once

// Pure C++ header — no CUDA types exposed in the public interface.
// All CUDA objects live exclusively in CudaContext.cu (PIMPL).

#include <anacapa/gpu/IBackend.h>
#include <memory>
#include <string>

namespace anacapa {

// ---------------------------------------------------------------------------
// CudaContext — owns the CUDA device and stream.
//
// Create with CudaContext::create(). Returns nullptr if CUDA is
// unavailable or initialization fails.
// ---------------------------------------------------------------------------
class CudaContext : public IBackend {
public:
    static bool isAvailable();

    // Factory — returns nullptr on failure.
    static std::unique_ptr<CudaContext> create();

    ~CudaContext() override;

    // IBackend
    bool        isValid()     const override;
    std::string name()        const override;  // e.g. "NVIDIA RTX A400"
    std::string backendType() const override { return "CUDA"; }

    // Opaque accessor used by CUDA translation units.
    // Caller casts the returned void* to CUstream / cudaStream_t.
    void* cuStream() const;

private:
    CudaContext();

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
