#pragma once

// Pure C++ header — no Objective-C types exposed.
// Objective-C Metal objects live exclusively in MetalContext.mm (PIMPL).

#include <anacapa/gpu/IBackend.h>
#include <memory>
#include <string>

namespace anacapa {

// ---------------------------------------------------------------------------
// MetalContext — owns MTLDevice, MTLCommandQueue, and the compiled
// MTLLibrary (anacapa.metallib built at CMake time).
//
// Create with MetalContext::create().  Returns nullptr if Metal is
// unavailable (i.e. not running on Apple Silicon / macOS).
// ---------------------------------------------------------------------------
class MetalContext : public IBackend {
public:
    // Returns true if a Metal device is available on this machine.
    static bool isAvailable();

    // Factory — returns nullptr on failure (device not found, library missing).
    // metallibPath: absolute path to the compiled anacapa.metallib file.
    static std::unique_ptr<MetalContext> create(const std::string& metallibPath);

    ~MetalContext() override;

    // IBackend
    bool        isValid()     const override;
    std::string name()        const override;   // MTLDevice.name
    std::string backendType() const override { return "Metal"; }

    // Opaque accessors used by Metal-only translation units (.mm files).
    // Callers must cast the returned void* to the appropriate id<MTL...> type.
    void* device()       const;  // id<MTLDevice>
    void* commandQueue() const;  // id<MTLCommandQueue>
    void* library()      const;  // id<MTLLibrary>

private:
    MetalContext();

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
