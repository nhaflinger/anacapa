#pragma once

#include <string>

namespace anacapa {

// ---------------------------------------------------------------------------
// IBackend — abstract GPU backend
//
// Implemented by MetalBackend (macOS/Apple Silicon) and CUDABackend (NVIDIA).
// Each backend owns device/queue/shader-library lifetime.
// ---------------------------------------------------------------------------
class IBackend {
public:
    virtual ~IBackend() = default;

    virtual bool        isValid()     const = 0;
    virtual std::string name()        const = 0;   // e.g. "Apple M1 Pro"
    virtual std::string backendType() const = 0;   // "Metal" | "CUDA"
};

} // namespace anacapa
