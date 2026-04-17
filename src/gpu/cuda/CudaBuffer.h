#pragma once

// Pure C++ header — RAII wrapper for device memory (CUdeviceptr / cudaMalloc).
// Mirrors MetalBuffer.h. Include only from .cu translation units (nvcc).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// CudaBuffer<T> — typed RAII device memory buffer
// ---------------------------------------------------------------------------
template<typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) : m_count(count) {
        assert(count > 0);
        cudaMalloc(&m_ptr, count * sizeof(T));
    }

    ~CudaBuffer() { if (m_ptr) cudaFree(m_ptr); }

    CudaBuffer(const CudaBuffer&)            = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& o) noexcept : m_ptr(o.m_ptr), m_count(o.m_count) {
        o.m_ptr = nullptr; o.m_count = 0;
    }
    CudaBuffer& operator=(CudaBuffer&& o) noexcept {
        if (this != &o) {
            if (m_ptr) cudaFree(m_ptr);
            m_ptr = o.m_ptr; m_count = o.m_count;
            o.m_ptr = nullptr; o.m_count = 0;
        }
        return *this;
    }

    bool   isValid()  const { return m_ptr != nullptr; }
    size_t count()    const { return m_count; }
    size_t byteSize() const { return m_count * sizeof(T); }

    // Raw device pointer
    T*          ptr()       const { return m_ptr; }
    CUdeviceptr devPtr()    const { return reinterpret_cast<CUdeviceptr>(m_ptr); }

    void upload(const std::vector<T>& src) {
        assert(src.size() <= m_count);
        cudaMemcpy(m_ptr, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    void upload(const T* src, size_t count) {
        assert(count <= m_count);
        cudaMemcpy(m_ptr, src, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    void download(std::vector<T>& dst) const {
        dst.resize(m_count);
        cudaMemcpy(dst.data(), m_ptr, m_count * sizeof(T), cudaMemcpyDeviceToHost);
    }

    void zero() { cudaMemset(m_ptr, 0, m_count * sizeof(T)); }

private:
    T*     m_ptr   = nullptr;
    size_t m_count = 0;
};

// Convenience: raw byte buffer
using CudaByteBuffer = CudaBuffer<uint8_t>;

} // namespace anacapa
