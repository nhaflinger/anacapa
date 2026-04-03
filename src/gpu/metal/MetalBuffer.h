#pragma once

// Pure C++ header — no Objective-C types.
// MetalBuffer<T> is a typed RAII wrapper around a MTLBuffer allocated in
// MTLStorageModeShared (CPU + GPU accessible on Apple Silicon).

#include <cstddef>
#include <cstring>
#include <cassert>
#include <vector>

namespace anacapa {

// Forward-declared; resolved only in .mm translation units.
void* metalBufferCreate(void* device, size_t byteSize);
void  metalBufferRelease(void* buffer);
void* metalBufferContents(void* buffer);

// ---------------------------------------------------------------------------
// MetalBuffer<T> — RAII MTLBuffer with typed CPU-side access
// ---------------------------------------------------------------------------
template<typename T>
class MetalBuffer {
public:
    MetalBuffer() = default;

    // Allocate a buffer for `count` elements of type T.
    // device: id<MTLDevice> (passed as void* to keep header ObjC-free)
    MetalBuffer(void* device, size_t count)
        : m_count(count)
    {
        assert(count > 0);
        m_handle = metalBufferCreate(device, count * sizeof(T));
    }

    ~MetalBuffer() {
        if (m_handle) metalBufferRelease(m_handle);
    }

    // Non-copyable
    MetalBuffer(const MetalBuffer&) = delete;
    MetalBuffer& operator=(const MetalBuffer&) = delete;

    // Movable
    MetalBuffer(MetalBuffer&& o) noexcept
        : m_handle(o.m_handle), m_count(o.m_count)
    { o.m_handle = nullptr; o.m_count = 0; }

    MetalBuffer& operator=(MetalBuffer&& o) noexcept {
        if (this != &o) {
            if (m_handle) metalBufferRelease(m_handle);
            m_handle = o.m_handle; m_count = o.m_count;
            o.m_handle = nullptr;  o.m_count = 0;
        }
        return *this;
    }

    bool   isValid() const { return m_handle != nullptr; }
    size_t count()   const { return m_count; }
    size_t byteSize() const { return m_count * sizeof(T); }

    // Raw MTLBuffer handle — used in .mm files to bind to command encoders
    void* handle() const { return m_handle; }

    // CPU-side typed pointer (MTLStorageModeShared — always valid)
    T*       data()       { return static_cast<T*>(metalBufferContents(m_handle)); }
    const T* data() const { return static_cast<const T*>(metalBufferContents(m_handle)); }

    T&       operator[](size_t i)       { return data()[i]; }
    const T& operator[](size_t i) const { return data()[i]; }

    // Bulk upload from a std::vector
    void upload(const std::vector<T>& src) {
        assert(src.size() <= m_count);
        std::memcpy(data(), src.data(), src.size() * sizeof(T));
    }

    // Bulk download into a std::vector
    void download(std::vector<T>& dst) const {
        dst.resize(m_count);
        std::memcpy(dst.data(), data(), m_count * sizeof(T));
    }

private:
    void*  m_handle = nullptr;
    size_t m_count  = 0;
};

} // namespace anacapa
