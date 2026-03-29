#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

namespace anacapa {

// ---------------------------------------------------------------------------
// ArenaAllocator
//
// A bump-pointer allocator backed by a single contiguous block.
// All allocations are O(1). There is no per-object free — reset() reclaims
// the entire arena at once.
//
// Design notes:
//   - Per-tile arenas are constructed on the stack, reset between tiles.
//   - Maps cleanly to per-block shared memory on GPU (Phase 5).
//   - Not thread-safe — one arena per tile worker thread.
// ---------------------------------------------------------------------------
class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t capacityBytes)
        : m_capacity(capacityBytes)
        , m_offset(0)
    {
        m_base = static_cast<std::byte*>(
            ::operator new(capacityBytes, std::align_val_t{64}));
    }

    ~ArenaAllocator() {
        ::operator delete(m_base, std::align_val_t{64});
    }

    // Non-copyable, movable
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    ArenaAllocator(ArenaAllocator&& o) noexcept
        : m_base(o.m_base), m_capacity(o.m_capacity), m_offset(o.m_offset)
    {
        o.m_base = nullptr;
        o.m_capacity = 0;
        o.m_offset = 0;
    }

    // Allocate `bytes` bytes aligned to `align` (must be power of two).
    // Returns nullptr if capacity is exhausted.
    void* alloc(size_t bytes, size_t align = alignof(std::max_align_t)) {
        assert((align & (align - 1)) == 0 && "align must be a power of two");

        // Align current offset
        size_t aligned = (m_offset + align - 1) & ~(align - 1);
        if (aligned + bytes > m_capacity)
            return nullptr;

        void* ptr = m_base + aligned;
        m_offset = aligned + bytes;
        return ptr;
    }

    // Typed allocation — constructs T with placement new
    template<typename T, typename... Args>
    T* create(Args&&... args) {
        void* mem = alloc(sizeof(T), alignof(T));
        assert(mem && "ArenaAllocator exhausted");
        return new (mem) T(std::forward<Args>(args)...);
    }

    // Typed array allocation — no construction, raw storage
    template<typename T>
    T* allocArray(size_t count) {
        void* mem = alloc(sizeof(T) * count, alignof(T));
        assert(mem && "ArenaAllocator exhausted");
        return static_cast<T*>(mem);
    }

    // Reset — reclaims all memory in O(1). Does NOT call destructors.
    // Only use when you know all objects in the arena are trivially destructible
    // or have already been manually destroyed.
    void reset() { m_offset = 0; }

    // Diagnostics
    size_t used()      const { return m_offset; }
    size_t capacity()  const { return m_capacity; }
    size_t remaining() const { return m_capacity - m_offset; }

    // Save/restore a watermark — useful for temporary sub-allocations
    // within a tile that should be freed before the next loop iteration.
    size_t mark()                   const { return m_offset; }
    void   rewind(size_t watermark)       { assert(watermark <= m_offset); m_offset = watermark; }

private:
    std::byte* m_base     = nullptr;
    size_t     m_capacity = 0;
    size_t     m_offset   = 0;
};

// ---------------------------------------------------------------------------
// PoolAllocator<T>
//
// Fixed-size object pool for homogeneous allocations (e.g., path vertices
// of a single type during subpath tracing). Free list–based, O(1) alloc/free.
// ---------------------------------------------------------------------------
template<typename T>
class PoolAllocator {
public:
    explicit PoolAllocator(size_t capacity)
        : m_capacity(capacity)
    {
        m_storage = static_cast<T*>(
            ::operator new(sizeof(T) * capacity, std::align_val_t{alignof(T)}));
        // Build free list
        m_freeList = nullptr;
        for (size_t i = 0; i < capacity; ++i) {
            Node* node = reinterpret_cast<Node*>(&m_storage[i]);
            node->next = m_freeList;
            m_freeList = node;
        }
    }

    ~PoolAllocator() {
        ::operator delete(m_storage, std::align_val_t{alignof(T)});
    }

    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    template<typename... Args>
    T* acquire(Args&&... args) {
        assert(m_freeList && "PoolAllocator exhausted");
        Node* node = m_freeList;
        m_freeList = node->next;
        return new (node) T(std::forward<Args>(args)...);
    }

    void release(T* obj) {
        obj->~T();
        Node* node = reinterpret_cast<Node*>(obj);
        node->next = m_freeList;
        m_freeList = node;
    }

    size_t capacity() const { return m_capacity; }

private:
    struct Node { Node* next; };

    T*     m_storage  = nullptr;
    Node*  m_freeList = nullptr;
    size_t m_capacity = 0;
};

} // namespace anacapa
