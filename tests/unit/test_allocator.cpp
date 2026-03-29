#include <anacapa/core/Allocator.h>
#include <gtest/gtest.h>

using namespace anacapa;

TEST(ArenaAllocator, BasicAlloc) {
    ArenaAllocator arena(1024);
    void* p = arena.alloc(64, 16);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(arena.used(), 64u);
}

TEST(ArenaAllocator, AlignmentRespected) {
    ArenaAllocator arena(1024);
    void* p = arena.alloc(1, 1);      // 1 byte
    void* q = arena.alloc(4, 16);     // must be 16-byte aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(q) % 16, 0u);
    (void)p;
}

TEST(ArenaAllocator, ResetReclaims) {
    ArenaAllocator arena(256);
    arena.alloc(100);
    EXPECT_EQ(arena.used(), 100u);
    arena.reset();
    EXPECT_EQ(arena.used(), 0u);
}

TEST(ArenaAllocator, WatermarkRewind) {
    ArenaAllocator arena(512);
    arena.alloc(64);
    size_t mark = arena.mark();
    arena.alloc(128);
    EXPECT_EQ(arena.used(), 192u);
    arena.rewind(mark);
    EXPECT_EQ(arena.used(), 64u);
}

TEST(ArenaAllocator, ExhaustionReturnsNull) {
    ArenaAllocator arena(64);
    void* p = arena.alloc(48);
    EXPECT_NE(p, nullptr);
    void* q = arena.alloc(48);  // Would exceed capacity
    EXPECT_EQ(q, nullptr);
}

TEST(ArenaAllocator, Create) {
    struct Point { float x, y; };
    ArenaAllocator arena(256);
    Point* p = arena.create<Point>(1.f, 2.f);
    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p->x, 1.f);
    EXPECT_FLOAT_EQ(p->y, 2.f);
}
