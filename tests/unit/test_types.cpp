#include <anacapa/core/Types.h>
#include <gtest/gtest.h>
#include <cmath>

using namespace anacapa;

TEST(Vec3f, BasicArithmetic) {
    Vec3f a{1, 2, 3}, b{4, 5, 6};
    Vec3f sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 5.f);
    EXPECT_FLOAT_EQ(sum.y, 7.f);
    EXPECT_FLOAT_EQ(sum.z, 9.f);
}

TEST(Vec3f, DotProduct) {
    Vec3f a{1, 0, 0}, b{0, 1, 0};
    EXPECT_FLOAT_EQ(dot(a, b), 0.f);
    EXPECT_FLOAT_EQ(dot(a, a), 1.f);
}

TEST(Vec3f, CrossProduct) {
    Vec3f x{1, 0, 0}, y{0, 1, 0};
    Vec3f z = cross(x, y);
    EXPECT_FLOAT_EQ(z.x, 0.f);
    EXPECT_FLOAT_EQ(z.y, 0.f);
    EXPECT_FLOAT_EQ(z.z, 1.f);
}

TEST(Vec3f, Normalize) {
    Vec3f v{3, 0, 0};
    Vec3f n = normalize(v);
    EXPECT_NEAR(n.length(), 1.f, 1e-6f);
    EXPECT_FLOAT_EQ(n.x, 1.f);
}

TEST(Vec3f, OrthonormalBasis) {
    Vec3f n{0, 1, 0};
    Vec3f t, bt;
    buildOrthonormalBasis(n, t, bt);

    EXPECT_NEAR(dot(n, t),  0.f, 1e-6f);
    EXPECT_NEAR(dot(n, bt), 0.f, 1e-6f);
    EXPECT_NEAR(dot(t, bt), 0.f, 1e-6f);
    EXPECT_NEAR(t.length(),  1.f, 1e-6f);
    EXPECT_NEAR(bt.length(), 1.f, 1e-6f);
}

TEST(BBox3f, ExpandAndCentroid) {
    BBox3f b;
    b.expand({0,0,0});
    b.expand({2,4,6});
    Vec3f c = b.centroid();
    EXPECT_FLOAT_EQ(c.x, 1.f);
    EXPECT_FLOAT_EQ(c.y, 2.f);
    EXPECT_FLOAT_EQ(c.z, 3.f);
}

TEST(Ray, SpawnRayAvoidsSelfIntersection) {
    Vec3f origin{0,0,0}, normal{0,1,0}, dir{0,1,0};
    Ray r = spawnRay(origin, normal, dir);
    // Origin should be offset in the normal direction
    EXPECT_GT(r.origin.y, 0.f);
}
