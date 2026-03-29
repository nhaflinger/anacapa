#include <anacapa/accel/GeometryPool.h>
#include <gtest/gtest.h>
#include "../../src/accel/BVHBackend.h"

using namespace anacapa;

// Build a single triangle and verify ray-triangle intersection
TEST(BVH, SingleTriangleHit) {
    GeometryPool pool;
    MeshDesc mesh;
    mesh.positions = {{-1,0,0}, {1,0,0}, {0,1,0}};
    mesh.normals   = {{0,0,1}, {0,0,1}, {0,0,1}};
    mesh.uvs       = {{0,0},{1,0},{0.5f,1}};
    mesh.indices   = {0, 1, 2};
    pool.addMesh(std::move(mesh));

    BVHBackend bvh(pool);
    bvh.commit();

    // Ray pointing straight at the triangle
    Ray r{{0, 0.4f, -1}, {0, 0, 1}};
    auto result = bvh.trace(r);
    EXPECT_TRUE(result.hit);
    EXPECT_NEAR(result.si.t, 1.f, 1e-4f);
}

TEST(BVH, SingleTriangleMiss) {
    GeometryPool pool;
    MeshDesc mesh;
    mesh.positions = {{-1,0,0},{1,0,0},{0,1,0}};
    mesh.normals   = {{0,0,1},{0,0,1},{0,0,1}};
    mesh.indices   = {0,1,2};
    pool.addMesh(std::move(mesh));

    BVHBackend bvh(pool);
    bvh.commit();

    // Ray offset to miss
    Ray r{{5, 5, -1}, {0, 0, 1}};
    auto result = bvh.trace(r);
    EXPECT_FALSE(result.hit);
}

TEST(BVH, OcclusionTest) {
    GeometryPool pool;
    MeshDesc mesh;
    mesh.positions = {{-1,0,0},{1,0,0},{0,1,0}};
    mesh.normals   = {{0,0,1},{0,0,1},{0,0,1}};
    mesh.indices   = {0,1,2};
    pool.addMesh(std::move(mesh));

    BVHBackend bvh(pool);
    bvh.commit();

    Ray r{{0, 0.3f, -2}, {0, 0, 1}, 1e-4f, 5.f};
    EXPECT_TRUE(bvh.occluded(r));

    Ray rMiss{{5, 5, -2}, {0, 0, 1}, 1e-4f, 5.f};
    EXPECT_FALSE(bvh.occluded(rMiss));
}

TEST(BVH, MultipleTriangles) {
    GeometryPool pool;
    MeshDesc mesh;

    // Two triangles side by side
    mesh.positions = {
        {-2,0,0},{-0.1f,0,0},{-1,1,0},  // left tri
        {0.1f,0,0},{2,0,0},{1,1,0}       // right tri
    };
    mesh.normals = std::vector<Vec3f>(6, {0,0,1});
    mesh.indices = {0,1,2, 3,4,5};
    pool.addMesh(std::move(mesh));

    BVHBackend bvh(pool);
    bvh.commit();

    // Hit left triangle
    Ray rLeft{{-1, 0.2f, -1}, {0,0,1}};
    EXPECT_TRUE(bvh.trace(rLeft).hit);

    // Hit right triangle
    Ray rRight{{1, 0.2f, -1}, {0,0,1}};
    EXPECT_TRUE(bvh.trace(rRight).hit);

    // Miss both
    Ray rMiss{{0, 0.5f, -1}, {0,0,1}};
    EXPECT_FALSE(bvh.trace(rMiss).hit);
}
