//  RayGen.metal — camera ray generation kernel
//
//  One thread per pixel.  Generates a primary ray into the wavefront ray
//  buffer.  PCG random jitter is applied for anti-aliasing.
//
//  Dispatch: [imageWidth, imageHeight, 1] threads, threadgroup [8, 8, 1].

#include <metal_stdlib>
#include "SharedTypes.h"

using namespace metal;

// ---------------------------------------------------------------------------
// PCG hash — fast, good quality 32-bit random from two seeds
// ---------------------------------------------------------------------------
static uint pcg(uint state) {
    uint s = state * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

static float randFloat(thread uint& state) {
    state = pcg(state);
    return float(state) * (1.0f / 4294967296.0f);  // [0, 1)
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
kernel void rayGen(
    constant GpuCameraParams& cam        [[ buffer(0) ]],
    device   GpuRay*          rays       [[ buffer(1) ]],
    uint2                     gid        [[ thread_position_in_grid ]])
{
    uint px = gid.x;
    uint py = gid.y;

    if (px >= cam.imageWidth || py >= cam.imageHeight) return;

    uint pixelIdx = py * cam.imageWidth + px;

    // PCG seed — unique per pixel, deterministic
    uint rngState = pcg(pixelIdx ^ 0xdeadbeef);

    // Sub-pixel jitter for anti-aliasing (single sample — outer loop drives spp)
    float jx = randFloat(rngState);
    float jy = randFloat(rngState);

    float u = (float(px) + jx) / float(cam.imageWidth);
    float v = (float(py) + jy) / float(cam.imageHeight);

    // Construct ray in world space
    float3 origin = float3(cam.origin.x, cam.origin.y, cam.origin.z);
    float3 horiz  = float3(cam.horizontal.x, cam.horizontal.y, cam.horizontal.z);
    float3 vert   = float3(cam.vertical.x,   cam.vertical.y,   cam.vertical.z);
    float3 ll     = float3(cam.lowerLeft.x,  cam.lowerLeft.y,  cam.lowerLeft.z);

    float3 dir = normalize(ll + u * horiz + v * vert - origin);

    GpuRay ray;
    ray.origin.x   = origin.x; ray.origin.y   = origin.y; ray.origin.z   = origin.z;
    ray.direction.x = dir.x;   ray.direction.y = dir.y;   ray.direction.z = dir.z;
    ray.tMin       = 1e-4f;
    ray.tMax       = 1e10f;
    ray.pixelIdx   = pixelIdx;
    ray.sampleIdx  = 0;
    ray.bounce     = 0;
    ray._pad       = 0;

    rays[pixelIdx] = ray;
}
