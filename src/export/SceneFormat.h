#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// Anacapa binary scene format — written by the Blender Python addon and read
// by SceneExporter to produce a USD file.
//
// File layout (all values little-endian):
//   SceneHeader
//   [MeshRecord] × num_meshes
//   [LightRecord] × num_lights
//   CameraRecord  (only present when SceneHeader::has_camera != 0)
//
// Variable-length strings are length-prefixed:
//   uint32_t  byte_count
//   char      data[byte_count]   (UTF-8, no null terminator)
//
// MeshRecord layout (immediately following one another):
//   MeshHeader               fixed fields (no matrix — see below)
//   float[num_motion_keys*16] world transform matrices, one per key (row-major, column-vector)
//   float[num_motion_keys]   USD time code for each key (frame number; 1 key = static)
//   string                   object name
//   string[num_mat_slots]    material slot names (packed sequentially)
//   uint32[num_mat_slots]    mat_flags  (bit 0 = translucent; bit 1 = sss; others reserved)
//   float[num_mat_slots * 3] mat_translucency_colors  (RGB, only meaningful when bit 0 set)
//   float[num_mat_slots * 9] mat_sss_data  (per slot: weight, color[3], radius[3], scale, anisotropy;
//                                           only meaningful when bit 1 set)
//   float[num_verts * 3]     vertex positions
//   float[num_loops * 3]     per-loop split normals (face-varying)
//   [UVLayer] × num_uvlayers
//     string                 layer name
//     float[num_loops * 2]   UVs (face-varying, matching loop order)
//   uint32[num_polys]        loop_start   (first loop index of each polygon)
//   uint32[num_polys]        loop_total   (vertex count of each polygon)
//   uint32[num_polys]        mat_index    (material slot index per polygon)
//   uint32[num_loops]        vert_index   (vertex index per loop — faceVertexIndices)
// ---------------------------------------------------------------------------

namespace anacapa {
namespace scene_fmt {

static constexpr uint32_t kMagic   = 0x41434E41u; // "ANCA"
static constexpr uint32_t kVersion = 4u;

// Coordinate convention written by the Python exporter:
//   Y-up, right-handed (Blender Z-up corrected to Y-up by the Python side).
// Matrices are written row-major (16 floats, row 0 first) in anacapa's
// column-vector convention — no further transpose needed on read.

#pragma pack(push, 1)

struct SceneHeader {
    uint32_t magic;       // kMagic
    uint32_t version;     // kVersion
    uint32_t num_meshes;
    uint32_t num_lights;
    uint32_t has_camera;  // 0 or 1
    uint32_t _pad[3];
};
static_assert(sizeof(SceneHeader) == 32, "SceneHeader must be 32 bytes");

struct MeshHeader {
    uint32_t num_verts;
    uint32_t num_loops;
    uint32_t num_polys;
    uint32_t num_uvlayers;
    uint32_t num_mat_slots;
    uint32_t num_motion_keys;  // ≥ 1; >1 enables motion blur (matrices + time codes follow)
};

enum class LightType : uint32_t {
    Distant = 0,
    Rect    = 1,
    Sphere  = 2,
    Disk    = 3,
};

struct LightRecord {
    uint32_t type;        // LightType
    float    matrix[16];  // world transform, row-major, column-vector convention
    float    color[3];
    float    intensity;
    uint32_t normalize;   // 1 = Blender normalize=true convention
    float    params[4];   // Distant: [angle_deg,0,0,0]
                          // Rect:    [0,width,height,0]
                          // Sphere:  [0,0,0,radius]
                          // Disk:    [0,0,0,radius]
    // followed by: string  name
};

struct CameraRecord {
    float    matrix[16];    // world transform, row-major, column-vector convention
    float    lens;          // focal length (mm)
    float    sensor_width;  // mm
    float    sensor_height; // mm
    float    clip_start;
    float    clip_end;
    float    dof_distance;
    float    dof_fstop;
};

#pragma pack(pop)

} // namespace scene_fmt
} // namespace anacapa
