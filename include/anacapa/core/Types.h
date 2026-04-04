#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace anacapa {

// ---------------------------------------------------------------------------
// Vec2f
// ---------------------------------------------------------------------------
struct Vec2f {
    float x = 0.f, y = 0.f;

    Vec2f() = default;
    constexpr Vec2f(float x, float y) : x(x), y(y) {}
    explicit constexpr Vec2f(float v) : x(v), y(v) {}

    Vec2f operator+(Vec2f o) const { return {x + o.x, y + o.y}; }
    Vec2f operator-(Vec2f o) const { return {x - o.x, y - o.y}; }
    Vec2f operator*(float s) const { return {x * s, y * s}; }
    Vec2f operator/(float s) const { return {x / s, y / s}; }

    float& operator[](int i) { return (&x)[i]; }
    float  operator[](int i) const { return (&x)[i]; }
};

// ---------------------------------------------------------------------------
// Vec3f
// ---------------------------------------------------------------------------
struct alignas(16) Vec3f {
    float x = 0.f, y = 0.f, z = 0.f;

    Vec3f() = default;
    constexpr Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3f(float v) : x(v), y(v), z(v) {}

    Vec3f operator+(Vec3f o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3f operator-(Vec3f o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3f operator*(Vec3f o) const { return {x * o.x, y * o.y, z * o.z}; }
    Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3f operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3f operator-()         const { return {-x, -y, -z}; }

    Vec3f& operator+=(Vec3f o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3f& operator*=(float s) { x *= s;   y *= s;   z *= s;   return *this; }
    Vec3f& operator*=(Vec3f o) { x *= o.x; y *= o.y; z *= o.z; return *this; }

    bool operator==(Vec3f o) const { return x == o.x && y == o.y && z == o.z; }

    float& operator[](int i) { return (&x)[i]; }
    float  operator[](int i) const { return (&x)[i]; }

    float lengthSq() const { return x*x + y*y + z*z; }
    float length()   const { return std::sqrt(lengthSq()); }

    bool isZero()    const { return x == 0.f && y == 0.f && z == 0.f; }
    bool hasNaN()    const { return std::isnan(x) || std::isnan(y) || std::isnan(z); }
    bool hasInf()    const { return std::isinf(x) || std::isinf(y) || std::isinf(z); }
    bool isFinite()  const { return !hasNaN() && !hasInf(); }

    float maxComponent() const { return std::max({x, y, z}); }
    float minComponent() const { return std::min({x, y, z}); }
};

inline Vec3f operator*(float s, Vec3f v) { return v * s; }

inline float dot(Vec3f a, Vec3f b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline Vec3f cross(Vec3f a, Vec3f b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}

inline Vec3f normalize(Vec3f v) {
    float len = v.length();
    assert(len > 0.f);
    return v * (1.f / len);
}

inline Vec3f safeNormalize(Vec3f v, Vec3f fallback = {0,0,1}) {
    float len = v.length();
    return len > 1e-8f ? v * (1.f / len) : fallback;
}

inline float absDot(Vec3f a, Vec3f b) { return std::abs(dot(a, b)); }

inline Vec3f lerp(Vec3f a, Vec3f b, float t) { return a + (b - a) * t; }

inline Vec3f abs(Vec3f v) {
    return {std::abs(v.x), std::abs(v.y), std::abs(v.z)};
}

inline Vec3f min(Vec3f a, Vec3f b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

inline Vec3f max(Vec3f a, Vec3f b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

// Orthonormal basis from a single normal (Frisvad / Duff et al. 2017)
inline void buildOrthonormalBasis(Vec3f n, Vec3f& t, Vec3f& bt) {
    float sign = std::copysign(1.f, n.z);
    float a    = -1.f / (sign + n.z);
    float b    = n.x * n.y * a;
    t  = Vec3f{1.f + sign * n.x * n.x * a, sign * b, -sign * n.x};
    bt = Vec3f{b, sign + n.y * n.y * a, -n.y};
}

// ---------------------------------------------------------------------------
// Vec4f
// ---------------------------------------------------------------------------
struct alignas(16) Vec4f {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;

    Vec4f() = default;
    constexpr Vec4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    constexpr Vec4f(Vec3f v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}

    Vec3f xyz() const { return {x, y, z}; }
};

// ---------------------------------------------------------------------------
// Mat4f — row-major 4x4 matrix
// ---------------------------------------------------------------------------
struct Mat4f {
    float m[4][4] = {};

    static Mat4f identity() {
        Mat4f r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.f;
        return r;
    }

    Vec4f row(int i) const { return {m[i][0], m[i][1], m[i][2], m[i][3]}; }
    Vec4f col(int j) const { return {m[0][j], m[1][j], m[2][j], m[3][j]}; }

    Vec3f transformPoint(Vec3f p) const {
        float x = m[0][0]*p.x + m[0][1]*p.y + m[0][2]*p.z + m[0][3];
        float y = m[1][0]*p.x + m[1][1]*p.y + m[1][2]*p.z + m[1][3];
        float z = m[2][0]*p.x + m[2][1]*p.y + m[2][2]*p.z + m[2][3];
        float w = m[3][0]*p.x + m[3][1]*p.y + m[3][2]*p.z + m[3][3];
        return {x/w, y/w, z/w};
    }

    Vec3f transformVector(Vec3f v) const {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        };
    }

    Vec3f transformNormal(Vec3f n) const {
        // Normals transform by the inverse-transpose
        return {
            m[0][0]*n.x + m[1][0]*n.y + m[2][0]*n.z,
            m[0][1]*n.x + m[1][1]*n.y + m[2][1]*n.z,
            m[0][2]*n.x + m[1][2]*n.y + m[2][2]*n.z
        };
    }

    Mat4f operator*(const Mat4f& o) const {
        Mat4f r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    r.m[i][j] += m[i][k] * o.m[k][j];
        return r;
    }

    Mat4f transposed() const {
        Mat4f r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = m[j][i];
        return r;
    }

    // Linear interpolation between two matrices (used for motion blur).
    // Correct for translation and scale; approximate for rotation
    // (sufficient for small angular velocities between shutter samples).
    static Mat4f lerp(const Mat4f& a, const Mat4f& b, float t) {
        Mat4f r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = a.m[i][j] * (1.f - t) + b.m[i][j] * t;
        return r;
    }

    // Gauss-Jordan elimination. Returns identity if matrix is singular.
    Mat4f inverse() const {
        float a[4][8];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) a[i][j]     = m[i][j];
            for (int j = 0; j < 4; ++j) a[i][j + 4] = (i == j) ? 1.f : 0.f;
        }
        for (int col = 0; col < 4; ++col) {
            // Partial pivoting
            int pivot = col;
            for (int row = col + 1; row < 4; ++row)
                if (std::abs(a[row][col]) > std::abs(a[pivot][col]))
                    pivot = row;
            if (pivot != col)
                for (int j = 0; j < 8; ++j)
                    std::swap(a[col][j], a[pivot][j]);
            float diag = a[col][col];
            if (std::abs(diag) < 1e-10f) return identity(); // singular
            float invDiag = 1.f / diag;
            for (int j = 0; j < 8; ++j) a[col][j] *= invDiag;
            for (int row = 0; row < 4; ++row) {
                if (row == col) continue;
                float factor = a[row][col];
                for (int j = 0; j < 8; ++j)
                    a[row][j] -= factor * a[col][j];
            }
        }
        Mat4f r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = a[i][j + 4];
        return r;
    }
};

// ---------------------------------------------------------------------------
// Spectrum — RGB in ACEScg working space
// Alias allows future swap to full spectral without changing algorithm code.
// ---------------------------------------------------------------------------
using Spectrum = Vec3f;

inline Spectrum makeSpectrum(float r, float g, float b) { return {r, g, b}; }
inline Spectrum makeSpectrum(float v)                    { return Vec3f{v}; }

inline bool isBlack(Spectrum s) {
    return s.x == 0.f && s.y == 0.f && s.z == 0.f;
}

inline float luminance(Spectrum s) {
    // ACEScg luminance coefficients
    return 0.2722287f*s.x + 0.6740818f*s.y + 0.0536895f*s.z;
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------
struct Ray {
    Vec3f  origin;
    Vec3f  direction;   // Must be normalized
    float  tMin  = 1e-4f;
    float  tMax  = std::numeric_limits<float>::infinity();
    uint32_t depth = 0;
    float  time  = 0.f; // Normalized shutter time in [0, 1]; 0 = shutter open

    Ray() = default;
    Ray(Vec3f o, Vec3f d, float tMin = 1e-4f, float tMax = std::numeric_limits<float>::infinity())
        : origin(o), direction(d), tMin(tMin), tMax(tMax) {}

    Vec3f at(float t) const { return origin + direction * t; }
};

// Spawn a ray offset along the normal to avoid self-intersection
inline Ray spawnRay(Vec3f origin, Vec3f normal, Vec3f direction) {
    // Offset along normal to clear the surface
    Vec3f offset = normal * 1e-4f;
    // Flip offset if direction points into the surface
    if (dot(direction, normal) < 0.f) offset = -offset;
    return Ray{origin + offset, direction};
}

inline Ray spawnRayTo(Vec3f origin, Vec3f normal, Vec3f target) {
    Vec3f dir = target - origin;
    float dist = dir.length();
    dir = dir * (1.f / dist);
    Ray r = spawnRay(origin, normal, dir);
    r.tMax = dist * (1.f - 1e-4f);
    return r;
}

// ---------------------------------------------------------------------------
// BBox3f — axis-aligned bounding box
// ---------------------------------------------------------------------------
struct BBox3f {
    Vec3f pMin = Vec3f{ std::numeric_limits<float>::infinity()};
    Vec3f pMax = Vec3f{-std::numeric_limits<float>::infinity()};

    BBox3f() = default;
    BBox3f(Vec3f pMin, Vec3f pMax) : pMin(pMin), pMax(pMax) {}

    void expand(Vec3f p) {
        pMin = min(pMin, p);
        pMax = max(pMax, p);
    }

    void expand(const BBox3f& o) {
        pMin = min(pMin, o.pMin);
        pMax = max(pMax, o.pMax);
    }

    Vec3f centroid() const { return (pMin + pMax) * 0.5f; }
    Vec3f diagonal() const { return pMax - pMin; }

    bool valid() const {
        return pMin.x <= pMax.x && pMin.y <= pMax.y && pMin.z <= pMax.z;
    }

    bool contains(Vec3f p) const {
        return p.x >= pMin.x && p.x <= pMax.x &&
               p.y >= pMin.y && p.y <= pMax.y &&
               p.z >= pMin.z && p.z <= pMax.z;
    }
};

// ---------------------------------------------------------------------------
// Span<T> — non-owning view over a contiguous array
// ---------------------------------------------------------------------------
template<typename T>
struct Span {
    T*     data   = nullptr;
    size_t count  = 0;

    Span() = default;
    Span(T* data, size_t count) : data(data), count(count) {}

    template<typename Container>
    Span(Container& c) : data(c.data()), count(c.size()) {}

    T&     operator[](size_t i)       { assert(i < count); return data[i]; }
    T      operator[](size_t i) const { assert(i < count); return data[i]; }

    T*     begin() { return data; }
    T*     end()   { return data + count; }
    size_t size()  const { return count; }
    bool   empty() const { return count == 0; }
};

} // namespace anacapa
