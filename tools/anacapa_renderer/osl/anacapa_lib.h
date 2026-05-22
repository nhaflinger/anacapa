// anacapa_lib.h — Anacapa custom OSL shader library.
// Implements Blender texture nodes that have no MaterialX / OSL stdlib
// equivalent.  Injected into _mx_stdlib.h at OSL generation time so that
// every per-material .osl shader can call these functions.
//
// All functions follow the naming convention anacapa_<node>_<variant>().
// Integer mode parameters (feature, metric, depth …) are always baked in
// as literals by the Python exporter; only float/vector inputs are dynamic.

#ifndef ANACAPA_LIB_H
#define ANACAPA_LIB_H

// ===========================================================================
// Internal hash helpers
// ===========================================================================

// Three decorrelated cell-noise values for a given integer-lattice cell.
// cellnoise(point) returns a stable float in [0,1] keyed on floor(point).
void _acl_rand3(point cell,
                output float rx, output float ry, output float rz)
{
    rx = cellnoise(cell + vector(0.12345, 0.0,     0.0    ));
    ry = cellnoise(cell + vector(0.0,     0.67891,  0.0    ));
    rz = cellnoise(cell + vector(0.0,     0.0,      0.31416));
}

// ===========================================================================
// Voronoi texture
//
// feature  : 0=F1  1=F2  2=SmoothF1  3=DistanceToEdge  4=NSphereRadius
// metric   : 0=Euclidean  1=Manhattan  2=Chebychev  3=Minkowski(exponent)
// randomness: [0,1] — 0 = regular grid, 1 = fully random cell centres
// smoothness: kernel radius for SmoothF1 (feature==2)
// exponent : Minkowski p-value (metric==3)
// ===========================================================================

float _acl_vor_dist(vector d, int metric, float exponent)
{
    if (metric == 1)  // Manhattan
        return abs(d[0]) + abs(d[1]) + abs(d[2]);
    if (metric == 2)  // Chebychev
        return max(abs(d[0]), max(abs(d[1]), abs(d[2])));
    if (metric == 3) {  // Minkowski
        float e = max(exponent, 1e-4);
        return pow(pow(abs(d[0]), e) + pow(abs(d[1]), e) + pow(abs(d[2]), e),
                   1.0 / e);
    }
    // Euclidean (squared during search, then sqrt at output)
    return dot(d, d);
}

// Shared computation: fills F1, F2 distances and the winning cell.
void _acl_vor_search(point sp, float randomness,
                     int metric, float exponent,
                     output float f1, output float f2, output point win_cell)
{
    point  ip = floor(sp);
    vector fp = (vector)(sp - ip);
    f1 = 1e18;  f2 = 1e18;  win_cell = ip;

    for (int k = -1; k <= 1; ++k)
    for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i) {
        vector ofs  = vector(i, j, k);
        point  cell = ip + (point)ofs;
        float  rx, ry, rz;
        _acl_rand3(cell, rx, ry, rz);
        // Cell centre offset in [–0.5, 0.5] * randomness
        vector jitter = ofs - fp
                       + vector(rx - 0.5, ry - 0.5, rz - 0.5) * randomness;
        float d = _acl_vor_dist(jitter, metric, exponent);
        if (d < f1) { f2 = f1; f1 = d; win_cell = cell; }
        else if (d < f2) { f2 = d; }
    }
    // Convert squared Euclidean back to real distance
    if (metric == 0) { f1 = sqrt(max(f1, 0.0)); f2 = sqrt(max(f2, 0.0)); }
}

// float output — distance / feature value
float anacapa_voronoi_f(point pos, float scale, float randomness,
                        int feature, int metric,
                        float exponent, float smoothness)
{
    point sp = pos * max(scale, 1e-6);
    float f1, f2;  point wc;
    _acl_vor_search(sp, randomness, metric, exponent, f1, f2, wc);

    if (feature == 1)  return f2;
    if (feature == 2) {  // Smooth F1
        // Smooth minimum of f1 over all neighbours
        point ip = floor(sp);  vector fp = (vector)(sp - ip);
        float smooth_f1 = 1e18;
        float k = max(smoothness, 1e-6);
        for (int k2 = -1; k2 <= 1; ++k2)
        for (int j2 = -1; j2 <= 1; ++j2)
        for (int i2 = -1; i2 <= 1; ++i2) {
            vector ofs  = vector(i2, j2, k2);
            point  cell = ip + (point)ofs;
            float  rx, ry, rz;
            _acl_rand3(cell, rx, ry, rz);
            vector jitter = ofs - fp
                           + vector(rx-0.5, ry-0.5, rz-0.5) * randomness;
            float d = _acl_vor_dist(jitter, metric, exponent);
            if (metric == 0) d = sqrt(max(d, 0.0));
            float h = clamp(0.5 + 0.5*(d - smooth_f1)/k, 0.0, 1.0);
            smooth_f1 = mix(d, smooth_f1, h) - k * h * (1.0 - h);
        }
        return smooth_f1;
    }
    if (feature == 3)  return f2 - f1;          // Distance to Edge
    if (feature == 4)  return (f1 + f2) * 0.5;  // N-Sphere Radius
    return f1;  // F1 (default)
}

// color output — stable per-cell random colour
color anacapa_voronoi_c(point pos, float scale, float randomness,
                        int feature, int metric,
                        float exponent, float smoothness)
{
    point sp = pos * max(scale, 1e-6);
    float f1, f2;  point wc;
    _acl_vor_search(sp, randomness, metric, exponent, f1, f2, wc);
    return color(cellnoise(wc + vector(0.5, 0.0, 0.0)),
                 cellnoise(wc + vector(0.0, 0.5, 0.0)),
                 cellnoise(wc + vector(0.0, 0.0, 0.5)));
}

// ===========================================================================
// Magic texture  (Blender's sinusoidal chaos pattern)
//
// depth : turbulence depth [0..9] — integer, baked by exporter
// ===========================================================================

color anacapa_magic(point pos, float scale, float distortion, int depth)
{
    point p = pos * scale;
    float x =  sin((p[0] + p[1] + p[2]) *  5.0);
    float y =  cos((-p[0] + p[1] - p[2]) * 5.0);
    float z = -cos((-p[0] - p[1] + p[2]) * 5.0);

    if (depth > 0) {
        x *= distortion;  y *= distortion;  z *= distortion;
        y = -cos(x - y + z);       y *= distortion;
    }
    if (depth > 1) { x = cos(x - y - z);        x *= distortion; }
    if (depth > 2) { z = sin(-x - y - z);        z *= distortion; }
    if (depth > 3) { x = -cos(-x + y - z);       x *= distortion; }
    if (depth > 4) { y = -sin(-x + y + z);       y *= distortion; }
    if (depth > 5) { z = -cos(-x + y + z);       z *= distortion; }
    if (depth > 6) { x =  cos( x + y + z);       x *= distortion; }
    if (depth > 7) { y =  sin( x + y - z);       y *= distortion; }
    if (depth > 8) { z =  cos(-x + y - z);       z *= distortion; }
    if (depth > 9) { x = -sin( x - y - z);       x *= distortion; }

    return color(clamp(0.5 - x / 3.0, 0.0, 1.0),
                 clamp(0.5 - y / 3.0, 0.0, 1.0),
                 clamp(0.5 - z / 3.0, 0.0, 1.0));
}

// ===========================================================================
// Gabor noise
//
// Simplified band-limited noise: Gaussian-weighted oriented sinusoids summed
// over a 5x5x5 impulse grid.  Matches Blender's Gabor texture at low
// frequency well enough for material authoring.
//
// anisotropic : 0 = isotropic (random orientation per impulse)
//               1 = anisotropic (all impulses use `direction`)
// ===========================================================================

float anacapa_gabor(point pos, float scale, float frequency,
                    int anisotropic, vector direction)
{
    point  p  = pos * max(scale, 1e-6);
    point  ip = floor(p);
    float  acc = 0.0;

    for (int k = -2; k <= 2; ++k)
    for (int j = -2; j <= 2; ++j)
    for (int i = -2; i <= 2; ++i) {
        point  cell = ip + point(i, j, k);
        // Random impulse centre within the cell
        float  rx, ry, rz;
        _acl_rand3(cell, rx, ry, rz);
        vector d = (vector)(p - cell) - vector(rx - 0.5, ry - 0.5, rz - 0.5);

        float  gauss = exp(-dot(d, d) * 6.0);
        // Direction: use supplied vector (anisotropic) or random per-impulse
        vector dir;
        if (anisotropic == 1) {
            float len = length(direction);
            dir = (len > 1e-6) ? direction / len : vector(1.0, 0.0, 0.0);
        } else {
            // Random orientation: map two hash values to a unit sphere direction
            float  ha = cellnoise(cell + vector(0.42, 0.0, 0.0)) * M_2PI;
            float  hb = cellnoise(cell + vector(0.0, 0.77, 0.0)) * 2.0 - 1.0;
            float  st = sqrt(max(0.0, 1.0 - hb * hb));
            dir = vector(st * cos(ha), st * sin(ha), hb);
        }
        float phase = cellnoise(cell + vector(0.0, 0.0, 0.13)) * M_2PI;
        acc += gauss * cos(frequency * dot(d, dir) + phase);
    }
    return clamp(acc * 0.25 + 0.5, 0.0, 1.0);
}

// ===========================================================================
// Brick texture
//
// offset_freq, squash_freq : integer properties, baked by exporter
// ===========================================================================

color anacapa_brick(point pos, float scale,
                    color color1, color color2, color mortar_color,
                    float mortar_size, float mortar_smooth,
                    float bias,
                    float brick_width, float row_height,
                    float row_offset,
                    int   offset_freq,
                    float squash, int squash_freq)
{
    point p = pos * max(scale, 1e-6);

    float row     = floor(p[1] / max(row_height, 1e-6));
    int   irow    = (int)row;

    // Per-row horizontal offset
    float col_off = (mod((float)irow, (float)offset_freq) == 0.0)
                    ? row_offset * brick_width : 0.0;

    // Per-row squash
    float bw = brick_width
             * ((mod((float)irow, (float)squash_freq) == 0.0) ? squash : 1.0);

    float bx = (p[0] + col_off) / max(bw, 1e-6);
    float by = p[1] / max(row_height, 1e-6);
    float fx = bx - floor(bx);  // fraction within brick (horizontal)
    float fy = by - floor(by);  // fraction within brick (vertical)

    // Distance to nearest mortar edge (normalised to [0, 0.5])
    float ex = min(fx, 1.0 - fx);
    float ey = min(fy, 1.0 - fy);
    float em = min(ex, ey);

    // Mortar blend: 0 = mortar, 1 = brick
    float hms = max(mortar_size * 0.5, 1e-6);
    float hms_s = max(mortar_smooth * 0.5, 0.0);
    float t = smoothstep(hms - hms_s, hms + hms_s, em);

    // Per-brick random tint driven by bias
    float brick_id = floor(bx) * 1.0 + floor(by) * 1000.0;
    float rnd  = cellnoise(point(brick_id * 0.31, brick_id * 0.17, 0.0)) * 2.0 - 1.0;
    float b    = clamp(bias + rnd * 0.2, -1.0, 1.0);
    color brick_col = mix(color1, color2, clamp(b * 0.5 + 0.5, 0.0, 1.0));

    return mix(mortar_color, brick_col, t);
}

// float (factor) output for brick — 1 inside brick, 0 in mortar
float anacapa_brick_fac(point pos, float scale,
                        float mortar_size, float mortar_smooth,
                        float brick_width, float row_height,
                        float row_offset,
                        int   offset_freq,
                        float squash, int squash_freq)
{
    point p = pos * max(scale, 1e-6);
    float row  = floor(p[1] / max(row_height, 1e-6));
    int   irow = (int)row;
    float col_off = (mod((float)irow, (float)offset_freq) == 0.0)
                    ? row_offset * brick_width : 0.0;
    float bw = brick_width
             * ((mod((float)irow, (float)squash_freq) == 0.0) ? squash : 1.0);
    float bx = (p[0] + col_off) / max(bw, 1e-6);
    float by = p[1] / max(row_height, 1e-6);
    float fx = bx - floor(bx);
    float fy = by - floor(by);
    float em = min(min(fx, 1.0 - fx), min(fy, 1.0 - fy));
    float hms = max(mortar_size * 0.5, 1e-6);
    float hms_s = max(mortar_smooth * 0.5, 0.0);
    return smoothstep(hms - hms_s, hms + hms_s, em);
}

// ===========================================================================
// Object random color  (Blender's OBJECT_INFO → Color / Random output)
//
// Returns a stable greyscale color(v,v,v) where v ∈ [0,1] is uniquely
// derived from the object's world-space origin.  Each GN instance placed
// at a different world position produces a distinct value, matching the
// per-object random color behaviour in Blender Cycles.
// ===========================================================================

color anacapa_object_random_color3()
{
    // Transform the object-space origin (0,0,0) to world space to obtain a
    // position unique to this instance.
    point world_origin = transform("object", "world", point(0.0, 0.0, 0.0));
    float h = cellnoise(world_origin * 13.7 + vector(0.42, 0.73, 0.17));
    return color(h, h, h);
}

#endif // ANACAPA_LIB_H
