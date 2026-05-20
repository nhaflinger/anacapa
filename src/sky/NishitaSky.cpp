#include <sky/NishitaSky.h>

namespace anacapa {

// ---------------------------------------------------------------------------
// skyRadiance — integrate single-scattering along view ray d
//
// Algorithm:
//   1. March kNView samples along the view ray (viewer → atmosphere exit).
//   2. At each sample point, march kNLight samples toward the sun to
//      compute the solar transmittance T_sun reaching that point.
//   3. Accumulate Rayleigh and Mie in-scatter contributions:
//        dL = T_view * T_sun * (betaR * phaseR + betaM_scat * phaseM) * dt
//   4. Return zero for rays pointing into the ground.
// ---------------------------------------------------------------------------
Spectrum NishitaSky::skyRadiance(Vec3f d) const
{
    Vec3f orig = viewerPos();

    // Ground rays → black (or could return ground albedo in future)
    if (hitsEarth(orig, d))
        return {};

    float tAtmos = raySphereExit(orig, d, kRAtmos);
    if (tAtmos < 0.f)
        return {};  // shouldn't happen for valid directions, but be safe

    float dt = tAtmos / static_cast<float>(kNView);

    float cosTheta = dot(d, m_p.sunDir);
    float phaseR   = phaseRayleigh(cosTheta);
    float phaseM   = phaseMie(cosTheta);

    // Accumulated optical depth along view ray (viewer → current sample)
    OD od_view;

    Spectrum Lr = {};   // Rayleigh scatter accumulator
    Spectrum Lm = {};   // Mie scatter accumulator

    for (int i = 0; i < kNView; ++i)
    {
        Vec3f p = orig + d * (dt * (i + 0.5f));
        float h = heightAt(p);

        float dR = densityR(h) * dt;
        float dM = densityM(h) * dt;
        float dO = densityO(h) * dt;

        od_view.R += dR;
        od_view.M += dM;
        od_view.O += dO;

        // Shadow ray toward the sun — skip if sun direction points into ground
        float tSun = raySphereExit(p, m_p.sunDir, kRAtmos);
        if (tSun < 0.f || hitsEarth(p, m_p.sunDir))
            continue;

        OD od_sun = marchOD(p, m_p.sunDir, tSun, kNLight);

        // Combined transmittance: viewer→p and p→sun
        OD od_total;
        od_total.R = od_view.R + od_sun.R;
        od_total.M = od_view.M + od_sun.M;
        od_total.O = od_view.O + od_sun.O;

        Spectrum T = transmittance(od_total);

        // Rayleigh: β_R is per-channel, phase is scalar
        Lr.x += T.x * kBetaR.x * densityR(h) * dt;
        Lr.y += T.y * kBetaR.y * densityR(h) * dt;
        Lr.z += T.z * kBetaR.z * densityR(h) * dt;

        // Mie: β_M_scat = β_M_ext * albedo (wavelength-independent)
        float betaMScat = kBetaMExt * m_p.dustDensity * kMieAlbedo;
        Lm.x += T.x * betaMScat * densityM(h) * dt;
        Lm.y += T.y * betaMScat * densityM(h) * dt;
        Lm.z += T.z * betaMScat * densityM(h) * dt;
    }

    float solarScale = kSolarRadiance * m_p.sunIntensity;

    Spectrum L;
    L.x = solarScale * (Lr.x * phaseR + Lm.x * phaseM);
    L.y = solarScale * (Lr.y * phaseR + Lm.y * phaseM);
    L.z = solarScale * (Lr.z * phaseR + Lm.z * phaseM);
    return L;
}

// ---------------------------------------------------------------------------
// sunDiscRadiance — attenuated solar beam from viewer to atmosphere
//
// The disc is treated as a directional emitter; its colour comes from the
// transmittance along the viewer→atmosphere path in the sun's direction.
// ---------------------------------------------------------------------------
Spectrum NishitaSky::sunDiscRadiance() const
{
    Vec3f orig = viewerPos();

    // If sun is below the horizon, disc is invisible
    if (hitsEarth(orig, m_p.sunDir))
        return {};

    float tAtmos = raySphereExit(orig, m_p.sunDir, kRAtmos);
    if (tAtmos < 0.f)
        return {};

    OD od = marchOD(orig, m_p.sunDir, tAtmos, kNLight * 2);
    Spectrum T = transmittance(od);

    float solarScale = kSolarRadiance * m_p.sunIntensity;
    return { T.x * solarScale, T.y * solarScale, T.z * solarScale };
}

} // namespace anacapa
