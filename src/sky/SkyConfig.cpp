#include <sky/SkyConfig.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <cmath>

namespace anacapa {

static constexpr float kDeg2Rad = 3.14159265358979323846f / 180.f;

// Build a Y-up world-space sun direction from elevation+azimuth in degrees.
// Elevation: 0 = horizon, 90 = directly overhead.
// Azimuth:   0 = +Z (north), increases clockwise (+X = east).
static Vec3f sunDirFromAngles(float elevDeg, float azimDeg)
{
    float elev = elevDeg * kDeg2Rad;
    float az   = azimDeg * kDeg2Rad;
    float cosE = std::cos(elev);
    return {
        cosE * std::sin(az),   // X: east component
        std::sin(elev),        // Y: up component
        cosE * std::cos(az)    // Z: north component
    };
}

SkyConfig loadSkyConfig(const std::string& jsonPath)
{
    SkyConfig cfg;

    std::ifstream f(jsonPath);
    if (!f.is_open()) {
        spdlog::warn("[sky] Cannot open sky config '{}': file not found", jsonPath);
        return cfg;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const nlohmann::json::exception& e) {
        spdlog::warn("[sky] JSON parse error in '{}': {}", jsonPath, e.what());
        return cfg;
    }

    // Validate type field if present
    if (j.contains("type")) {
        std::string t = j["type"].get<std::string>();
        if (t != "nishita") {
            spdlog::warn("[sky] Unknown sky type '{}' in '{}'; only 'nishita' is supported",
                         t, jsonPath);
            return cfg;
        }
    }

    // Read elevation + azimuth and convert to a sun direction vector.
    // Default: elevation=45°, azimuth=180° (sun in the south at mid-afternoon angle).
    float elevDeg = j.value("sun_elevation", 45.f);
    float azimDeg = j.value("sun_azimuth",  180.f);
    cfg.params.sunDir = sunDirFromAngles(elevDeg, azimDeg);

    cfg.params.sunIntensity = j.value("sun_intensity",  1.f);
    cfg.params.sunDiscAngle = j.value("sun_disc_size",  2.0f) * kDeg2Rad;
    cfg.params.altitude     = j.value("altitude",       0.f);
    cfg.params.airDensity   = j.value("air_density",    1.f);
    cfg.params.dustDensity  = j.value("dust_density",   1.f);
    cfg.params.ozoneDensity  = j.value("ozone_density",  1.f);
    cfg.params.transparentBg = j.value("transparent_bg", false);

    cfg.enabled = true;
    spdlog::info("[sky] Loaded Nishita sky: elevation={:.1f}° azimuth={:.1f}° "
                 "intensity={:.2f} alt={:.0f}m",
                 elevDeg, azimDeg, cfg.params.sunIntensity, cfg.params.altitude);
    return cfg;
}

} // namespace anacapa
