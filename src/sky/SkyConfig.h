#pragma once

#include <sky/NishitaSky.h>
#include <string>

namespace anacapa {

// ---------------------------------------------------------------------------
// SkyConfig — sky light configuration loaded from a JSON file
//
// JSON schema (all fields optional; omitted fields use NishitaParams defaults):
//
//   {
//     "type": "nishita",          // must be "nishita" if present
//     "sun_elevation": 45.0,      // degrees above horizon (0=horizon, 90=overhead)
//     "sun_azimuth":   180.0,     // degrees clockwise from north (0=N, 90=E, 180=S)
//     "sun_intensity": 1.0,       // radiance multiplier
//     "sun_disc_size": 2.0,       // angular radius in degrees (~0.27 = real sun, 2.0 = Blender default)
//     "altitude":      0.0,       // viewer altitude above sea level, meters
//     "air_density":   1.0,       // Rayleigh scale (1.0 = Earth standard)
//     "dust_density":  1.0,       // Mie scale (1.0 = standard haze)
//     "ozone_density": 1.0,       // ozone layer scale (1.0 = standard)
//     "transparent_bg":  false   // render sky as alpha=0 for compositing; scene still lit by sky
//   }
//
// Sun direction is derived from elevation + azimuth in Y-up world space:
//   Y = sin(elevation)
//   X = cos(elevation) * sin(azimuth)  (azimuth: 0=+Z/-north, clockwise)
//   Z = cos(elevation) * cos(azimuth)
// ---------------------------------------------------------------------------
struct SkyConfig {
    bool          enabled = false;
    NishitaParams params;
};

// Parses a sky JSON file and returns the config.
// Returns a disabled SkyConfig on parse error (logs a warning via spdlog).
SkyConfig loadSkyConfig(const std::string& jsonPath);

} // namespace anacapa
