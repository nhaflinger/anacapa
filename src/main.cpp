#include "render/RenderSession.h"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    CLI::App app{"anacapa — bidirectional path tracer"};

    anacapa::RenderSettings settings;

    app.add_option("-o,--output",  settings.outputPath,      "Output EXR path")
       ->default_val("out.exr");
    app.add_option("-W,--width",   settings.imageWidth,      "Image width")
       ->default_val(800);
    app.add_option("-H,--height",  settings.imageHeight,     "Image height")
       ->default_val(800);
    app.add_option("-s,--spp",     settings.samplesPerPixel, "Samples per pixel")
       ->default_val(64);
    app.add_option("-d,--depth",   settings.maxDepth,        "Max path depth")
       ->default_val(8);
    app.add_option("-t,--threads", settings.numThreads,      "Thread count (0=auto)")
       ->default_val(0);
    app.add_option("--tile-size",  settings.tileSize,        "Tile size in pixels")
       ->default_val(64);

    std::string integratorName = "bdpt";
    app.add_option("--integrator", integratorName,
                   "Integrator: bdpt (default) or path")
       ->default_val("bdpt");

    app.add_option("--firefly-clamp", settings.fireflyClamp,
                   "BDPT: max luminance per strategy contribution (0=off, default=10)")
       ->default_val(10.f);

    app.add_flag("--adaptive", settings.adaptive,
                 "Enable adaptive per-tile sample allocation: base pass + high-variance refinement");
    app.add_option("--adaptive-base-spp", settings.adaptiveBaseSpp,
                   "Adaptive: base-pass SPP (0=auto: spp/4, min 16)")
       ->default_val(0);

    app.add_option("--scene", settings.scenePath,
                   "USD/USDA/USDC scene file to load (requires ANACAPA_ENABLE_USD)");
    app.add_option("--camera", settings.cameraPath,
                   "USD prim path of camera to use (e.g. /World/RenderCam). "
                   "If omitted, uses UsdRenderSettings.camera or first camera found.");
    app.add_option("--env", settings.envPath,
                   "Equirectangular HDRI environment map (EXR or HDR)");
    app.add_option("--env-intensity", settings.envIntensity,
                   "Intensity multiplier for the environment map")
       ->default_val(1.f);

    app.add_option("--fstop", settings.fStop,
                   "Lens f-stop (e.g. 2.8). Enables depth of field when combined "
                   "with --focus-distance. Overrides the USD camera value if present.")
       ->default_val(0.f);
    app.add_option("--focus-distance", settings.focusDistance,
                   "Distance from camera to the focal plane in scene units. "
                   "Overrides the USD camera value if present.")
       ->default_val(0.f);

    app.add_option("--shutter-open", settings.shutterOpen,
                   "Shutter open time (USD time units, default 0). "
                   "Must be less than --shutter-close to enable motion blur.")
       ->default_val(0.f);
    app.add_option("--shutter-close", settings.shutterClose,
                   "Shutter close time (USD time units, default 0). "
                   "Set > shutter-open to enable transformation motion blur.")
       ->default_val(0.f);

    app.add_flag("--interactive", settings.interactive,
                 "Use GPU (Metal) backend for fast preview renders — "
                 "lower quality, much faster (requires ANACAPA_ENABLE_METAL)");

    app.add_flag("--override-lights", settings.overrideLights,
                 "Replace all scene lights with a single white directional light "
                 "(useful for isolating material issues from lighting issues)");

    app.add_flag("--override-materials", settings.overrideMaterials,
                 "Replace all scene materials with white Lambertian "
                 "(useful for isolating lighting issues from material issues)");

    app.add_option("--png", settings.pngPath,
                   "Write an ACES-tonemapped sRGB PNG for easy comparison (e.g. out.png)");
    app.add_option("--exposure", settings.exposure,
                   "EV exposure adjustment for --png output (default 0)")
       ->default_val(0.f);

    app.add_flag("--denoise",    settings.denoise.enabled,
                 "Run Intel OIDN denoiser on the beauty buffer after rendering");
    app.add_flag("--write-aovs", settings.denoise.writeAOVs,
                 "Include albedo and normals layers in the output EXR");

    CLI11_PARSE(app, argc, argv);

    if (integratorName == "path")
        settings.integrator = anacapa::IntegratorType::Path;
    else
        settings.integrator = anacapa::IntegratorType::BDPT;

    spdlog::set_level(spdlog::level::info);

    anacapa::RenderSession session(settings);
    session.loadScene();
    session.render();

    return 0;
}
