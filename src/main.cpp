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
