#include "render/RenderSession.h"
#include "export/SceneExporter.h"
#include <anacapa/render/SocketDisplayDriver.h>
#include <anacapa/render/DisplayProtocol.h>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    CLI::App app{"anacapa — bidirectional path tracer"};

    // --export-scene <blob> <usd>  — convert a binary scene blob to USD.
    // This subcommand is invoked by the Blender addon and exits immediately.
    std::string exportBlobPath, exportUsdPath;
    auto* exportSub = app.add_subcommand("export-scene",
        "Convert a binary scene blob written by the Blender addon to USD");
    exportSub->add_option("blob", exportBlobPath, "Input binary blob path")->required();
    exportSub->add_option("usd",  exportUsdPath,  "Output USD path")->required();

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

    std::string integratorName = "path";
    app.add_option("--integrator", integratorName,
                   "Integrator: path (default), bdpt, or photon")
       ->default_val("path");

    app.add_option("--num-photons", settings.numPhotons,
                   "Photon map: total photons emitted per frame (default 500000)")
       ->default_val(500'000);
    app.add_option("--photon-radius", settings.photonRadius,
                   "Photon map: density estimate search radius in world units (default 0.1)")
       ->default_val(0.1f);

    app.add_option("--firefly-clamp", settings.fireflyClamp,
                   "Max luminance per path sample; applies to path, BDPT, and GPU integrators (0=off, default=10)")
       ->default_val(10.f);

    app.add_option("--light-angle", settings.lightAngle,
                   "Angular radius for directional lights in degrees (0=hard shadows, "
                   "0.27=sun, 1-5=soft). Turns hard point sources into soft area lights.")
       ->default_val(0.f);

    app.add_flag("--no-adaptive{false}", settings.adaptive,
                 "Disable adaptive per-tile sample allocation (on by default)");
    app.add_option("--adaptive-base-spp", settings.adaptiveBaseSpp,
                   "Adaptive: base-pass SPP (0=auto: spp/4, min 16)")
       ->default_val(0);

    app.add_option("--scene", settings.scenePath,
                   "USD/USDA/USDC scene file to load (requires ANACAPA_ENABLE_USD)");
    app.add_option("--curves", settings.curvesPath,
                   "Alembic .abc file containing hair/fur curves (requires ANACAPA_ENABLE_ALEMBIC)");
    app.add_option("--particles", settings.particlesPath,
                   "Alembic .abc file containing IPoints particle clouds (requires ANACAPA_ENABLE_ALEMBIC)");
    app.add_option("--matassign", settings.matassignPaths,
                   "Material assignment JSON file (repeatable; later files override earlier ones). "
                   "<abc>.matassign.json is always auto-discovered alongside --curves.")
       ->allow_extra_args(false);
    app.add_option("--camera", settings.cameraPath,
                   "USD prim path of camera to use (e.g. /World/RenderCam). "
                   "If omitted, uses UsdRenderSettings.camera or first camera found.");
    app.add_option("--env", settings.envPath,
                   "Equirectangular HDRI environment map (EXR or HDR)");
    app.add_option("--env-intensity", settings.envIntensity,
                   "Intensity multiplier for the environment map")
       ->default_val(1.f);
    app.add_option("--sky", settings.skyConfigPath,
                   "Nishita sky config JSON file. Replaces --env when specified. "
                   "See sky JSON schema in docs.");

    app.add_option("--fstop", settings.fStop,
                   "Lens f-stop (e.g. 2.8). Enables depth of field when combined "
                   "with --focus-distance. Overrides the USD camera value if present.")
       ->default_val(0.f);
    app.add_option("--focus-distance", settings.focusDistance,
                   "Distance from camera to the focal plane in scene units. "
                   "Overrides the USD camera value if present.")
       ->default_val(0.f);
    app.add_flag("--no-dof", settings.noDof,
                 "Force pinhole camera — disables depth of field regardless of USD camera settings.");

    auto* frameOpt = app.add_option("--frame", settings.frameNumber,
                   "Frame to render (USD time code / Blender frame number). "
                   "Defaults to the stage's start time code if not specified.")
       ->default_val(0);

    app.add_option("--shutter-open", settings.shutterOpen,
                   "Shutter open offset from --frame (in frames/time codes, default 0). "
                   "Must be less than --shutter-close to enable motion blur.")
       ->default_val(0.f);
    app.add_option("--shutter-close", settings.shutterClose,
                   "Shutter close offset from --frame (in frames/time codes, default 0). "
                   "Set > shutter-open to enable transformation motion blur.")
       ->default_val(0.f);

    app.add_flag("--gpu-assist", settings.gpuAssist,
                 "Use GPU (Metal|CUDA) backend for rendering "
                 "(requires ANACAPA_ENABLE_METAL|ANACAPA_ENABLE_CUDA)");
    // Legacy alias so existing scripts don't break immediately.
    app.add_flag("--interactive", settings.gpuAssist)->group("");

    app.add_flag("--override-lights", settings.overrideLights,
                 "Replace all scene lights with a single white directional light "
                 "(useful for isolating material issues from lighting issues)");

    app.add_flag("--override-materials", settings.overrideMaterials,
                 "Replace all scene materials with white Lambertian "
                 "(useful for isolating lighting issues from material issues)");

    app.add_flag("--skip-osl", settings.skipOSL,
                 "Skip OSL shader loading; fall through to StandardSurface/UsdPreviewSurface "
                 "(useful for verifying material assignment without OSL execution).");

    app.add_option("--debug-mesh", settings.debugMeshID,
                   "Debug: primary rays that hit any other mesh ID return "
                   "black, so only the target mesh's pixels are shaded.  "
                   "Indirect bounces are unaffected.  Default: -1 (off).")
       ->default_val(-1);

    app.add_option("--highlight-mesh", settings.highlightMeshID,
                   "Debug: renders the target mesh as flat red; all other "
                   "geometry is dimmed to 20%%.  Useful for visually locating "
                   "a mesh by its ID in the render.  Default: -1 (off).")
       ->default_val(-1);


    app.add_option("--png", settings.pngPath,
                   "Write an sRGB-encoded PNG for easy comparison (exposure + sRGB "
                   "gamma only, no tone mapping; e.g. out.png)");
    app.add_option("--preview-exr", settings.previewExrPath,
                   "Write progressive linear EXR previews to this path during rendering "
                   "(includes alpha when transparent-bg is enabled; e.g. preview.exr)");
    app.add_option("--exposure", settings.exposure,
                   "EV exposure adjustment for --png output (default 0)")
       ->default_val(0.f);

    app.add_flag("--denoise",    settings.denoise.enabled,
                 "Run Intel OIDN denoiser on the beauty buffer after rendering");
    app.add_flag("--write-aovs", settings.denoise.writeAOVs,
                 "Include albedo and normals layers in the output EXR");

    std::string filterName = "mitchell";
    app.add_option("--filter", filterName,
                   "Pixel reconstruction filter: box, triangle, gaussian, "
                   "mitchell (default), blackman-harris, catmull-rom, lanczos. "
                   "Mitchell-Netravali is a good general default; blackman-harris "
                   "matches Cycles' look.")
       ->default_val("mitchell");
    app.add_option("--filter-width", settings.pixelFilterRadius,
                   "Pixel filter radius in pixel units. 0 = use the filter's "
                   "default (Box=0.5, Triangle=1.0, Gaussian=1.5, Mitchell=2.0, "
                   "Blackman-Harris=1.5, Catmull-Rom=2.0, Lanczos=4.0).")
       ->default_val(0.f);
    app.add_option("--hair-tess-steps", settings.hairTessSteps,
                   "Quads per cubic Bézier hair segment for ribbon tessellation "
                   "(CPU and GPU). Higher = smoother curves. Default 4.")
       ->default_val(4)
       ->check(CLI::Range(1, 32));
    std::string displaySock;
    app.add_option("--display", displaySock,
                   "Push tiles to viewer at this Unix socket path. "
                   "Defaults to " + std::string(anacapa::proto::kDefaultSockPath) +
                   " when flag is given without a value.")
       ->expected(0, 1)
       ->default_val("");

    CLI11_PARSE(app, argc, argv);

    if (*exportSub) {
        bool ok = anacapa::exportSceneToUsd(exportBlobPath, exportUsdPath);
        return ok ? 0 : 1;
    }

    settings.frameSet = (frameOpt->count() > 0);

    if (integratorName == "path")
        settings.integrator = anacapa::IntegratorType::Path;
    else if (integratorName == "photon")
        settings.integrator = anacapa::IntegratorType::PhotonMap;
    else
        settings.integrator = anacapa::IntegratorType::BDPT;

    if (!anacapa::parsePixelFilterName(filterName, settings.pixelFilter)) {
        spdlog::warn("Unknown --filter '{}'; using mitchell", filterName);
        settings.pixelFilter = anacapa::PixelFilterType::Mitchell;
    }

    spdlog::set_level(spdlog::level::info);

    anacapa::RenderSession session(settings);
    session.loadScene();

    // If --display was given (with or without an explicit path), connect to the
    // viewer socket and use SocketDisplayDriver.  Fall back silently if the
    // viewer is not running — the FileDisplayDriver default takes over.
    std::unique_ptr<anacapa::SocketDisplayDriver> sockDriver;
    if (!displaySock.empty() || app.count("--display")) {
        const std::string path = displaySock.empty()
                               ? anacapa::proto::kDefaultSockPath
                               : displaySock;
        sockDriver = std::make_unique<anacapa::SocketDisplayDriver>(path);
        if (sockDriver->isConnected()) {
            session.setDisplayDriver(sockDriver.get());
            sockDriver->onCancelFn = [&session]{ session.requestCancel(); };
            sockDriver->onPauseFn  = [&session]{ session.requestPause();  };
            sockDriver->onResumeFn = [&session]{ session.requestResume(); };
            sockDriver->onCropFn   = [&session](uint32_t cx, uint32_t cy,
                                                uint32_t cw, uint32_t ch) {
                session.requestCrop(cx, cy, cw, ch);
            };
            sockDriver->onClearCropFn = [&session]{ session.requestClearCrop(); };
        } else {
            sockDriver.reset();
        }
    }

    session.render();

    return 0;
}
