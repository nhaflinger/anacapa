#include "render/RenderSession.h"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    CLI::App app{"anacapa — bidirectional path tracer"};

    anacapa::RenderSettings settings;

    app.add_option("-o,--output",  settings.outputPath,     "Output EXR path")
       ->default_val("out.exr");
    app.add_option("-W,--width",   settings.imageWidth,     "Image width")
       ->default_val(800);
    app.add_option("-H,--height",  settings.imageHeight,    "Image height")
       ->default_val(600);
    app.add_option("-s,--spp",     settings.samplesPerPixel,"Samples per pixel")
       ->default_val(64);
    app.add_option("-d,--depth",   settings.maxDepth,       "Max path depth")
       ->default_val(8);
    app.add_option("-t,--threads", settings.numThreads,     "Thread count (0=auto)")
       ->default_val(0);
    app.add_option("--tile-size",  settings.tileSize,       "Tile size in pixels")
       ->default_val(64);

    CLI11_PARSE(app, argc, argv);

    spdlog::set_level(spdlog::level::info);

    anacapa::RenderSession session(settings);
    session.buildCornellBox();
    session.render();

    return 0;
}
