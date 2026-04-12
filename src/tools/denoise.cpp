// denoise — standalone OIDN denoiser for Anacapa EXR renders.
//
// Reads an EXR whose R, G, B channels are a noisy linear HDR beauty.
// Optionally reads albedo and normal auxiliary channels from the same file
// (written by anacapa --write-aovs) to guide the denoiser.
// Writes the denoised result as a new EXR.
//
// Usage:
//   denoise -i noisy.exr -o denoised.exr
//   denoise -i noisy.exr -o denoised.exr --albedo-channel albedo.R --normal-channel normals.R

#include <CLI/CLI.hpp>
#include <OpenImageIO/imageio.h>
#include <OpenImageDenoise/oidn.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace OIIO;

// ---------------------------------------------------------------------------
// Read a flat RGB float buffer from an EXR, selecting specific channel names.
// rChannel should be the R channel name (e.g. "R", "albedo.R", "normals.R").
// G and B are derived by replacing the trailing 'R' with 'G'/'B'.
// Returns false if any of the three channels are not found.
// ---------------------------------------------------------------------------
static bool readChannels(const std::string& path,
                         const std::string& rChannel,
                         uint32_t width, uint32_t height,
                         std::vector<float>& out)
{
    auto inp = ImageInput::open(path);
    if (!inp) {
        std::fprintf(stderr, "denoise: cannot open '%s'\n", path.c_str());
        return false;
    }

    const ImageSpec& spec = inp->spec();
    const uint32_t N = width * height;

    auto findChan = [&](const std::string& name) -> int {
        for (int i = 0; i < (int)spec.channelnames.size(); ++i)
            if (spec.channelnames[i] == name) return i;
        return -1;
    };

    // Derive G and B from R: strip trailing 'R' to get prefix, then append G/B
    // e.g. "albedo.R" → prefix "albedo." → "albedo.G", "albedo.B"
    // e.g. "R"        → prefix ""        → "G", "B"
    std::string prefix = rChannel;
    if (!prefix.empty() && prefix.back() == 'R')
        prefix.pop_back();
    const std::string gChan = prefix + "G";
    const std::string bChan = prefix + "B";

    int ri = findChan(rChannel);
    int gi = findChan(gChan);
    int bi = findChan(bChan);

    if (ri < 0 || gi < 0 || bi < 0) {
        std::fprintf(stderr,
            "denoise: channels '%s'/'%s'/'%s' not found in '%s' — skipping\n",
            rChannel.c_str(), gChan.c_str(), bChan.c_str(), path.c_str());
        inp->close();
        return false;
    }

    int nChans = spec.nchannels;
    std::vector<float> full(N * nChans);
    inp->read_image(0, 0, 0, nChans, TypeDesc::FLOAT, full.data());
    inp->close();

    out.resize(N * 3);
    for (uint32_t i = 0; i < N; ++i) {
        out[i*3+0] = full[i * nChans + ri];
        out[i*3+1] = full[i * nChans + gi];
        out[i*3+2] = full[i * nChans + bi];
    }
    return true;
}

// ---------------------------------------------------------------------------
// Write a flat RGB float buffer as a single-subimage EXR
// ---------------------------------------------------------------------------
static bool writeEXR(const std::string& path,
                     uint32_t width, uint32_t height,
                     const std::vector<float>& pixels)
{
    ImageSpec spec(static_cast<int>(width), static_cast<int>(height),
                   3, TypeDesc::FLOAT);
    spec.attribute("compression", "zip");
    spec.channelnames = {"R", "G", "B"};

    auto out = ImageOutput::create(path);
    if (!out || !out->open(path, spec)) {
        std::fprintf(stderr, "denoise: cannot write '%s'\n", path.c_str());
        return false;
    }
    bool ok = out->write_image(TypeDesc::FLOAT, pixels.data());
    out->close();
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    CLI::App app{"denoise — standalone OIDN denoiser for Anacapa EXR renders"};

    std::string inputPath, outputPath;
    std::string colorChannel  = "R";
    std::string albedoChannel = "";
    std::string normalChannel = "";

    app.add_option("-i,--input",  inputPath,  "Input EXR (noisy beauty)")->required();
    app.add_option("-o,--output", outputPath, "Output EXR (denoised beauty)")->required();
    app.add_option("--color-channel", colorChannel,
                   "R channel name of the noisy beauty layer (default: R)")
       ->default_val("R");
    app.add_option("--albedo-channel", albedoChannel,
                   "R channel name of the albedo AOV (e.g. albedo.R). "
                   "Omit to denoise without albedo guidance.");
    app.add_option("--normal-channel", normalChannel,
                   "R channel name of the normals AOV (e.g. normals.R). "
                   "Omit to denoise without normal guidance.");

    CLI11_PARSE(app, argc, argv);

    // Read input to get dimensions
    {
        auto inp = ImageInput::open(inputPath);
        if (!inp) {
            std::fprintf(stderr, "denoise: cannot open input '%s'\n", inputPath.c_str());
            return 1;
        }
        inp->close();
    }

    auto inp = ImageInput::open(inputPath);
    const ImageSpec& spec = inp->spec();
    const uint32_t width  = static_cast<uint32_t>(spec.width);
    const uint32_t height = static_cast<uint32_t>(spec.height);
    inp->close();

    std::fprintf(stderr, "denoise: %ux%u — reading '%s'\n", width, height, inputPath.c_str());

    // Read beauty
    std::vector<float> color;
    if (!readChannels(inputPath, colorChannel, width, height, color))
        return 1;

    // Read optional AOVs
    std::vector<float> albedo, normals;
    bool hasAlbedo  = !albedoChannel.empty() &&
                      readChannels(inputPath, albedoChannel, width, height, albedo);
    bool hasNormals = !normalChannel.empty() &&
                      readChannels(inputPath, normalChannel, width, height, normals);

    // OIDN
    const uint32_t N = width * height;
    std::vector<float> output(N * 3);

    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
    oidnCommitDevice(device);

    OIDNFilter filter = oidnNewFilter(device, "RT");

    oidnSetSharedFilterImage(filter, "color",  color.data(),
                             OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    if (hasAlbedo)
        oidnSetSharedFilterImage(filter, "albedo", albedo.data(),
                                 OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    if (hasNormals)
        oidnSetSharedFilterImage(filter, "normal", normals.data(),
                                 OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "output", output.data(),
                             OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);

    oidnSetFilterBool(filter, "hdr", true);
    oidnCommitFilter(filter);

    std::fprintf(stderr, "denoise: running OIDN (albedo=%s, normals=%s)...\n",
                 hasAlbedo ? "yes" : "no", hasNormals ? "yes" : "no");
    oidnExecuteFilter(filter);

    const char* errMsg = nullptr;
    if (oidnGetDeviceError(device, &errMsg) != OIDN_ERROR_NONE) {
        std::fprintf(stderr, "denoise: OIDN error: %s\n", errMsg);
        oidnReleaseFilter(filter);
        oidnReleaseDevice(device);
        return 1;
    }

    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);

    if (!writeEXR(outputPath, width, height, output))
        return 1;

    std::fprintf(stderr, "denoise: wrote '%s'\n", outputPath.c_str());
    return 0;
}
