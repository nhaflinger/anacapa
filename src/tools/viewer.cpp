// viewer.cpp — Anacapa progressive render viewer
//
// SDL2 + ImGui + OpenGL 3.3 display tool.  Watches a file written by the
// renderer and shows it with real-time color grading.  When OpenColorIO is
// available the display transform is driven by the active OCIO config
// (from $OCIO or the built-in ACES CG config); otherwise an ACES polynomial
// approximation is used as a fallback.
//
// Usage:
//   viewer preview.png
//   viewer preview.png --interval 250   # poll every 250 ms (default 500)

// glad must be included before any other GL headers
#include <glad/glad.h>

// Socket / networking
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>

#include <SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"

#include <CLI/CLI.hpp>

#ifdef ANACAPA_HAVE_OCIO
#include <OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>

#include <anacapa/render/DisplayProtocol.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_HDR
#include "stb_image.h"

#include <OpenImageIO/imageio.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "imfilebrowser.h"

// ---------------------------------------------------------------------------
// GL resources
// ---------------------------------------------------------------------------
static GLuint g_fbo        = 0;
static GLuint g_dstTexture = 0;
static int    g_fboWidth   = 0;
static int    g_fboHeight  = 0;

static GLuint g_quadVAO = 0;
static GLuint g_quadVBO = 0;

// Active shader program (rebuilt when OCIO view changes)
static GLuint g_shader = 0;

// ---------------------------------------------------------------------------
// Live socket render state
// ---------------------------------------------------------------------------
struct TileUpdate {
    uint32_t x0, y0, w, h;
    std::vector<float> rgba;  // w*h*4 floats, RGBA, top-left origin
};

struct LiveState {
    // Socket server
    int         serverFd  = -1;
    int         clientFd  = -1;
    std::thread readerThread;
    std::atomic<bool> readerStop{false};

    // Live texture (RGBA32F, updated tile-by-tile via glTexSubImage2D)
    GLuint      liveTex = 0;
    uint32_t    liveW   = 0;
    uint32_t    liveH   = 0;
    bool        active  = false;  // true while a render is connected
    bool        done    = false;  // true after IMAGE_CLOSE received

    // Tile queue (reader thread → main thread)
    std::mutex              queueMtx;
    std::queue<TileUpdate>  queue;
    std::atomic<bool>       dirty{false};

    // Outbound command socket fd (same as clientFd — protected by sendMtx)
    std::mutex sendMtx;

    // Crop rect (in image pixels; set once image dimensions are known)
    uint32_t cropX = 0, cropY = 0, cropW = 0, cropH = 0;
    bool cropActive = false;

    // Pre-launch crop (normalized [0,1]; stored before image dimensions are known).
    // Converted to pixels and sent as SET_CROP when IMAGE_OPEN arrives.
    float cropNormX0 = 0, cropNormY0 = 0, cropNormX1 = 0, cropNormY1 = 0;
    bool  cropNormActive = false;
    // True when cropNorm was stored relative to the viewer panel (pre-launch),
    // false when stored relative to the image display area (post-render).
    // On IMAGE_OPEN the main thread converts panel-relative to image-relative so
    // the overlay stays in the same screen position it was drawn.
    bool  cropNormIsPanel = false;
    // Panel dimensions at the time the pre-launch crop was drawn; used for
    // the panel→image-display conversion on IMAGE_OPEN.
    float cropPanelW = 0, cropPanelH = 0;

    // Render controls
    bool paused = false;

    // Progress tracking (reader thread writes, main thread reads)
    std::atomic<uint32_t> tilesReceived{0};
    std::atomic<uint64_t> pixelsFilled{0};
    uint32_t totalPixels = 0;  // set from IMAGE_OPEN width*height
    uint32_t imageSpp    = 0;  // set from IMAGE_OPEN spp field
    bool     hasAlpha    = false; // set from IMAGE_OPEN hasAlpha field
} g_live;

// Send a command message to the connected renderer (viewer→renderer direction)
static void sendCommand(const std::vector<uint8_t>& msg) {
    if (g_live.clientFd < 0) return;
    std::lock_guard<std::mutex> lk(g_live.sendMtx);
    size_t sent = 0;
    while (sent < msg.size()) {
        ssize_t n = ::send(g_live.clientFd, msg.data() + sent,
                           msg.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
}

// ---------------------------------------------------------------------------
// OCIO state
// ---------------------------------------------------------------------------
#ifdef ANACAPA_HAVE_OCIO
struct OcioTexture {
    GLuint id     = 0;
    GLenum target = GL_TEXTURE_3D;   // GL_TEXTURE_3D or GL_TEXTURE_1D/2D
    std::string samplerName;
    int unit = 1;  // GL texture unit assigned to this texture
};

static OCIO::ConstConfigRcPtr    g_ocioConfig;
static std::vector<std::string>  g_ocioDisplays;
static std::vector<std::string>  g_ocioViews;
static int                       g_ocioDisplayIdx = 0;
static int                       g_ocioViewIdx    = 0;
static bool                      g_ocioEnabled    = true;
static bool                      g_ocioShaderDirty = true;
static std::string               g_ocioStatus;       // shown in Color Management panel
static std::vector<OcioTexture>  g_ocioTextures;
static std::string               g_ocioConfigName;   // display name for the UI
#endif

// ---------------------------------------------------------------------------
// Shader source — split into preamble (uniforms + helpers) and main().
// When OCIO is available, the OCIO-generated GLSL is injected between them
// and %OCIO_FUNC% in the main is replaced with the generated function call.
// ---------------------------------------------------------------------------
static const char* kVertSrc = R"glsl(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() { vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }
)glsl";

// Everything before the OCIO-generated GLSL and before void main().
static const char* kFragPreamble = R"glsl(
#version 330 core
in  vec2 vUV;
out vec4 fragColor;

uniform sampler2D uTex;
uniform float uExposure;      // EV stops
uniform float uSaturation;    // 0=greyscale, 1=original, 2=vivid
uniform float uContrast;      // -1 to +1
uniform float uTemperature;   // -1=cool, 0=neutral, +1=warm
uniform bool  uTonemap;       // fallback ACES on/off (when OCIO disabled)
uniform bool  uUseLut;        // true when OCIO shader is active
uniform bool  uIsHdr;         // true when source is a float HDR image
uniform int   uChannelMode;   // 0=RGB, 1=R, 2=G, 3=B, 4=A

// sRGB decode/encode (approximate gamma 2.2)
vec3 srgbToLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 linearToSrgb(vec3 c) { return pow(max(c, 0.0), vec3(1.0/2.2)); }

// Log-based filmic tone mapping, matching Blender Cycles "Filmic" dynamic range.
// 16-stop log2 range centred on 18% grey; smoothstep S-curve in log space.
// Sky values of 5–20 (sunIntensity=10) remain distinctly coloured, not white.
float filmicChan(float x) {
    if (x <= 0.0) return 0.0;
    float lx = log2(max(x, 1e-10) / 0.18) / 16.0 + 0.5;
    lx = clamp(lx, 0.0, 1.0);
    return lx * lx * (3.0 - 2.0 * lx);
}
vec3 filmic(vec3 x) {
    return vec3(filmicChan(x.r), filmicChan(x.g), filmicChan(x.b));
}

vec3 applyTemperature(vec3 c, float t) {
    c.r *= 1.0 + t * 0.2;
    c.g *= 1.0 + t * 0.05;
    c.b *= 1.0 - t * 0.2;
    return c;
}
)glsl";

// Our void main() — %OCIO_FUNC% is replaced before compilation.
// When OCIO is active it calls the injected function; when not, dead branch.
static const char* kFragMain = R"glsl(
void main() {
    vec4 texel = texture(uTex, vUV);

    // Channel isolation — show a single channel as greyscale, bypassing grading.
    if (uChannelMode >= 1 && uChannelMode <= 3) {
        float v = (uChannelMode == 1) ? texel.r
                : (uChannelMode == 2) ? texel.g : texel.b;
        if (!uIsHdr) v = pow(max(v, 0.0), 2.2);  // sRGB decode for LDR source
        v *= pow(2.0, uExposure);
        fragColor = vec4(linearToSrgb(vec3(clamp(v, 0.0, 1.0))), 1.0);
        return;
    }
    if (uChannelMode == 4) {
        fragColor = vec4(vec3(texel.a), 1.0);
        return;
    }

    vec3 c = texel.rgb;

    // Decode to linear light (skip for HDR which is already linear)
    if (!uIsHdr)
        c = srgbToLinear(c);

    // Scene-linear adjustments
    c *= pow(2.0, uExposure);
    c  = applyTemperature(c, uTemperature);
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(luma), c, uSaturation);
    c = (c - 0.5) * (1.0 + uContrast) + 0.5;
    c = max(c, 0.0);  // clamp negatives before display transform

    vec3 display;
    if (uUseLut) {
        // OCIO display transform — output is already display-encoded
        display = %OCIO_FUNC%(vec4(c, 1.0)).rgb;
    } else if (uTonemap) {
        display = linearToSrgb(filmic(c));
    } else {
        display = linearToSrgb(clamp(c, 0.0, 1.0));
    }

    fragColor = vec4(display, 1.0);
}
)glsl";

// ---------------------------------------------------------------------------
// Shader compilation
// ---------------------------------------------------------------------------
static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[2048]; glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
        std::fprintf(stderr, "Shader compile error:\n%s\n", buf);
    }
    return s;
}

static GLuint linkProgram(GLuint vert, const std::string& fragSrc)
{
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc.c_str());
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    glDeleteShader(frag);
    GLint ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[512]; glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        std::fprintf(stderr, "Shader link error: %s\n", buf);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

static GLuint g_vertShader = 0;  // shared across rebuilds

static void initQuad()
{
    float verts[] = {
        -1,-1, 0,1,
         1,-1, 1,1,
         1, 1, 1,0,
        -1,-1, 0,1,
         1, 1, 1,0,
        -1, 1, 0,0,
    };
    glGenVertexArrays(1, &g_quadVAO);
    glGenBuffers(1, &g_quadVBO);
    glBindVertexArray(g_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    glBindVertexArray(0);
}

// Build the fallback shader (no OCIO injection, ACES polynomial only).
static GLuint buildFallbackShader()
{
    std::string mainSrc = kFragMain;
    size_t pos = mainSrc.find("%OCIO_FUNC%");
    if (pos != std::string::npos)
        mainSrc.replace(pos, 11, "vec4");  // dead branch — uUseLut is false
    std::string full = std::string(kFragPreamble) + "\n" + mainSrc;
    return linkProgram(g_vertShader, full);
}

// ---------------------------------------------------------------------------
// OCIO — initialise config and build/rebuild display shader
// ---------------------------------------------------------------------------
#ifdef ANACAPA_HAVE_OCIO

static void ocioRefreshViews()
{
    g_ocioViews.clear();
    if (!g_ocioConfig || g_ocioDisplays.empty()) return;
    const char* disp = g_ocioDisplays[g_ocioDisplayIdx].c_str();
    int n = g_ocioConfig->getNumViews(disp);
    for (int i = 0; i < n; ++i)
        g_ocioViews.push_back(g_ocioConfig->getView(disp, i));
    g_ocioViewIdx = std::min(g_ocioViewIdx, (int)g_ocioViews.size() - 1);
    if (g_ocioViewIdx < 0) g_ocioViewIdx = 0;
}

static void ocioFreeTextures()
{
    for (auto& t : g_ocioTextures)
        if (t.id) glDeleteTextures(1, &t.id);
    g_ocioTextures.clear();
}

// Build the OCIO-enhanced shader for the current display/view selection.
// Returns true on success.  On failure leaves g_shader as the fallback.
static bool ocioRebuildShader()
{
    if (!g_ocioConfig || g_ocioDisplays.empty() || g_ocioViews.empty()) return false;

    const char* disp = g_ocioDisplays[g_ocioDisplayIdx].c_str();
    const char* view = g_ocioViews[g_ocioViewIdx].c_str();

    try {
        auto dt = OCIO::DisplayViewTransform::Create();
        dt->setSrc(OCIO::ROLE_SCENE_LINEAR);
        dt->setDisplay(disp);
        dt->setView(view);

        auto proc    = g_ocioConfig->getProcessor(dt);
        auto gpuProc = proc->getOptimizedGPUProcessor(OCIO::OPTIMIZATION_DEFAULT);

        auto shaderDesc = OCIO::GpuShaderDesc::CreateShaderDesc();
        shaderDesc->setLanguage(OCIO::GPU_LANGUAGE_GLSL_1_3);
        shaderDesc->setFunctionName("ocioDisplay");
        shaderDesc->setResourcePrefix("ocio_");
        gpuProc->extractGpuShaderInfo(shaderDesc);

        // Upload 1D/2D LUT textures (OCIO calls these "textures" vs "3DTextures")
        ocioFreeTextures();
        int unit = 1;  // unit 0 = uTex

        unsigned numTex = shaderDesc->getNumTextures();
        for (unsigned i = 0; i < numTex; ++i) {
            const char* texName     = nullptr;
            const char* samplerName = nullptr;
            unsigned width = 0, height = 0;
            OCIO::GpuShaderDesc::TextureType      channel    = OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL;
            OCIO::GpuShaderDesc::TextureDimensions dimensions = OCIO::GpuShaderDesc::TEXTURE_1D;
            OCIO::Interpolation                    interp     = OCIO::INTERP_LINEAR;
            shaderDesc->getTexture(i, texName, samplerName, width, height, channel, dimensions, interp);
            const float* values = nullptr;
            shaderDesc->getTextureValues(i, values);
            if (!values) continue;

            OcioTexture ot;
            ot.samplerName = samplerName ? samplerName : "";
            ot.unit        = unit++;

            GLenum internalFmt = (channel == OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL)
                                 ? GL_R32F : GL_RGB32F;
            GLenum fmt         = (channel == OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL)
                                 ? GL_RED : GL_RGB;
            GLenum filter      = (interp  == OCIO::INTERP_NEAREST) ? GL_NEAREST : GL_LINEAR;

            glGenTextures(1, &ot.id);
            if (dimensions == OCIO::GpuShaderDesc::TEXTURE_2D) {
                ot.target = GL_TEXTURE_2D;
                glBindTexture(GL_TEXTURE_2D, ot.id);
                glTexImage2D(GL_TEXTURE_2D, 0, internalFmt, width, height, 0, fmt, GL_FLOAT, values);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            } else {
                ot.target = GL_TEXTURE_1D;
                glBindTexture(GL_TEXTURE_1D, ot.id);
                glTexImage1D(GL_TEXTURE_1D, 0, internalFmt, width, 0, fmt, GL_FLOAT, values);
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, filter);
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, filter);
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            }
            glBindTexture(ot.target, 0);
            g_ocioTextures.push_back(ot);
        }

        // Upload 3D LUT textures
        unsigned num3D = shaderDesc->getNum3DTextures();
        for (unsigned i = 0; i < num3D; ++i) {
            const char* texName     = nullptr;
            const char* samplerName = nullptr;
            unsigned    edgeLen     = 0;
            OCIO::Interpolation interp = OCIO::INTERP_LINEAR;
            shaderDesc->get3DTexture(i, texName, samplerName, edgeLen, interp);
            const float* values = nullptr;
            shaderDesc->get3DTextureValues(i, values);
            if (!values) continue;

            OcioTexture ot;
            ot.target      = GL_TEXTURE_3D;
            ot.samplerName = samplerName ? samplerName : "";
            ot.unit        = unit++;

            glGenTextures(1, &ot.id);
            glBindTexture(GL_TEXTURE_3D, ot.id);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F,
                         edgeLen, edgeLen, edgeLen, 0, GL_RGB, GL_FLOAT, values);
            GLenum filter = (interp == OCIO::INTERP_NEAREST) ? GL_NEAREST : GL_LINEAR;
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filter);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, filter);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_3D, 0);
            g_ocioTextures.push_back(ot);
        }

        // Build the combined fragment shader: preamble + OCIO code + our main
        std::string fragSrc = std::string(kFragPreamble)
                            + "\n"
                            + shaderDesc->getShaderText()
                            + "\n";
        std::string mainSrc = kFragMain;
        size_t pos = mainSrc.find("%OCIO_FUNC%");
        if (pos != std::string::npos)
            mainSrc.replace(pos, 11, "ocioDisplay");
        fragSrc += mainSrc;

        GLuint newProg = linkProgram(g_vertShader, fragSrc);
        if (!newProg) {
            ocioFreeTextures();
            return false;
        }

        // Assign texture units for OCIO samplers (do this while program is active)
        glUseProgram(newProg);
        glUniform1i(glGetUniformLocation(newProg, "uTex"), 0);
        for (const auto& ot : g_ocioTextures) {
            GLint loc = glGetUniformLocation(newProg, ot.samplerName.c_str());
            if (loc >= 0) glUniform1i(loc, ot.unit);
        }
        glUseProgram(0);

        if (g_shader) glDeleteProgram(g_shader);
        g_shader = newProg;
        g_ocioStatus = std::string(disp) + " / " + view;
        return true;

    } catch (const OCIO::Exception& e) {
        g_ocioStatus = std::string("Error: ") + e.what();
        ocioFreeTextures();
        return false;
    }
}

static void ocioInit()
{
    try {
        // Try $OCIO env var first, then fall back to OCIO's built-in studio
        // config — same content as the Blender 4.x bundled config: AgX (and
        // AgX Log / Punchy), Filmic / Filmic Log, ACES 1.0 / 2.0, Khronos PBR
        // Neutral, False Color, Raw, Standard.  The CG config we used before
        // had ACES views only, missing Filmic and AgX.
        bool fromEnv = (std::getenv("OCIO") != nullptr);
        if (fromEnv) {
            g_ocioConfig    = OCIO::GetCurrentConfig();
            g_ocioConfigName = std::getenv("OCIO");
        } else {
            g_ocioConfig    = OCIO::Config::CreateFromBuiltinConfig(
                                  "ocio://studio-config-v2.2.0_aces-v1.3_ocio-v2.4");
            g_ocioConfigName = "Built-in Studio";
        }

        int n = g_ocioConfig->getNumDisplays();
        for (int i = 0; i < n; ++i)
            g_ocioDisplays.push_back(g_ocioConfig->getDisplay(i));

        // Locate the default display
        std::string defDisp = g_ocioConfig->getDefaultDisplay();
        for (int i = 0; i < (int)g_ocioDisplays.size(); ++i) {
            if (g_ocioDisplays[i] == defDisp) { g_ocioDisplayIdx = i; break; }
        }

        ocioRefreshViews();

        // Prefer "Filmic" as the initial view; fall back to the config default.
        // The built-in studio config defaults to "ACES 1.0 SDR-video" which is
        // too contrasty for HDR beauty buffers.  "Filmic" matches Blender's
        // default and handles the 16-stop dynamic range of sky renders.
        const std::vector<std::string> kPreferredViews = { "Filmic", "AgX" };
        bool viewFound = false;
        for (const auto& pref : kPreferredViews) {
            for (int i = 0; i < (int)g_ocioViews.size(); ++i) {
                if (g_ocioViews[i] == pref) { g_ocioViewIdx = i; viewFound = true; break; }
            }
            if (viewFound) break;
        }
        if (!viewFound) {
            std::string defView = g_ocioConfig->getDefaultView(defDisp.c_str());
            for (int i = 0; i < (int)g_ocioViews.size(); ++i) {
                if (g_ocioViews[i] == defView) { g_ocioViewIdx = i; break; }
            }
        }

        g_ocioStatus = "Loaded";
        g_ocioShaderDirty = true;

    } catch (const OCIO::Exception& e) {
        g_ocioConfig = nullptr;
        g_ocioStatus = std::string("Init failed: ") + e.what();
        g_ocioEnabled = false;
    }
}

#endif // ANACAPA_HAVE_OCIO

// ---------------------------------------------------------------------------
// FBO
// ---------------------------------------------------------------------------
static void ensureFBO(int w, int h)
{
    if (w == g_fboWidth && h == g_fboHeight && g_fbo) return;

    if (g_fbo)        { glDeleteFramebuffers(1, &g_fbo);    g_fbo = 0; }
    if (g_dstTexture) { glDeleteTextures(1, &g_dstTexture); g_dstTexture = 0; }

    glGenTextures(1, &g_dstTexture);
    glBindTexture(GL_TEXTURE_2D, g_dstTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_dstTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    g_fboWidth  = w;
    g_fboHeight = h;
}

// ---------------------------------------------------------------------------
// Slot state
// ---------------------------------------------------------------------------
constexpr int kNumSlots = 8;

struct SlotState {
    GLuint   srcTex     = 0;
    int      texW       = 0, texH = 0;
    bool     isHdr      = false;  // float HDR source (skip sRGB decode)
    bool     hasAlpha   = false;  // source has a valid alpha channel
    uint64_t lastMod    = 0;
    float    exposure   = 0.f;
    float    saturation = 1.f;
    float    contrast   = 0.f;
    float    temperature= 0.f;
    bool     toneMap    = true;   // fallback ACES (when OCIO disabled)
    int      channelMode = 0;    // 0=RGB, 1=R, 2=G, 3=B, 4=A
};

// ---------------------------------------------------------------------------
// processImage — render active slot through the color-grading shader into FBO
// ---------------------------------------------------------------------------
static void processImage(const SlotState& s, bool useOcio)
{
    if (!s.srcTex || s.texW == 0) return;
    ensureFBO(s.texW, s.texH);

    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glViewport(0, 0, g_fboWidth, g_fboHeight);
    glUseProgram(g_shader);

    // Source image on unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, s.srcTex);
    glUniform1i(glGetUniformLocation(g_shader, "uTex"), 0);

#ifdef ANACAPA_HAVE_OCIO
    // Bind OCIO LUT textures on units 1+
    if (useOcio) {
        for (const auto& ot : g_ocioTextures) {
            glActiveTexture(GL_TEXTURE0 + ot.unit);
            glBindTexture(ot.target, ot.id);
        }
    }
#endif

    glUniform1f(glGetUniformLocation(g_shader, "uExposure"),     s.exposure);
    glUniform1f(glGetUniformLocation(g_shader, "uSaturation"),  s.saturation);
    glUniform1f(glGetUniformLocation(g_shader, "uContrast"),    s.contrast);
    glUniform1f(glGetUniformLocation(g_shader, "uTemperature"), s.temperature);
    glUniform1i(glGetUniformLocation(g_shader, "uChannelMode"), s.channelMode);
#ifdef ANACAPA_HAVE_OCIO
    glUniform1i(glGetUniformLocation(g_shader, "uTonemap"),     useOcio ? 0 : 1);
#else
    glUniform1i(glGetUniformLocation(g_shader, "uTonemap"),     s.toneMap ? 1 : 0);
#endif
    glUniform1i(glGetUniformLocation(g_shader, "uUseLut"),      useOcio    ? 1 : 0);
    glUniform1i(glGetUniformLocation(g_shader, "uIsHdr"),       s.isHdr    ? 1 : 0);

    glBindVertexArray(g_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Restore texture unit 0
    glActiveTexture(GL_TEXTURE0);
}

// ---------------------------------------------------------------------------
// Texture upload — EXR via OIIO (scene-linear float); PNG/JPEG/HDR via stb
// ---------------------------------------------------------------------------
static bool hasExtCI(const char* path, const char* ext)
{
    size_t plen = std::strlen(path), elen = std::strlen(ext);
    if (plen < elen) return false;
    const char* tail = path + plen - elen;
    for (size_t i = 0; i < elen; ++i)
        if (std::tolower((unsigned char)tail[i]) != std::tolower((unsigned char)ext[i]))
            return false;
    return true;
}

static bool uploadTextureToSlot(const char* path, SlotState& s)
{
    if (s.srcTex) { glDeleteTextures(1, &s.srcTex); s.srcTex = 0; }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    bool ok = false;

    if (hasExtCI(path, ".exr")) {
        // EXR: load as scene-linear float via OIIO.  Anacapa writes a
        // multi-channel beauty/denoised/AOV layout (3, 6, 9, or 12
        // channels in one subimage), so we always read at the native
        // nchannels and copy the first three (R, G, B) into the
        // RGBA32F texture with alpha = 1.  Using a 4-float xstride
        // here used to overflow the destination by (nc - 4) * 4 bytes
        // per pixel when nc > 4, which corrupted the heap and crashed
        // the viewer on EXRs containing AOV layers.
        auto inp = OIIO::ImageInput::open(path);
        if (inp) {
            const OIIO::ImageSpec& spec = inp->spec();
            int w  = spec.width;
            int h  = spec.height;
            int nc = spec.nchannels;
            std::vector<float> buf(w * h * 4, 0.f);
            if (nc >= 3) {
                std::vector<float> row(static_cast<size_t>(w) * nc);
                bool readOk = true;
                for (int y = 0; y < h; ++y) {
                    int flippedY = h - 1 - y;
                    if (!inp->read_scanline(y + spec.y, 0,
                                            OIIO::TypeDesc::FLOAT,
                                            row.data())) {
                        readOk = false;
                        break;
                    }
                    float* dst = buf.data() + static_cast<size_t>(flippedY) * w * 4;
                    for (int x = 0; x < w; ++x) {
                        dst[x*4 + 0] = row[x*nc + 0];
                        dst[x*4 + 1] = row[x*nc + 1];
                        dst[x*4 + 2] = row[x*nc + 2];
                        dst[x*4 + 3] = (nc >= 4) ? row[x*nc + 3] : 1.f;
                    }
                }
                if (readOk) {
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0,
                                  GL_RGBA, GL_FLOAT, buf.data());
                    s.texW    = w;
                    s.texH    = h;
                    s.isHdr   = true;
                    s.hasAlpha = (nc >= 4);
                    ok = true;
                }
            }
            inp->close();
        }
    } else {
        // PNG / JPEG / HDR — use stb_image
        bool isHdr = stbi_is_hdr(path);
        int  w, h, ch;
        stbi_set_flip_vertically_on_load(1);
        if (isHdr) {
            float* data = stbi_loadf(path, &w, &h, &ch, 4);
            if (data) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, data);
                stbi_image_free(data);
                s.texW  = w; s.texH = h; s.isHdr = true; ok = true;
            }
        } else {
            unsigned char* data = stbi_load(path, &w, &h, &ch, 4);
            if (data) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
                stbi_image_free(data);
                s.texW  = w; s.texH = h; s.isHdr = false; ok = true;
            }
        }
    }

    if (!ok) { glDeleteTextures(1, &tex); return false; }
    s.srcTex = tex;
    return true;
}

// ---------------------------------------------------------------------------
// File mod time (nanosecond resolution)
// ---------------------------------------------------------------------------
static uint64_t fileModTime(const std::string& path)
{
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<uint64_t>(st.st_mtimespec.tv_sec) * 1'000'000'000ULL
           + static_cast<uint64_t>(st.st_mtimespec.tv_nsec);
#else
    return static_cast<uint64_t>(st.st_mtim.tv_sec) * 1'000'000'000ULL
           + static_cast<uint64_t>(st.st_mtim.tv_nsec);
#endif
}

// ---------------------------------------------------------------------------
// Save the current FBO contents to a PNG file
// ---------------------------------------------------------------------------
static bool saveFboPng(const char* path)
{
    if (!g_fbo || g_fboWidth == 0 || g_fboHeight == 0) return false;
    int w = g_fboWidth, h = g_fboHeight;
    std::vector<uint8_t> pixels(w * h * 4);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // Flip vertically (OpenGL origin is bottom-left)
    std::vector<uint8_t> flipped(w * h * 4);
    for (int row = 0; row < h; ++row)
        std::memcpy(flipped.data() + row * w * 4,
                    pixels.data() + (h - 1 - row) * w * 4, w * 4);
    return stbi_write_png(path, w, h, 4, flipped.data(), w * 4) != 0;
}

// ---------------------------------------------------------------------------
// Socket reader thread — runs while a renderer is connected.
// Reads IMAGE_OPEN / TILE / IMAGE_CLOSE messages and enqueues tile updates.
// ---------------------------------------------------------------------------
static bool recvAll(int fd, uint8_t* buf, size_t len) {
    while (len > 0) {
        ssize_t n = ::recv(fd, buf, len, MSG_WAITALL);
        if (n <= 0) return false;
        buf += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

static void socketReaderThread(int clientFd) {
    using namespace anacapa::proto;

    while (!g_live.readerStop.load()) {
        MsgHeader hdr{};
        if (!recvAll(clientFd, reinterpret_cast<uint8_t*>(&hdr), sizeof(hdr)))
            break;

        std::vector<uint8_t> payload(hdr.payloadLen);
        if (hdr.payloadLen > 0 &&
            !recvAll(clientFd, payload.data(), hdr.payloadLen))
            break;

        switch (static_cast<MsgType>(hdr.type)) {
        case IMAGE_OPEN: {
            if (hdr.payloadLen < 12) break;
            uint32_t w, h, spp = 0, hasAlpha = 0;
            std::memcpy(&w,        payload.data() + 0,  4);
            std::memcpy(&h,        payload.data() + 4,  4);
            std::memcpy(&spp,      payload.data() + 8,  4);
            if (hdr.payloadLen >= 16)
                std::memcpy(&hasAlpha, payload.data() + 12, 4);
            g_live.imageSpp  = spp;
            g_live.hasAlpha  = (hasAlpha != 0);
            g_live.tilesReceived.store(0);
            g_live.pixelsFilled.store(0);
            // If a pre-launch normalized crop was drawn, convert to pixels
            // and send SET_CROP to the renderer before it starts rendering tiles.
            // Send crop to renderer if one is active.  cropNormActive persists
            // across renders — don't clear it here so subsequent renders also
            // honour the crop window without the user having to redraw it.
            if (g_live.cropNormActive && w > 0 && h > 0) {
                // Convert stored norm to image pixels.  When the crop was drawn
                // pre-launch (cropNormIsPanel=true), the norm is panel-relative
                // and must be remapped through the letterbox transform to get
                // correct image-space pixels.  Image-relative norms (drawn during
                // a live render) convert directly with norm * image_size.
                float nx0 = g_live.cropNormX0, ny0 = g_live.cropNormY0;
                float nx1 = g_live.cropNormX1, ny1 = g_live.cropNormY1;
                if (g_live.cropNormIsPanel &&
                    g_live.cropPanelW > 0 && g_live.cropPanelH > 0) {
                    float s  = std::min(g_live.cropPanelW / float(w),
                                        g_live.cropPanelH / float(h));
                    float dW = w * s, dH = h * s;
                    float oX = (g_live.cropPanelW - dW) * 0.5f;
                    float oY = (g_live.cropPanelH - dH) * 0.5f;
                    auto cvt = [](float n, float pSz, float off, float disp) {
                        return std::max(0.f, std::min(1.f, (n * pSz - off) / disp));
                    };
                    nx0 = cvt(nx0, g_live.cropPanelW, oX, dW);
                    ny0 = cvt(ny0, g_live.cropPanelH, oY, dH);
                    nx1 = cvt(nx1, g_live.cropPanelW, oX, dW);
                    ny1 = cvt(ny1, g_live.cropPanelH, oY, dH);
                }
                uint32_t cx  = static_cast<uint32_t>(nx0 * w);
                uint32_t cy  = static_cast<uint32_t>(ny0 * h);
                uint32_t cx1 = static_cast<uint32_t>(nx1 * w);
                uint32_t cy1 = static_cast<uint32_t>(ny1 * h);
                if (cx1 > cx && cy1 > cy) {
                    g_live.cropX = cx; g_live.cropY = cy;
                    g_live.cropW = cx1 - cx; g_live.cropH = cy1 - cy;
                    g_live.cropActive = true;
                    // leave cropNormActive = true so the next render sees it too
                    std::vector<uint8_t> msg;
                    anacapa::proto::encodeCrop(msg, cx, cy, cx1-cx, cy1-cy);
                    sendCommand(msg);
                }
            } else if (g_live.cropActive && g_live.liveW > 0 && h > 0) {
                // cropActive set mid-render (norm not stored), carry it forward
                uint32_t cx  = static_cast<uint32_t>(float(g_live.cropX) / float(g_live.liveW) * w);
                uint32_t cy  = static_cast<uint32_t>(float(g_live.cropY) / float(g_live.liveH) * h);
                uint32_t cx1 = static_cast<uint32_t>(float(g_live.cropX + g_live.cropW) / float(g_live.liveW) * w);
                uint32_t cy1 = static_cast<uint32_t>(float(g_live.cropY + g_live.cropH) / float(g_live.liveH) * h);
                if (cx1 > cx && cy1 > cy) {
                    g_live.cropX = cx; g_live.cropY = cy;
                    g_live.cropW = cx1 - cx; g_live.cropH = cy1 - cy;
                    std::vector<uint8_t> msg;
                    anacapa::proto::encodeCrop(msg, cx, cy, cx1-cx, cy1-cy);
                    sendCommand(msg);
                }
            }
            // Signal main thread to allocate live texture
            TileUpdate init;
            init.x0 = 0; init.y0 = 0; init.w = w; init.h = h;
            init.rgba.clear();  // empty rgba = IMAGE_OPEN signal
            {
                std::lock_guard<std::mutex> lk(g_live.queueMtx);
                g_live.queue.push(std::move(init));
            }
            g_live.dirty.store(true);
            break;
        }
        case TILE: {
            if (hdr.payloadLen < 16) break;
            TileUpdate tu;
            std::memcpy(&tu.x0, payload.data() + 0,  4);
            std::memcpy(&tu.y0, payload.data() + 4,  4);
            std::memcpy(&tu.w,  payload.data() + 8,  4);
            std::memcpy(&tu.h,  payload.data() + 12, 4);
            uint32_t nFloats = tu.w * tu.h * 4;  // always RGBA
            if (hdr.payloadLen < 16 + nFloats * 4) break;
            // Flip rows to GL bottom-left convention; copy RGBA as-is
            const float* src = reinterpret_cast<const float*>(payload.data() + 16);
            tu.rgba.resize(tu.w * tu.h * 4);
            for (uint32_t row = 0; row < tu.h; ++row) {
                uint32_t srcRow = row;
                uint32_t dstRow = tu.h - 1 - row;
                for (uint32_t col = 0; col < tu.w; ++col) {
                    const float* s = src + (srcRow * tu.w + col) * 4;
                    float*       d = tu.rgba.data() + (dstRow * tu.w + col) * 4;
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
                }
            }
            g_live.tilesReceived.fetch_add(1);
            g_live.pixelsFilled.fetch_add(tu.w * tu.h);
            {
                std::lock_guard<std::mutex> lk(g_live.queueMtx);
                g_live.queue.push(std::move(tu));
            }
            g_live.dirty.store(true);
            break;
        }
        case IMAGE_CLOSE:
            g_live.done = true;
            g_live.dirty.store(true);
            break;
        default:
            break;
        }
    }
    // Renderer disconnected
    ::close(clientFd);
    g_live.clientFd = -1;
}

// Start the Unix socket server and accept one connection.
// Returns true if the server is listening (even before a client connects).
static bool startSocketServer(const std::string& sockPath) {
    ::unlink(sockPath.c_str());  // remove stale socket

    int sfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sfd < 0) {
        std::fprintf(stderr, "viewer: socket() failed: %s\n", std::strerror(errno));
        return false;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, sockPath.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::fprintf(stderr, "viewer: bind('%s') failed: %s\n",
                     sockPath.c_str(), std::strerror(errno));
        ::close(sfd);
        return false;
    }

    ::listen(sfd, 1);
    g_live.serverFd = sfd;

    // Accept connections in a background thread so the main loop isn't blocked
    std::thread([sockPath]() {
        while (!g_live.readerStop.load()) {
            // Non-blocking accept poll
            fd_set fds; FD_ZERO(&fds); FD_SET(g_live.serverFd, &fds);
            timeval tv{1, 0};
            if (::select(g_live.serverFd + 1, &fds, nullptr, nullptr, &tv) <= 0)
                continue;

            int cfd = ::accept(g_live.serverFd, nullptr, nullptr);
            if (cfd < 0) continue;

            // Store and launch reader
            g_live.clientFd = cfd;
            g_live.done     = false;
            g_live.paused   = false;
            g_live.readerThread = std::thread(socketReaderThread, cfd);
            g_live.readerThread.join();
        }
    }).detach();

    std::fprintf(stdout, "viewer: listening on %s\n", sockPath.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    CLI::App app{"viewer — Anacapa progressive render viewer"};

    std::string imagePath;
    int         pollMs = 500;
    std::string sockPath;

    app.add_option("image", imagePath, "PNG/JPEG/HDR/EXR file to watch (optional in socket mode)");
    app.add_option("--interval", pollMs, "File poll interval in milliseconds (default 500)")
       ->default_val(500);
    app.add_option("--listen", sockPath,
                   "Listen on this Unix socket path for renderer connections "
                   "(default: " + std::string(anacapa::proto::kDefaultSockPath) + ")")
       ->expected(0, 1)
       ->default_val("");

    CLI11_PARSE(app, argc, argv);

    // If --listen was given without a path, use the default
    const bool socketMode = (app.count("--listen") > 0 || !sockPath.empty());
    if (socketMode && sockPath.empty())
        sockPath = anacapa::proto::kDefaultSockPath;

    if (imagePath.empty() && !socketMode) {
        std::fprintf(stderr, "viewer: provide an image path or --listen\n");
        return 1;
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);

    std::string windowTitle = socketMode
        ? "Anacapa Viewer — waiting for renderer"
        : "Anacapa Viewer — " + imagePath;
    SDL_Window* window = SDL_CreateWindow(
        windowTitle.c_str(),
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 820,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GLContext glCtx = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, glCtx);
    SDL_GL_SetSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
        std::fprintf(stderr, "Failed to initialize glad\n");
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;

    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 4.f;
    ImGui::GetStyle().FrameRounding  = 3.f;

    ImGui_ImplSDL2_InitForOpenGL(window, glCtx);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // Build vertex shader (shared; fragment shader is rebuilt per OCIO view)
    g_vertShader = compileShader(GL_VERTEX_SHADER, kVertSrc);
    initQuad();

    // Build initial fallback shader (ACES polynomial)
    g_shader = buildFallbackShader();

#ifdef ANACAPA_HAVE_OCIO
    ocioInit();
#endif

    // -----------------------------------------------------------------------
    // Socket server (when --listen)
    // -----------------------------------------------------------------------
    if (socketMode)
        startSocketServer(sockPath);

    // -----------------------------------------------------------------------
    // Slot state
    // -----------------------------------------------------------------------
    SlotState slots[kNumSlots];
    int activeSlot = 0;
    int recordSlot = 0;

    // Load the watched file into slot 0 on startup if it already exists
    if (!imagePath.empty()) {
        uint64_t mod = fileModTime(imagePath);
        if (mod != 0) {
            uploadTextureToSlot(imagePath.c_str(), slots[0]);
            slots[0].lastMod = mod;
        }
    }

    uint64_t watchedMod = imagePath.empty() ? 0 : slots[0].lastMod;

    // -----------------------------------------------------------------------
    // UI state
    // -----------------------------------------------------------------------
    bool fitToWin = true;
    float zoom    = 1.0f;
    bool needsRedraw = true;

    // Dialog / panel visibility
    bool showColorMgmt = false;

    // File browsers
    ImGui::FileBrowser openBrowser;
    openBrowser.SetTitle("Open Image");
    openBrowser.SetTypeFilters({".exr", ".png", ".jpg", ".jpeg", ".hdr"});

    ImGui::FileBrowser saveBrowser(ImGuiFileBrowserFlags_EnterNewFilename |
                                   ImGuiFileBrowserFlags_CreateNewDir);
    saveBrowser.SetTitle("Save Processed Image");
    saveBrowser.SetTypeFilters({".png"});
    saveBrowser.SetInputName("output.png");

    bool pendingSave = false;  // true when saveBrowser confirmed a path

    auto lastPoll = std::chrono::steady_clock::now();
    bool running  = true;

    // Crop drag state (image-space pixels)
    bool  cropDragging = false;
    float cropDragStartX = 0, cropDragStartY = 0;

    while (running) {
        // ---- File poll (non-socket mode only) --------------------------------
        if (!socketMode && !imagePath.empty()) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPoll).count());
            if (elapsed >= pollMs) {
                lastPoll = now;
                uint64_t mod = fileModTime(imagePath);
                if (mod != 0 && mod != watchedMod) {
                    watchedMod = mod;
                    if (uploadTextureToSlot(imagePath.c_str(), slots[recordSlot]))
                        slots[recordSlot].lastMod = mod;
                    needsRedraw = true;
                }
            }
        }

        // ---- Drain socket tile queue (socket mode only) ----------------------
        if (socketMode && g_live.dirty.exchange(false)) {
            std::queue<TileUpdate> pending;
            {
                std::lock_guard<std::mutex> lk(g_live.queueMtx);
                std::swap(pending, g_live.queue);
            }
            while (!pending.empty()) {
                TileUpdate& tu = pending.front();
                if (tu.rgba.empty()) {
                    // IMAGE_OPEN — allocate live texture
                    uint32_t w = tu.w, h = tu.h;
                    // Render lands in the slot picked as the record target
                    // (Shift+1-8), not necessarily the one currently displayed.
                    int destSlot = recordSlot;
                    // Save old live texture ID before overwriting it.
                    GLuint oldLiveTex = g_live.liveTex;
                    g_live.liveTex = 0;
                    // Free whatever the destination slot currently holds.
                    // If that happens to be the old live texture, mark it
                    // consumed so we don't double-free below.
                    {
                        SlotState& ds = slots[destSlot];
                        if (ds.srcTex) {
                            if (ds.srcTex == oldLiveTex) oldLiveTex = 0;
                            glDeleteTextures(1, &ds.srcTex);
                            ds.srcTex = 0;
                        }
                    }
                    // Old live texture may still be alive in another slot
                    // (a prior render the user is comparing against).
                    // Only free it if no slot owns it any more.
                    if (oldLiveTex) {
                        bool stillOwned = false;
                        for (auto& sl : slots)
                            if (sl.srcTex == oldLiveTex) { stillOwned = true; break; }
                        if (!stillOwned) glDeleteTextures(1, &oldLiveTex);
                    }
                    glGenTextures(1, &g_live.liveTex);
                    glBindTexture(GL_TEXTURE_2D, g_live.liveTex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    std::vector<float> zeros(w * h * 4, 0.f);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0,
                                 GL_RGBA, GL_FLOAT, zeros.data());
                    g_live.liveW       = w;
                    g_live.liveH       = h;
                    g_live.totalPixels = w * h;
                    g_live.active      = true;
                    g_live.done        = false;
                    // Convert pre-launch panel-relative norm to image-display-relative
                    // so the overlay stays at the same screen position it was drawn.
                    if (g_live.cropNormIsPanel &&
                        g_live.cropPanelW > 0 && g_live.cropPanelH > 0) {
                        float s  = std::min(g_live.cropPanelW / float(w),
                                            g_live.cropPanelH / float(h));
                        float dW = w * s, dH = h * s;
                        float oX = (g_live.cropPanelW - dW) * 0.5f;
                        float oY = (g_live.cropPanelH - dH) * 0.5f;
                        auto cvt = [](float n, float pSz, float off, float disp) {
                            return std::max(0.f, std::min(1.f, (n * pSz - off) / disp));
                        };
                        g_live.cropNormX0 = cvt(g_live.cropNormX0, g_live.cropPanelW, oX, dW);
                        g_live.cropNormY0 = cvt(g_live.cropNormY0, g_live.cropPanelH, oY, dH);
                        g_live.cropNormX1 = cvt(g_live.cropNormX1, g_live.cropPanelW, oX, dW);
                        g_live.cropNormY1 = cvt(g_live.cropNormY1, g_live.cropPanelH, oY, dH);
                        g_live.cropNormIsPanel = false;
                    }
                    // Mirror into destSlot so color-grading pipeline sees it
                    slots[destSlot].srcTex   = g_live.liveTex;
                    slots[destSlot].texW     = static_cast<int>(w);
                    slots[destSlot].texH     = static_cast<int>(h);
                    slots[destSlot].isHdr    = true;
                    slots[destSlot].hasAlpha = g_live.hasAlpha;
                    SDL_SetWindowTitle(window, "Anacapa Viewer — rendering…");
                } else {
                    // TILE — subimage update
                    glBindTexture(GL_TEXTURE_2D, g_live.liveTex);
                    // y0 in wire format is top-left; GL texture is bottom-left
                    uint32_t glY = g_live.liveH - tu.y0 - tu.h;
                    glTexSubImage2D(GL_TEXTURE_2D, 0,
                                    static_cast<GLint>(tu.x0),
                                    static_cast<GLint>(glY),
                                    static_cast<GLsizei>(tu.w),
                                    static_cast<GLsizei>(tu.h),
                                    GL_RGBA, GL_FLOAT, tu.rgba.data());
                }
                pending.pop();
            }
            if (g_live.done)
                SDL_SetWindowTitle(window, "Anacapa Viewer — done");
            needsRedraw = true;
        }

        // ---- SDL events -----------------------------------------------------
        {
            int timeUntilPoll;
            {
                auto now = std::chrono::steady_clock::now();
                int el = static_cast<int>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPoll).count());
                timeUntilPoll = std::max(0, pollMs - el);
            }
            SDL_Event event;
            if (SDL_WaitEventTimeout(&event, timeUntilPoll) != 0) {
                do {
                    ImGui_ImplSDL2_ProcessEvent(&event);
                    if (event.type == SDL_QUIT) running = false;
                    if (event.type == SDL_KEYDOWN && !io.WantCaptureKeyboard) {
                        if (event.key.keysym.sym == SDLK_q) running = false;
                        if (event.key.keysym.sym == SDLK_r) {
                            auto& s = slots[activeSlot];
                            s.exposure = 0.f; s.saturation = 1.f;
                            s.contrast = 0.f; s.temperature = 0.f;
                            s.toneMap  = true;
                        }
                        if (event.key.keysym.sym == SDLK_o &&
                            (event.key.keysym.mod & KMOD_CTRL))
                            openBrowser.Open();
                        if (event.key.keysym.sym == SDLK_s &&
                            (event.key.keysym.mod & KMOD_CTRL))
                            saveBrowser.Open();
                        if (event.key.keysym.sym >= SDLK_1 &&
                            event.key.keysym.sym <= SDLK_8) {
                            int idx = event.key.keysym.sym - SDLK_1;
                            if (event.key.keysym.mod & KMOD_SHIFT)
                                recordSlot = idx;
                            else
                                activeSlot = idx;
                        }
                    }
                } while (SDL_PollEvent(&event));
                needsRedraw = true;
            }
        }

        if (!needsRedraw) continue;
        needsRedraw = false;

        // ---- OCIO shader rebuild (when view selection changed) ---------------
#ifdef ANACAPA_HAVE_OCIO
        bool ocioActive = false;
        if (g_ocioEnabled && g_ocioShaderDirty && g_ocioConfig) {
            GLuint oldShader = g_shader;
            bool ok = ocioRebuildShader();
            if (!ok) {
                // Rebuild failed — restore fallback
                if (g_shader != oldShader) glDeleteProgram(g_shader);
                g_shader = buildFallbackShader();
            }
            g_ocioShaderDirty = false;
        }
        ocioActive = g_ocioEnabled && (g_ocioConfig != nullptr);
#else
        bool ocioActive = false;
#endif

        // ---- Render into FBO ------------------------------------------------
        processImage(slots[activeSlot], ocioActive);

        // ---- ImGui frame ----------------------------------------------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // ---- Menubar --------------------------------------------------------
        float menubarH = 0.f;
        if (ImGui::BeginMainMenuBar()) {
            menubarH = ImGui::GetWindowHeight();

            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Open Image...", "Ctrl+O"))
                    openBrowser.Open();
                if (ImGui::MenuItem("Save Processed...", "Ctrl+S"))
                    saveBrowser.Open();
                ImGui::Separator();
                if (ImGui::MenuItem("Close Slot")) {
                    SlotState& s = slots[activeSlot];
                    if (s.srcTex) { glDeleteTextures(1, &s.srcTex); s.srcTex = 0; }
                    s.texW = s.texH = 0; s.lastMod = 0; s.isHdr = false;
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Quit", "Q")) running = false;
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Color")) {
                ImGui::MenuItem("Color Management", nullptr, &showColorMgmt);
                ImGui::EndMenu();
            }

            // Status bar right side — show current OCIO view or fallback label
#ifdef ANACAPA_HAVE_OCIO
            float statusW = ImGui::CalcTextSize(g_ocioStatus.c_str()).x + 16.f;
            ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - statusW);
            ImGui::TextDisabled("%s", g_ocioStatus.c_str());
#endif
            ImGui::EndMainMenuBar();
        }

        // ---- File browsers --------------------------------------------------
        openBrowser.Display();
        if (openBrowser.HasSelected()) {
            std::string path = openBrowser.GetSelected().string();
            uploadTextureToSlot(path.c_str(), slots[activeSlot]);
            openBrowser.ClearSelected();
            needsRedraw = true;
        }

        saveBrowser.Display();
        if (saveBrowser.HasSelected()) {
            std::string path = saveBrowser.GetSelected().string();
            // Ensure .png extension
            if (path.size() < 4 || path.substr(path.size()-4) != ".png")
                path += ".png";
            saveFboPng(path.c_str());
            saveBrowser.ClearSelected();
        }

        // ---- Color Management panel -----------------------------------------
        if (showColorMgmt) {
            ImGui::SetNextWindowSize({360, 0}, ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Color Management", &showColorMgmt)) {
#ifdef ANACAPA_HAVE_OCIO
                ImGui::TextDisabled("Config: %s", g_ocioConfigName.c_str());
                ImGui::Spacing();

                if (g_ocioConfig) {
                    ImGui::Checkbox("Enable OCIO display transform", &g_ocioEnabled);
                    if (!g_ocioEnabled)
                        ImGui::TextDisabled("Using built-in ACES polynomial fallback");

                    if (g_ocioEnabled) {
                        // Display selector
                        ImGui::AlignTextToFramePadding();
                        ImGui::TextUnformatted("Display:");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(-1);
                        if (ImGui::BeginCombo("##display",
                                g_ocioDisplays.empty() ? "—"
                                : g_ocioDisplays[g_ocioDisplayIdx].c_str())) {
                            for (int i = 0; i < (int)g_ocioDisplays.size(); ++i) {
                                bool sel = (i == g_ocioDisplayIdx);
                                if (ImGui::Selectable(g_ocioDisplays[i].c_str(), sel)) {
                                    if (i != g_ocioDisplayIdx) {
                                        g_ocioDisplayIdx = i;
                                        g_ocioViewIdx    = 0;
                                        ocioRefreshViews();
                                        g_ocioShaderDirty = true;
                                        needsRedraw = true;
                                    }
                                }
                                if (sel) ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        // View selector
                        ImGui::AlignTextToFramePadding();
                        ImGui::TextUnformatted("View:");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(-1);
                        if (ImGui::BeginCombo("##view",
                                g_ocioViews.empty() ? "—"
                                : g_ocioViews[g_ocioViewIdx].c_str())) {
                            for (int i = 0; i < (int)g_ocioViews.size(); ++i) {
                                bool sel = (i == g_ocioViewIdx);
                                if (ImGui::Selectable(g_ocioViews[i].c_str(), sel)) {
                                    if (i != g_ocioViewIdx) {
                                        g_ocioViewIdx     = i;
                                        g_ocioShaderDirty = true;
                                        needsRedraw = true;
                                    }
                                }
                                if (sel) ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        ImGui::Spacing();
                        ImGui::TextDisabled("Status: %s", g_ocioStatus.c_str());
                    }
                } else {
                    ImGui::TextColored({1,0.4f,0.4f,1}, "OCIO config unavailable");
                    ImGui::TextDisabled("%s", g_ocioStatus.c_str());
                    ImGui::TextDisabled("Set $OCIO env var and relaunch.");
                }
#else
                ImGui::TextDisabled("Built without OpenColorIO support.");
                ImGui::TextDisabled("Using built-in ACES polynomial.");
#endif
            }
            ImGui::End();
        }

        // ---- Left sidebar ---------------------------------------------------
        const float kPanelW  = 220.f;
        const float kStatusH = ImGui::GetFrameHeightWithSpacing() + 6.f;
        const float kContentH = io.DisplaySize.y - menubarH - kStatusH;
        ImGui::SetNextWindowPos({0, menubarH});
        ImGui::SetNextWindowSize({kPanelW, kContentH});
        ImGui::Begin("Controls", nullptr,
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        SlotState& as = slots[activeSlot];

        // Slot selector
        ImGui::SeparatorText("Slots");
        for (int i = 0; i < kNumSlots; ++i) {
            bool exists = slots[i].srcTex != 0;
            bool isView = (i == activeSlot);
            bool isRec  = (i == recordSlot);

            char label[32];
            std::snprintf(label, sizeof(label), "Slot %d", i + 1);

            if (isView) {
                ImGui::PushStyleColor(ImGuiCol_Button,        {0.2f,0.5f,0.9f,1});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.3f,0.6f,1.0f,1});
                ImGui::PushStyleColor(ImGuiCol_Text,          {1,1,1,1});
            } else if (!exists) {
                ImGui::PushStyleColor(ImGuiCol_Button,        {0.2f,0.2f,0.2f,1});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.3f,0.3f,0.3f,1});
                ImGui::PushStyleColor(ImGuiCol_Text,          {0.4f,0.4f,0.4f,1});
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        {0.3f,0.3f,0.35f,1});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.4f,0.4f,0.45f,1});
                ImGui::PushStyleColor(ImGuiCol_Text,          {0.9f,0.9f,0.9f,1});
            }
            float btnW = kPanelW - 16.f - 28.f;
            if (ImGui::Button(label, {btnW, 0})) activeSlot = i;
            ImGui::PopStyleColor(3);

            ImGui::SameLine();
            ImGui::PushID(i);
            if (isRec) {
                ImGui::PushStyleColor(ImGuiCol_Button,        {0.8f,0.2f,0.2f,1});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {1.0f,0.3f,0.3f,1});
                ImGui::PushStyleColor(ImGuiCol_Text,          {1,1,1,1});
                ImGui::Button("\xe2\x97\x8f", {24, 0});
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        {0.2f,0.2f,0.2f,1});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.4f,0.2f,0.2f,1});
                ImGui::PushStyleColor(ImGuiCol_Text,          {0.5f,0.5f,0.5f,1});
                if (ImGui::Button("\xe2\x97\x8b", {24, 0})) recordSlot = i;
            }
            ImGui::PopStyleColor(3);
            ImGui::PopID();
        }
        ImGui::TextDisabled("  view slot    rec");

        // Per-slot color controls
        ImGui::Spacing();
        ImGui::SeparatorText("Tone");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##exp", &as.exposure,  -4.f, 4.f, "Exposure: %.2f EV");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##con", &as.contrast,  -1.f, 1.f, "Contrast: %.2f");
#ifndef ANACAPA_HAVE_OCIO
        ImGui::Checkbox("Filmic", &as.toneMap);
#endif

        ImGui::SeparatorText("Color");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##sat", &as.saturation, 0.f, 2.f, "Saturation: %.2f");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##tmp", &as.temperature, -1.f, 1.f, "Temp: %.2f");

        ImGui::SeparatorText("Channels");
        {
            static const char* kChLabels[] = { "RGB", "R", "G", "B", "A" };
            float btnW = (kPanelW - 16.f - 4.f * 4.f) / 5.f;
            for (int i = 0; i < 5; ++i) {
                if (i > 0) ImGui::SameLine(0.f, 4.f);
                bool sel = (as.channelMode == i);
                if (sel) {
                    ImGui::PushStyleColor(ImGuiCol_Button,        {0.2f,0.5f,0.9f,1.f});
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.3f,0.6f,1.0f,1.f});
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Button,        {0.25f,0.25f,0.28f,1.f});
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.35f,0.35f,0.40f,1.f});
                }
                if (ImGui::Button(kChLabels[i], {btnW, 0}))
                    as.channelMode = i;
                ImGui::PopStyleColor(2);
            }
        }

        ImGui::SeparatorText("View");
        ImGui::Checkbox("Fit to window", &fitToWin);
        if (!fitToWin) {
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##zoom", &zoom, 0.1f, 8.f, "Zoom: %.2fx");
        }

        ImGui::Spacing();
        if (ImGui::Button("Reset  (R)", {-1, 0})) {
            as.exposure = 0.f; as.saturation = 1.f;
            as.contrast = 0.f; as.temperature = 0.f;
            as.toneMap  = true;
        }

        ImGui::Spacing();
        ImGui::SeparatorText("Info");
        if (as.texW)
            ImGui::TextDisabled("%d x %d%s", as.texW, as.texH, as.isHdr ? " HDR" : "");
        ImGui::TextDisabled("1-8: switch view");
        ImGui::TextDisabled("Shift+1-8: set rec slot");
        ImGui::TextDisabled("Ctrl+O: open  Ctrl+S: save");
        ImGui::TextDisabled("R: reset  Q: quit");

        ImGui::End();

        // ---- Image panel ----------------------------------------------------
        float imgX = kPanelW;
        float imgW = io.DisplaySize.x - kPanelW;
        float imgH = kContentH;

        ImGui::SetNextWindowPos({imgX, menubarH});
        ImGui::SetNextWindowSize({imgW, imgH});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##image", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PopStyleVar();

        if (g_dstTexture && as.texW > 0) {
            float dispW, dispH;
            if (fitToWin) {
                float s = std::min(imgW / float(as.texW), imgH / float(as.texH));
                dispW = as.texW * s;
                dispH = as.texH * s;
            } else {
                dispW = as.texW * zoom;
                dispH = as.texH * zoom;
            }
            float offX = (imgW - dispW) * 0.5f;
            float offY = (imgH - dispH) * 0.5f;
            if (offX > 0) ImGui::SetCursorPosX(offX);
            if (offY > 0) ImGui::SetCursorPosY(offY);

            ImVec2 imageScreenPos = ImGui::GetCursorScreenPos();
            ImGui::Image((ImTextureID)(intptr_t)g_dstTexture, {dispW, dispH});

            // ---- Crop drag over loaded image (pixel-precise) -----------------
            if (socketMode) {
                ImVec2 mp = ImGui::GetIO().MousePos;
                bool inImage = mp.x >= imageScreenPos.x &&
                               mp.x <= imageScreenPos.x + dispW &&
                               mp.y >= imageScreenPos.y &&
                               mp.y <= imageScreenPos.y + dispH;

                // Use IsAnyItemActive() so a drag that started on a UI widget
                // (slider, button) doesn't bleed into crop-drag.
                if (inImage && ImGui::IsMouseClicked(ImGuiMouseButton_Left)
                            && !ImGui::IsAnyItemActive()) {
                    cropDragging   = true;
                    cropDragStartX = mp.x;
                    cropDragStartY = mp.y;
                }
                if (cropDragging && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    cropDragging = false;
                    float x0s = std::min(cropDragStartX, mp.x);
                    float y0s = std::min(cropDragStartY, mp.y);
                    float x1s = std::max(cropDragStartX, mp.x);
                    float y1s = std::max(cropDragStartY, mp.y);
                    float scaleX = float(as.texW) / dispW;
                    float scaleY = float(as.texH) / dispH;
                    uint32_t cx  = static_cast<uint32_t>(std::max(0.f, (x0s - imageScreenPos.x) * scaleX));
                    uint32_t cy  = static_cast<uint32_t>(std::max(0.f, (y0s - imageScreenPos.y) * scaleY));
                    uint32_t cx1 = static_cast<uint32_t>(std::min(float(as.texW), (x1s - imageScreenPos.x) * scaleX));
                    uint32_t cy1 = static_cast<uint32_t>(std::min(float(as.texH), (y1s - imageScreenPos.y) * scaleY));
                    if (cx1 > cx && cy1 > cy) {
                        g_live.cropX = cx; g_live.cropY = cy;
                        g_live.cropW = cx1 - cx; g_live.cropH = cy1 - cy;
                        // Always store norm so the crop persists to the next render.
                        g_live.cropNormX0 = float(cx)  / float(as.texW);
                        g_live.cropNormY0 = float(cy)  / float(as.texH);
                        g_live.cropNormX1 = float(cx1) / float(as.texW);
                        g_live.cropNormY1 = float(cy1) / float(as.texH);
                        g_live.cropNormActive  = true;
                        g_live.cropNormIsPanel = false;  // image-display-relative
                        if (!g_live.done) {
                            // Renderer is running — also send pixel-precise SET_CROP now
                            g_live.cropActive = true;
                            std::vector<uint8_t> msg;
                            anacapa::proto::encodeCrop(msg, cx, cy, cx1-cx, cy1-cy);
                            sendCommand(msg);
                        } else {
                            g_live.cropActive = false;
                        }
                    } else if (g_live.cropActive || g_live.cropNormActive) {
                        // Click without a real drag — clear the crop
                        g_live.cropActive     = false;
                        g_live.cropNormActive = false;
                        if (!g_live.done) {
                            std::vector<uint8_t> msg;
                            anacapa::proto::encodeSimple(msg, anacapa::proto::CLEAR_CROP);
                            sendCommand(msg);
                        }
                    }
                }

                // Draw crop rect overlay.  Always use norm coords — they are
                // always set alongside cropActive, so there is no stale-texW risk.
                if (g_live.cropNormActive || g_live.cropActive || cropDragging) {
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    float x0s, y0s, x1s, y1s;
                    if (cropDragging) {
                        ImVec2 cmp = ImGui::GetIO().MousePos;
                        x0s = std::min(cropDragStartX, cmp.x);
                        y0s = std::min(cropDragStartY, cmp.y);
                        x1s = std::max(cropDragStartX, cmp.x);
                        y1s = std::max(cropDragStartY, cmp.y);
                    } else if (g_live.cropNormActive) {
                        x0s = imageScreenPos.x + g_live.cropNormX0 * dispW;
                        y0s = imageScreenPos.y + g_live.cropNormY0 * dispH;
                        x1s = imageScreenPos.x + g_live.cropNormX1 * dispW;
                        y1s = imageScreenPos.y + g_live.cropNormY1 * dispH;
                    } else {
                        float sx = dispW / float(as.texW);
                        float sy = dispH / float(as.texH);
                        x0s = imageScreenPos.x + g_live.cropX * sx;
                        y0s = imageScreenPos.y + g_live.cropY * sy;
                        x1s = x0s + g_live.cropW * sx;
                        y1s = y0s + g_live.cropH * sy;
                    }
                    dl->AddRect({x0s, y0s}, {x1s, y1s},
                                IM_COL32(255, 200, 0, 220), 0.f, 0, 1.5f);
                    dl->AddRectFilled({x0s, y0s}, {x1s, y1s},
                                      IM_COL32(255, 200, 0, 20));
                }
                needsRedraw = true;
            }
        } else {
            ImGui::SetCursorPos({20, 20});
            if (socketMode)
                ImGui::TextDisabled("Waiting for renderer on %s", sockPath.c_str());
            else
                ImGui::TextDisabled("Slot %d is empty", activeSlot + 1);

            // ---- Pre-launch crop drag (no image loaded yet) ------------------
            // Store as normalized [0,1] coords; converted to pixels on IMAGE_OPEN.
            if (socketMode) {
                ImVec2 panelOrigin = ImGui::GetWindowPos();
                ImVec2 mp = ImGui::GetIO().MousePos;
                bool inPanel = mp.x >= panelOrigin.x && mp.x <= panelOrigin.x + imgW &&
                               mp.y >= panelOrigin.y && mp.y <= panelOrigin.y + imgH;

                if (inPanel && ImGui::IsMouseClicked(ImGuiMouseButton_Left)
                             && !ImGui::IsAnyItemActive()) {
                    cropDragging   = true;
                    cropDragStartX = mp.x;
                    cropDragStartY = mp.y;
                }
                if (cropDragging && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    cropDragging = false;
                    float x0s = std::min(cropDragStartX, mp.x);
                    float y0s = std::min(cropDragStartY, mp.y);
                    float x1s = std::max(cropDragStartX, mp.x);
                    float y1s = std::max(cropDragStartY, mp.y);
                    if (x1s > x0s + 4.f && y1s > y0s + 4.f) {
                        g_live.cropNormX0 = (x0s - panelOrigin.x) / imgW;
                        g_live.cropNormY0 = (y0s - panelOrigin.y) / imgH;
                        g_live.cropNormX1 = (x1s - panelOrigin.x) / imgW;
                        g_live.cropNormY1 = (y1s - panelOrigin.y) / imgH;
                        g_live.cropNormActive  = true;
                        g_live.cropNormIsPanel = true;   // panel-relative coords
                        g_live.cropPanelW      = imgW;
                        g_live.cropPanelH      = imgH;
                        g_live.cropActive      = false;
                    }
                }

                // Draw pre-launch crop rect
                if (g_live.cropNormActive || cropDragging) {
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    float x0s, y0s, x1s, y1s;
                    if (cropDragging) {
                        ImVec2 cmp = ImGui::GetIO().MousePos;
                        x0s = std::min(cropDragStartX, cmp.x);
                        y0s = std::min(cropDragStartY, cmp.y);
                        x1s = std::max(cropDragStartX, cmp.x);
                        y1s = std::max(cropDragStartY, cmp.y);
                    } else {
                        x0s = panelOrigin.x + g_live.cropNormX0 * imgW;
                        y0s = panelOrigin.y + g_live.cropNormY0 * imgH;
                        x1s = panelOrigin.x + g_live.cropNormX1 * imgW;
                        y1s = panelOrigin.y + g_live.cropNormY1 * imgH;
                    }
                    dl->AddRect({x0s, y0s}, {x1s, y1s},
                                IM_COL32(255, 200, 0, 220), 0.f, 0, 1.5f);
                    dl->AddRectFilled({x0s, y0s}, {x1s, y1s},
                                      IM_COL32(255, 200, 0, 20));
                }
                needsRedraw = true;
            }
        }

        // ---- Render control toolbar (socket mode only) -----------------------
        if (socketMode && g_live.active && !g_live.done) {
            ImGui::SetNextWindowPos({imgX + 8.f, menubarH + 8.f});
            ImGui::SetNextWindowSize({0, 0});
            ImGui::Begin("##controls", nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoFocusOnAppearing);

            if (!g_live.paused) {
                if (ImGui::Button("Pause")) {
                    g_live.paused = true;
                    std::vector<uint8_t> msg;
                    anacapa::proto::encodeSimple(msg, anacapa::proto::PAUSE);
                    sendCommand(msg);
                }
            } else {
                if (ImGui::Button("Resume")) {
                    g_live.paused = false;
                    std::vector<uint8_t> msg;
                    anacapa::proto::encodeSimple(msg, anacapa::proto::RESUME);
                    sendCommand(msg);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                std::vector<uint8_t> msg;
                anacapa::proto::encodeSimple(msg, anacapa::proto::CANCEL);
                sendCommand(msg);
            }
            if (g_live.cropActive || g_live.cropNormActive) {
                ImGui::SameLine();
                if (ImGui::Button("Clear Crop")) {
                    g_live.cropActive     = false;
                    g_live.cropNormActive = false;
                    if (!g_live.done) {
                        std::vector<uint8_t> msg;
                        anacapa::proto::encodeSimple(msg, anacapa::proto::CLEAR_CROP);
                        sendCommand(msg);
                    }
                }
            }
            ImGui::End();
        }

        // ---- Clear Crop button when no render is active (pre-launch or post-render)
        // g_live.active is never cleared, so use !g_live.done && g_live.active to
        // detect an in-progress render; the button shows at all other times.
        bool renderInProgress = g_live.active && !g_live.done;
        if (socketMode && (g_live.cropNormActive || g_live.cropActive) && !renderInProgress) {
            ImGui::SetNextWindowPos({imgX + 8.f, menubarH + 8.f});
            ImGui::SetNextWindowSize({0, 0});
            ImGui::Begin("##precrop", nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoFocusOnAppearing);
            if (ImGui::Button("Clear Crop")) {
                g_live.cropNormActive = false;
                g_live.cropActive     = false;
            }
            ImGui::End();
        }

        ImGui::End();

        // ---- Status bar -----------------------------------------------------
        ImGui::SetNextWindowPos({0, menubarH + kContentH});
        ImGui::SetNextWindowSize({io.DisplaySize.x, kStatusH});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {8.f, 3.f});
        ImGui::PushStyleColor(ImGuiCol_WindowBg, {0.13f, 0.13f, 0.14f, 1.f});
        ImGui::Begin("##statusbar", nullptr,
            ImGuiWindowFlags_NoTitleBar  | ImGuiWindowFlags_NoResize  |
            ImGuiWindowFlags_NoMove      | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoDecoration);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();

        if (socketMode) {
            if (g_live.active && !g_live.done) {
                uint32_t tiles   = g_live.tilesReceived.load();
                uint64_t filled  = g_live.pixelsFilled.load();
                uint32_t total   = g_live.totalPixels;
                float    pct     = (total > 0) ? float(filled) / float(total) * 100.f : 0.f;
                if (g_live.paused)
                    ImGui::TextDisabled("Paused  %u tiles, %.1f%%", tiles, pct);
                else
                    ImGui::TextDisabled("Rendering  %u tiles, %.1f%%", tiles, pct);
            } else if (g_live.done) {
                uint32_t w = g_live.liveW, h = g_live.liveH, spp = g_live.imageSpp;
                ImGui::TextDisabled("Done  %ux%u, %u spp", w, h, spp);
            } else {
                ImGui::TextDisabled("Waiting for renderer on %s", sockPath.c_str());
            }
        } else {
            const SlotState& ss = slots[activeSlot];
            if (ss.texW > 0)
                ImGui::TextDisabled("Slot %d — %dx%d%s", activeSlot + 1,
                                    ss.texW, ss.texH, ss.isHdr ? "  HDR" : "");
            else
                ImGui::TextDisabled("Slot %d — empty", activeSlot + 1);
        }
        ImGui::End();

        // ---- Render ---------------------------------------------------------
        ImGui::Render();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(0.1f, 0.1f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);

        if (ImGui::IsAnyItemActive() || io.MouseDown[0] || io.WantCaptureMouse)
            needsRedraw = true;
    }

    // ---- Cleanup ------------------------------------------------------------
    // Socket server
    g_live.readerStop.store(true);
    if (g_live.serverFd >= 0) { ::close(g_live.serverFd); g_live.serverFd = -1; }
    if (g_live.clientFd >= 0) { ::close(g_live.clientFd); g_live.clientFd = -1; }
    if (g_live.readerThread.joinable()) g_live.readerThread.join();
    {
        GLuint liveId = g_live.liveTex;
        if (liveId) { glDeleteTextures(1, &liveId); g_live.liveTex = 0; }
        if (socketMode && !sockPath.empty()) ::unlink(sockPath.c_str());
        // Nullify live texture from slot before slot cleanup (already deleted above)
        for (int i = 0; i < kNumSlots; ++i)
            if (slots[i].srcTex == liveId) slots[i].srcTex = 0;
    }

    for (int i = 0; i < kNumSlots; ++i)
        if (slots[i].srcTex) glDeleteTextures(1, &slots[i].srcTex);
#ifdef ANACAPA_HAVE_OCIO
    ocioFreeTextures();
#endif
    if (g_dstTexture) glDeleteTextures(1, &g_dstTexture);
    if (g_fbo)        glDeleteFramebuffers(1, &g_fbo);
    if (g_shader)     glDeleteProgram(g_shader);
    if (g_vertShader) glDeleteShader(g_vertShader);
    if (g_quadVAO)    { glDeleteVertexArrays(1, &g_quadVAO); glDeleteBuffers(1, &g_quadVBO); }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(glCtx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
