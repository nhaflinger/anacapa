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
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <sys/stat.h>

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

// sRGB decode/encode (approximate gamma 2.2)
vec3 srgbToLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 linearToSrgb(vec3 c) { return pow(max(c, 0.0), vec3(1.0/2.2)); }

// ACES RRT+ODT approximation (Narkowicz 2016) — fallback when OCIO unavailable
vec3 aces(vec3 x) {
    x *= 0.6;
    const float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    return clamp((x*(a*x+b))/(x*(c2*x+d)+e), 0.0, 1.0);
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
    vec3 c = texture(uTex, vUV).rgb;

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

    if (uUseLut) {
        // OCIO display transform — output is already display-encoded
        fragColor = vec4(%OCIO_FUNC%(vec4(c, 1.0)).rgb, 1.0);
    } else if (uTonemap) {
        fragColor = vec4(linearToSrgb(aces(c)), 1.0);
    } else {
        fragColor = vec4(linearToSrgb(clamp(c, 0.0, 1.0)), 1.0);
    }
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

        // Try to select the default view
        std::string defView = g_ocioConfig->getDefaultView(defDisp.c_str());
        for (int i = 0; i < (int)g_ocioViews.size(); ++i) {
            if (g_ocioViews[i] == defView) { g_ocioViewIdx = i; break; }
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
    uint64_t lastMod    = 0;
    float    exposure   = 0.f;
    float    saturation = 1.f;
    float    contrast   = 0.f;
    float    temperature= 0.f;
    bool     toneMap    = false;  // fallback ACES (when OCIO disabled)
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

    glUniform1f(glGetUniformLocation(g_shader, "uExposure"),    s.exposure);
    glUniform1f(glGetUniformLocation(g_shader, "uSaturation"),  s.saturation);
    glUniform1f(glGetUniformLocation(g_shader, "uContrast"),    s.contrast);
    glUniform1f(glGetUniformLocation(g_shader, "uTemperature"), s.temperature);
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
        // EXR: load as scene-linear float via OIIO
        auto inp = OIIO::ImageInput::open(path);
        if (inp) {
            const OIIO::ImageSpec& spec = inp->spec();
            int w  = spec.width;
            int h  = spec.height;
            int nc = spec.nchannels;
            std::vector<float> buf(w * h * 4, 0.f);
            // Read into RGBA float, flipping vertically for OpenGL
            if (nc >= 3) {
                std::vector<float> row(w * 4);
                for (int y = 0; y < h; ++y) {
                    int flippedY = h - 1 - y;
                    inp->read_scanline(y + spec.y, 0, OIIO::TypeDesc::FLOAT, row.data(),
                                       4 * sizeof(float));
                    std::memcpy(buf.data() + flippedY * w * 4, row.data(), w * 4 * sizeof(float));
                }
            }
            inp->close();
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, buf.data());
            s.texW  = w;
            s.texH  = h;
            s.isHdr = true;
            ok = true;
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
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    CLI::App app{"viewer — Anacapa progressive render viewer"};

    std::string imagePath;
    int         pollMs = 500;

    app.add_option("image", imagePath, "PNG/JPEG/HDR file to watch")->required();
    app.add_option("--interval", pollMs, "Poll interval in milliseconds (default 500)")
       ->default_val(500);

    CLI11_PARSE(app, argc, argv);

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

    SDL_Window* window = SDL_CreateWindow(
        ("Anacapa Viewer — " + imagePath).c_str(),
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
    // Slot state
    // -----------------------------------------------------------------------
    SlotState slots[kNumSlots];
    int activeSlot = 0;
    int recordSlot = 0;

    // Load the watched file into slot 0 on startup if it already exists
    {
        uint64_t mod = fileModTime(imagePath);
        if (mod != 0) {
            uploadTextureToSlot(imagePath.c_str(), slots[0]);
            slots[0].lastMod = mod;
        }
    }

    uint64_t watchedMod = slots[0].lastMod;

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

    while (running) {
        // ---- File poll -------------------------------------------------------
        {
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
                            s.toneMap  = false;
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
        const float kPanelW = 220.f;
        ImGui::SetNextWindowPos({0, menubarH});
        ImGui::SetNextWindowSize({kPanelW, io.DisplaySize.y - menubarH});
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
        ImGui::Checkbox("ACES Filmic", &as.toneMap);
#endif

        ImGui::SeparatorText("Color");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##sat", &as.saturation, 0.f, 2.f, "Saturation: %.2f");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##tmp", &as.temperature, -1.f, 1.f, "Temp: %.2f");

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
            as.toneMap  = false;
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
        float imgH = io.DisplaySize.y - menubarH;

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
            ImGui::Image((ImTextureID)(intptr_t)g_dstTexture, {dispW, dispH});
        } else {
            ImGui::SetCursorPos({20, 20});
            ImGui::TextDisabled("Slot %d is empty", activeSlot + 1);
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
