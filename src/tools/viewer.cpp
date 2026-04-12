// viewer.cpp — Anacapa progressive render viewer
//
// Watches a PNG file on disk and displays it in a window, refreshing
// automatically as Anacapa updates it during a render.
// Color adjustments (exposure, saturation, contrast, temperature) are applied
// via a GL fragment shader in real time.
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

#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_HDR
#include "stb_image.h"

// ---------------------------------------------------------------------------
// Source texture (loaded from disk)
// ---------------------------------------------------------------------------
static GLuint g_srcTexture = 0;
static int    g_texWidth   = 0;
static int    g_texHeight  = 0;

// ---------------------------------------------------------------------------
// FBO + processed texture (output of the color-grading shader)
// ---------------------------------------------------------------------------
static GLuint g_fbo        = 0;
static GLuint g_dstTexture = 0;
static int    g_fboWidth   = 0;
static int    g_fboHeight  = 0;

// ---------------------------------------------------------------------------
// Color-grading shader
// ---------------------------------------------------------------------------
static GLuint g_shader  = 0;
static GLuint g_quadVAO = 0;
static GLuint g_quadVBO = 0;

static const char* kVertSrc = R"glsl(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() { vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }
)glsl";

static const char* kFragSrc = R"glsl(
#version 330 core
in  vec2 vUV;
out vec4 fragColor;

uniform sampler2D uTex;
uniform float uExposure;      // EV stops
uniform float uSaturation;    // 0=greyscale, 1=original, 2=vivid
uniform float uContrast;      // -1 to +1
uniform float uTemperature;   // -1=cool, 0=neutral, +1=warm

// sRGB decode (approximate)
vec3 srgbToLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
// sRGB encode (approximate)
vec3 linearToSrgb(vec3 c) { return pow(max(c, 0.0), vec3(1.0/2.2)); }

vec3 applyTemperature(vec3 c, float t) {
    // Warm: boost R, reduce B. Cool: boost B, reduce R.
    c.r *= 1.0 + t * 0.2;
    c.g *= 1.0 + t * 0.05;
    c.b *= 1.0 - t * 0.2;
    return c;
}

void main() {
    vec3 c = texture(uTex, vUV).rgb;

    // Work in linear light
    c = srgbToLinear(c);

    // Exposure
    c *= pow(2.0, uExposure);

    // Temperature
    c = applyTemperature(c, uTemperature);

    // Saturation (luma-preserving)
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(luma), c, uSaturation);

    // Contrast (pivot around 0.5 in linear)
    c = (c - 0.5) * (1.0 + uContrast) + 0.5;

    c = linearToSrgb(c);
    fragColor = vec4(c, 1.0);
}
)glsl";

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512]; glGetShaderInfoLog(s, 512, nullptr, buf);
        std::fprintf(stderr, "Shader compile error: %s\n", buf);
    }
    return s;
}

static void initShaderAndQuad()
{
    GLuint vert = compileShader(GL_VERTEX_SHADER,   kVertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, kFragSrc);
    g_shader = glCreateProgram();
    glAttachShader(g_shader, vert);
    glAttachShader(g_shader, frag);
    glLinkProgram(g_shader);
    glDeleteShader(vert);
    glDeleteShader(frag);

    // Full-screen quad: pos(xy) + uv
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

static void ensureFBO(int w, int h)
{
    if (w == g_fboWidth && h == g_fboHeight && g_fbo) return;

    if (g_fbo)        { glDeleteFramebuffers(1, &g_fbo);  g_fbo = 0; }
    if (g_dstTexture) { glDeleteTextures(1, &g_dstTexture); g_dstTexture = 0; }

    glGenTextures(1, &g_dstTexture);
    glBindTexture(GL_TEXTURE_2D, g_dstTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, g_dstTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    g_fboWidth  = w;
    g_fboHeight = h;
}

static void processImage(float exposure, float saturation,
                         float contrast, float temperature)
{
    if (!g_srcTexture || g_texWidth == 0) return;

    ensureFBO(g_texWidth, g_texHeight);

    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glViewport(0, 0, g_fboWidth, g_fboHeight);

    glUseProgram(g_shader);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_srcTexture);
    glUniform1i(glGetUniformLocation(g_shader, "uTex"),         0);
    glUniform1f(glGetUniformLocation(g_shader, "uExposure"),    exposure);
    glUniform1f(glGetUniformLocation(g_shader, "uSaturation"),  saturation);
    glUniform1f(glGetUniformLocation(g_shader, "uContrast"),    contrast);
    glUniform1f(glGetUniformLocation(g_shader, "uTemperature"), temperature);

    glBindVertexArray(g_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ---------------------------------------------------------------------------
// Texture upload
// ---------------------------------------------------------------------------
static void uploadTexture(const char* path)
{
    stbi_set_flip_vertically_on_load(1);
    int w, h, channels;
    unsigned char* data = stbi_load(path, &w, &h, &channels, 4);
    if (!data) return;

    if (!g_srcTexture) glGenTextures(1, &g_srcTexture);
    glBindTexture(GL_TEXTURE_2D, g_srcTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);
    g_texWidth  = w;
    g_texHeight = h;
    stbi_image_free(data);
}

// ---------------------------------------------------------------------------
// File mod time — nanosecond resolution to detect two writes within the same
// second (e.g. progressive preview write followed immediately by final write).
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
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    CLI::App app{"viewer — Anacapa progressive render viewer"};

    std::string imagePath;
    int         pollMs = 500;

    app.add_option("image", imagePath, "PNG/JPEG file to watch")->required();
    app.add_option("--interval", pollMs,
                   "Poll interval in milliseconds (default 500)")
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
        1280, 800,
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
    // Make the toolbar panel a bit more subtle
    ImGui::GetStyle().WindowRounding = 4.f;
    ImGui::GetStyle().FrameRounding  = 3.f;

    ImGui_ImplSDL2_InitForOpenGL(window, glCtx);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    initShaderAndQuad();

    // ---------------------------------------------------------------------------
    // Slot state — 8 independent slots, each with its own file, mod time,
    // color controls, and source texture.
    // ---------------------------------------------------------------------------
    constexpr int kNumSlots = 8;

    struct SlotState {
        GLuint      srcTex     = 0;
        int         texW       = 0, texH = 0;
        uint64_t    lastMod    = 0;
        float       exposure   = 0.f;
        float       saturation = 1.f;
        float       contrast   = 0.f;
        float       temperature= 0.f;
    };
    SlotState slots[kNumSlots];

    int activeSlot = 0;  // which slot is displayed and receives new renders

    // Load existing file into slot 0 on startup if it's already there
    {
        uint64_t mod = fileModTime(imagePath);
        if (mod != 0) {
            uploadTexture(imagePath.c_str());
            slots[0].srcTex  = g_srcTexture;
            slots[0].texW    = g_texWidth;
            slots[0].texH    = g_texHeight;
            slots[0].lastMod = mod;
            g_srcTexture = 0; g_texWidth = 0; g_texHeight = 0;
        }
    }

    uint64_t watchedMod = slots[0].lastMod;

    auto lastPoll = std::chrono::steady_clock::now();
    bool fitToWin = true;
    float zoom    = 1.0f;

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_q) running = false;
                if (event.key.keysym.sym == SDLK_r) {
                    auto& s = slots[activeSlot];
                    s.exposure = 0.f; s.saturation = 1.f;
                    s.contrast = 0.f; s.temperature = 0.f;
                }
                // Number keys 1-8 switch slots
                if (event.key.keysym.sym >= SDLK_1 && event.key.keysym.sym <= SDLK_8)
                    activeSlot = event.key.keysym.sym - SDLK_1;
            }
        }

        // Poll the watched file — on change, load into the active slot
        auto now = std::chrono::steady_clock::now();
        int elapsed = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - lastPoll).count());
        if (elapsed >= pollMs) {
            lastPoll = now;
            uint64_t mod = fileModTime(imagePath);
            if (mod != 0 && mod != watchedMod) {
                watchedMod = mod;
                SlotState& s = slots[activeSlot];
                if (s.srcTex) glDeleteTextures(1, &s.srcTex);
                g_srcTexture = 0; g_texWidth = 0; g_texHeight = 0;
                uploadTexture(imagePath.c_str());
                s.srcTex  = g_srcTexture;
                s.texW    = g_texWidth;
                s.texH    = g_texHeight;
                s.lastMod = mod;
                g_srcTexture = 0; g_texWidth = 0; g_texHeight = 0;
            }
        }

        // Apply color grading for active slot
        SlotState& as = slots[activeSlot];
        // Temporarily point the globals at the active slot so processImage works
        g_srcTexture = as.srcTex;
        g_texWidth   = as.texW;
        g_texHeight  = as.texH;
        processImage(as.exposure, as.saturation, as.contrast, as.temperature);
        g_srcTexture = 0; g_texWidth = 0; g_texHeight = 0;

        // ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // ---- Left sidebar ---------------------------------------------------
        const float kPanelW = 220.f;
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({kPanelW, io.DisplaySize.y});
        ImGui::Begin("Controls", nullptr,
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        // Slot selector
        ImGui::SeparatorText("Slots");
        for (int i = 0; i < kNumSlots; ++i) {
            bool exists = slots[i].srcTex != 0;
            char label[32];
            std::snprintf(label, sizeof(label), "Slot %d", i + 1);

            if (i == activeSlot) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.2f,0.5f,0.9f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f,0.6f,1.0f,1));
                ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(1,1,1,1));
            } else if (!exists) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.2f,0.2f,0.2f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f,0.3f,0.3f,1));
                ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.4f,0.4f,0.4f,1));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.3f,0.3f,0.35f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f,0.4f,0.45f,1));
                ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.9f,0.9f,0.9f,1));
            }
            if (ImGui::Button(label, {-1, 0}))
                activeSlot = i;
            ImGui::PopStyleColor(3);
        }

        // Color controls for active slot
        ImGui::Spacing();
        ImGui::SeparatorText("Tone");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##exp", &as.exposure,    -4.f, 4.f, "Exposure: %.2f EV");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##con", &as.contrast,    -1.f, 1.f, "Contrast: %.2f");

        ImGui::SeparatorText("Color");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##sat", &as.saturation,   0.f, 2.f, "Saturation: %.2f");
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
        }

        ImGui::Spacing();
        ImGui::SeparatorText("Info");
        if (as.texW)
            ImGui::TextDisabled("%d x %d", as.texW, as.texH);
        ImGui::TextDisabled("Keys 1-8: switch slot");
        ImGui::TextDisabled("R: reset color  Q: quit");

        ImGui::End();

        // ---- Image panel ---------------------------------------------------
        float imgX = kPanelW;
        float imgW = io.DisplaySize.x - kPanelW;
        float imgH = io.DisplaySize.y;

        ImGui::SetNextWindowPos({imgX, 0});
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
                float sx = imgW / float(as.texW);
                float sy = imgH / float(as.texH);
                float s  = sx < sy ? sx : sy;
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
            ImGui::TextDisabled("Slot %d is empty — select it before rendering",
                                activeSlot + 1);
        }
        ImGui::End();

        // Render
        ImGui::Render();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(0.1f, 0.1f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    for (int i = 0; i < kNumSlots; ++i)
        if (slots[i].srcTex) glDeleteTextures(1, &slots[i].srcTex);
    if (g_dstTexture) glDeleteTextures(1, &g_dstTexture);
    if (g_fbo)        glDeleteFramebuffers(1, &g_fbo);
    if (g_shader)     glDeleteProgram(g_shader);
    if (g_quadVAO)    { glDeleteVertexArrays(1, &g_quadVAO); glDeleteBuffers(1, &g_quadVBO); }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(glCtx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
