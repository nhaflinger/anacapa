#pragma once

#include <anacapa/render/IDisplayDriver.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace anacapa {

class Film;

// ---------------------------------------------------------------------------
// FileDisplayDriver — writes progressive preview files to disk.
//
// Replaces the hardcoded preview thread in RenderSession.  A background
// thread wakes every kIntervalMs milliseconds and writes the preview when
// the film's dirty flag is set.  The final write is always flushed on
// imageClose() regardless of the timer.
//
// Two preview modes (same logic as the old RenderSession preview thread):
//   - PNG path supplied → sRGB-encoded PNG written every interval
//   - EXR output path  → lightweight beauty EXR written every interval
// If neither is configured, the driver is a no-op (passes through cleanly).
// ---------------------------------------------------------------------------
class FileDisplayDriver : public IDisplayDriver {
public:
    // film must outlive this driver.
    // previewPngPath: path for progressive PNG (empty = none).
    // previewExrPath: path for progressive EXR (empty = none).
    // exposure: EV stops for PNG tone mapping.
    FileDisplayDriver(Film& film,
                      std::string previewPngPath,
                      std::string previewExrPath,
                      float exposure = 0.f);
    ~FileDisplayDriver() override;

    void imageOpen(uint32_t width, uint32_t height) override;
    void writeTile(uint32_t x0, uint32_t y0,
                   uint32_t w,  uint32_t h,
                   const float* rgb) override;
    void imageClose() override;

private:
    void threadFunc();

    Film&       m_film;
    std::string m_pngPath;
    std::string m_exrPath;
    float       m_exposure;
    bool        m_active = false;  // true between imageOpen and imageClose

    std::thread             m_thread;
    std::mutex              m_mtx;
    std::condition_variable m_cv;
    std::atomic<bool>       m_stop{false};
};

} // namespace anacapa
