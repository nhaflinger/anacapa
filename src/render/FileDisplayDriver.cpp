#include <anacapa/render/FileDisplayDriver.h>
#include <anacapa/film/Film.h>
#include <spdlog/spdlog.h>

namespace anacapa {

static constexpr int kIntervalMs = 500;

FileDisplayDriver::FileDisplayDriver(Film& film,
                                     std::string previewPngPath,
                                     std::string previewExrPath,
                                     float exposure)
    : m_film(film)
    , m_pngPath(std::move(previewPngPath))
    , m_exrPath(std::move(previewExrPath))
    , m_exposure(exposure)
{}

FileDisplayDriver::~FileDisplayDriver() {
    // Ensure thread is stopped if imageClose() was never called
    if (m_thread.joinable()) {
        m_stop.store(true);
        m_cv.notify_all();
        m_thread.join();
    }
}

void FileDisplayDriver::imageOpen(uint32_t /*width*/, uint32_t /*height*/) {
    if (m_pngPath.empty() && m_exrPath.empty())
        return;

    m_active = true;
    m_stop.store(false);
    m_thread = std::thread(&FileDisplayDriver::threadFunc, this);

    if (!m_pngPath.empty())
        spdlog::info("Progressive preview: writing PNG every {} ms to '{}'",
                     kIntervalMs, m_pngPath);
    else
        spdlog::info("Progressive preview: writing EXR every {} ms to '{}'",
                     kIntervalMs, m_exrPath);
}

void FileDisplayDriver::writeTile(uint32_t /*x0*/, uint32_t /*y0*/,
                                  uint32_t /*w*/,  uint32_t /*h*/,
                                  const float* /*rgba*/) {
    // The film is already updated by the time this is called (mergeTile
    // happens before writeTile).  The preview thread picks up m_film.isDirty().
    // Nothing to do here — file writes are timer-driven.
}

void FileDisplayDriver::imageClose() {
    if (!m_active) return;

    // Stop the timer thread
    m_stop.store(true);
    m_cv.notify_all();
    if (m_thread.joinable())
        m_thread.join();

    // Final flush — write regardless of dirty flag
    if (!m_pngPath.empty())
        m_film.writePNG(m_pngPath, m_exposure);
    else if (!m_exrPath.empty())
        m_film.writeEXRPreview(m_exrPath);

    m_active = false;
}

void FileDisplayDriver::threadFunc() {
    while (true) {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cv.wait_for(lk, std::chrono::milliseconds(kIntervalMs),
                      [this] { return m_stop.load(); });

        if (m_film.isDirty()) {
            m_film.clearDirty();
            if (!m_pngPath.empty())
                m_film.writePNG(m_pngPath, m_exposure);
            else
                m_film.writeEXRPreview(m_exrPath);
        }

        if (m_stop.load()) break;
    }
}

} // namespace anacapa
