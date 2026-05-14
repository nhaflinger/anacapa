#pragma once

#include <anacapa/render/IDisplayDriver.h>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// SocketDisplayDriver — pushes tile data to the viewer over a Unix socket.
//
// The renderer connects to the viewer (viewer is the server).  writeTile()
// is thread-safe — tiles from multiple worker threads are serialized through
// an internal mutex before being sent.
//
// pollCommands() does a non-blocking read for inbound viewer commands and
// fires the appropriate on*() callbacks registered via setCommandHandler().
// RenderSession calls pollCommands() between tile dispatches.
//
// Usage:
//   auto drv = std::make_unique<SocketDisplayDriver>("/tmp/anacapa_viewer.sock");
//   if (!drv->isConnected()) { /* viewer not running — fall back */ }
//   session.setDisplayDriver(drv.get());
// ---------------------------------------------------------------------------
class SocketDisplayDriver : public IDisplayDriver {
public:
    explicit SocketDisplayDriver(std::string sockPath);
    ~SocketDisplayDriver() override;

    bool isConnected() const { return m_fd >= 0; }

    // IDisplayDriver
    void imageOpen(uint32_t width, uint32_t height) override;
    void writeTile(uint32_t x0, uint32_t y0,
                   uint32_t w,  uint32_t h,
                   const float* rgb) override;
    void imageClose() override;
    void pollCommands() override;

    // Command callbacks — called from pollCommands() on the calling thread.
    // Default: delegate to IDisplayDriver virtual methods (for subclassing).
    // Override by setting these functors if you prefer composition.
    std::function<void(uint32_t,uint32_t,uint32_t,uint32_t)> onCropFn;
    std::function<void()> onClearCropFn;
    std::function<void()> onCancelFn;
    std::function<void()> onPauseFn;
    std::function<void()> onResumeFn;

private:
    bool sendAll(const uint8_t* data, size_t len);

    std::string m_sockPath;
    int         m_fd = -1;
    std::mutex  m_sendMtx;
    uint32_t    m_width  = 0;
    uint32_t    m_height = 0;

    // Partial receive buffer for inbound command messages.
    // Protected by m_pollMtx — pollCommands() can be called from worker threads.
    std::mutex          m_pollMtx;
    std::vector<uint8_t> m_recvBuf;
};

} // namespace anacapa
