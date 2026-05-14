#pragma once

#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// IDisplayDriver — RenderMan-style display driver interface.
//
// The renderer calls imageOpen() once, writeTile() after each tile merges
// into the film, and imageClose() when the frame is complete.
//
// Viewer → renderer commands (onCrop, onCancel, onPause, onResume) are
// no-ops by default.  Implementations that support bidirectional
// communication (e.g. SocketDisplayDriver) override them; RenderSession
// polls for pending commands between tile dispatches.
//
// Thread safety: writeTile() may be called from multiple worker threads
// concurrently.  Implementations must synchronize internally.
// ---------------------------------------------------------------------------
class IDisplayDriver {
public:
    virtual ~IDisplayDriver() = default;

    // Called once before rendering begins.
    virtual void imageOpen(uint32_t width, uint32_t height) = 0;

    // Called after each tile is merged into the film.
    // rgb: row-major RGB float data, w*h*3 elements, linear scene-linear.
    // Thread-safe — may arrive from any worker thread.
    virtual void writeTile(uint32_t x0, uint32_t y0,
                           uint32_t w,  uint32_t h,
                           const float* rgb) = 0;

    // Called once after all tiles are complete (before final EXR write).
    virtual void imageClose() = 0;

    // ---------------------------------------------------------------------------
    // Viewer → renderer commands.  RenderSession calls pollCommands() between
    // tile dispatches; implementations that receive inbound messages should
    // trigger the appropriate on*() callback from within pollCommands().
    // Default implementations are no-ops so file-only drivers need not override.
    // ---------------------------------------------------------------------------

    // Check for pending inbound commands (non-blocking).  Called between tiles.
    virtual void pollCommands() {}

    // Re-prioritize the tile queue to render this pixel rect first.
    // Tiles outside the rect are not cancelled — they follow behind.
    virtual void onCrop(uint32_t /*x0*/, uint32_t /*y0*/,
                        uint32_t /*w*/,  uint32_t /*h*/) {}

    // Clear any active crop — restore normal tile order.
    virtual void onClearCrop() {}

    // Abort the current render.
    virtual void onCancel() {}

    // Suspend tile dispatch (render workers finish current tile then wait).
    virtual void onPause() {}

    // Resume after pause.
    virtual void onResume() {}
};

} // namespace anacapa
