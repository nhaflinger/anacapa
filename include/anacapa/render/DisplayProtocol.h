#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Anacapa display driver wire protocol — simple length-prefixed binary frames.
//
// Every message:
//   [uint32_t type][uint32_t payloadLen][payloadLen bytes]
//
// Renderer → Viewer:
//   IMAGE_OPEN  payload: uint32_t width, height, spp
//   TILE        payload: uint32_t x0, y0, w, h  then w*h*3 floats (linear RGB)
//   IMAGE_CLOSE payload: (empty)
//
// Viewer → Renderer:
//   SET_CROP    payload: uint32_t x0, y0, w, h
//   CLEAR_CROP  payload: (empty)
//   CANCEL      payload: (empty)
//   PAUSE       payload: (empty)
//   RESUME      payload: (empty)
//
// Default socket path (Unix domain socket):
//   ANACAPA_DISPLAY_SOCK
// ---------------------------------------------------------------------------

namespace anacapa {
namespace proto {

constexpr const char* kDefaultSockPath = "/tmp/anacapa_viewer.sock";

enum MsgType : uint32_t {
    // Renderer → Viewer
    IMAGE_OPEN  = 0x01,
    TILE        = 0x02,
    IMAGE_CLOSE = 0x03,
    // Viewer → Renderer
    SET_CROP    = 0x10,
    CLEAR_CROP  = 0x11,
    CANCEL      = 0x12,
    PAUSE       = 0x13,
    RESUME      = 0x14,
};

struct MsgHeader {
    uint32_t type;
    uint32_t payloadLen;
};

// ---------------------------------------------------------------------------
// Encode helpers — append a message into a byte vector
// ---------------------------------------------------------------------------

inline void encodeImageOpen(std::vector<uint8_t>& buf,
                             uint32_t w, uint32_t h, uint32_t spp) {
    MsgHeader hdr{ IMAGE_OPEN, 12 };
    buf.resize(sizeof(hdr) + 12);
    std::memcpy(buf.data(), &hdr, sizeof(hdr));
    std::memcpy(buf.data() + sizeof(hdr) + 0,  &w,   4);
    std::memcpy(buf.data() + sizeof(hdr) + 4,  &h,   4);
    std::memcpy(buf.data() + sizeof(hdr) + 8,  &spp, 4);
}

inline void encodeImageClose(std::vector<uint8_t>& buf) {
    MsgHeader hdr{ IMAGE_CLOSE, 0 };
    buf.resize(sizeof(hdr));
    std::memcpy(buf.data(), &hdr, sizeof(hdr));
}

inline void encodeTile(std::vector<uint8_t>& buf,
                       uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                       const float* rgb) {
    uint32_t pixBytes = w * h * 3 * sizeof(float);
    uint32_t payLen   = 16 + pixBytes;
    buf.resize(sizeof(MsgHeader) + payLen);
    uint8_t* p = buf.data();
    MsgHeader hdr{ TILE, payLen };
    std::memcpy(p, &hdr, sizeof(hdr)); p += sizeof(hdr);
    std::memcpy(p, &x0, 4); p += 4;
    std::memcpy(p, &y0, 4); p += 4;
    std::memcpy(p, &w,  4); p += 4;
    std::memcpy(p, &h,  4); p += 4;
    std::memcpy(p, rgb, pixBytes);
}

inline void encodeCrop(std::vector<uint8_t>& buf,
                       uint32_t x0, uint32_t y0, uint32_t w, uint32_t h) {
    MsgHeader hdr{ SET_CROP, 16 };
    buf.resize(sizeof(hdr) + 16);
    uint8_t* p = buf.data();
    std::memcpy(p, &hdr, sizeof(hdr)); p += sizeof(hdr);
    std::memcpy(p, &x0, 4); p += 4;
    std::memcpy(p, &y0, 4); p += 4;
    std::memcpy(p, &w,  4); p += 4;
    std::memcpy(p, &h,  4);
}

inline void encodeSimple(std::vector<uint8_t>& buf, MsgType type) {
    MsgHeader hdr{ static_cast<uint32_t>(type), 0 };
    buf.resize(sizeof(hdr));
    std::memcpy(buf.data(), &hdr, sizeof(hdr));
}

} // namespace proto
} // namespace anacapa
