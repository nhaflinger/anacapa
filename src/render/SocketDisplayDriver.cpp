#include <anacapa/render/SocketDisplayDriver.h>
#include <anacapa/render/DisplayProtocol.h>

#include <spdlog/spdlog.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>

namespace anacapa {

SocketDisplayDriver::SocketDisplayDriver(std::string sockPath)
    : m_sockPath(std::move(sockPath))
{
    m_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (m_fd < 0) {
        spdlog::warn("SocketDisplayDriver: socket() failed: {}", std::strerror(errno));
        return;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, m_sockPath.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(m_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        spdlog::warn("SocketDisplayDriver: connect to '{}' failed: {} "
                     "(is the viewer running?)", m_sockPath, std::strerror(errno));
        ::close(m_fd);
        m_fd = -1;
        return;
    }

    spdlog::info("SocketDisplayDriver: connected to viewer at '{}'", m_sockPath);
}

SocketDisplayDriver::~SocketDisplayDriver() {
    if (m_fd >= 0) {
        ::close(m_fd);
        m_fd = -1;
    }
}

bool SocketDisplayDriver::sendAll(const uint8_t* data, size_t len) {
    while (len > 0) {
        ssize_t n = ::send(m_fd, data, len, MSG_NOSIGNAL);
        if (n <= 0) {
            spdlog::warn("SocketDisplayDriver: send failed: {}",
                         n < 0 ? std::strerror(errno) : "connection closed");
            ::close(m_fd);
            m_fd = -1;
            return false;
        }
        data += n;
        len  -= static_cast<size_t>(n);
    }
    return true;
}

void SocketDisplayDriver::imageOpen(uint32_t width, uint32_t height) {
    m_width  = width;
    m_height = height;
    if (m_fd < 0) return;

    std::vector<uint8_t> buf;
    proto::encodeImageOpen(buf, width, height, 0, m_hasAlpha ? 1u : 0u);
    std::lock_guard<std::mutex> lk(m_sendMtx);
    sendAll(buf.data(), buf.size());
}

void SocketDisplayDriver::writeTile(uint32_t x0, uint32_t y0,
                                    uint32_t w,  uint32_t h,
                                    const float* rgba) {
    if (m_fd < 0) return;
    std::vector<uint8_t> buf;
    proto::encodeTile(buf, x0, y0, w, h, rgba);
    std::lock_guard<std::mutex> lk(m_sendMtx);
    sendAll(buf.data(), buf.size());
}

void SocketDisplayDriver::imageClose() {
    if (m_fd < 0) return;
    std::vector<uint8_t> buf;
    proto::encodeImageClose(buf);
    std::lock_guard<std::mutex> lk(m_sendMtx);
    sendAll(buf.data(), buf.size());
}

void SocketDisplayDriver::pollCommands() {
    if (m_fd < 0) return;

    // Only one thread at a time — worker threads call this concurrently.
    std::unique_lock<std::mutex> lk(m_pollMtx, std::try_to_lock);
    if (!lk.owns_lock()) return;

    // Non-blocking peek — set O_NONBLOCK temporarily
    int flags = ::fcntl(m_fd, F_GETFL, 0);
    ::fcntl(m_fd, F_SETFL, flags | O_NONBLOCK);

    uint8_t tmp[256];
    ssize_t n = ::recv(m_fd, tmp, sizeof(tmp), 0);

    ::fcntl(m_fd, F_SETFL, flags);  // restore blocking

    if (n <= 0) return;  // nothing waiting or error; try again next tile

    // Append to partial buffer and try to parse complete messages
    m_recvBuf.insert(m_recvBuf.end(), tmp, tmp + n);

    while (m_recvBuf.size() >= sizeof(proto::MsgHeader)) {
        proto::MsgHeader hdr;
        std::memcpy(&hdr, m_recvBuf.data(), sizeof(hdr));

        size_t total = sizeof(hdr) + hdr.payloadLen;
        if (m_recvBuf.size() < total) break;  // wait for rest

        const uint8_t* pay = m_recvBuf.data() + sizeof(hdr);

        switch (static_cast<proto::MsgType>(hdr.type)) {
        case proto::SET_CROP:
            if (hdr.payloadLen >= 16) {
                uint32_t cx, cy, cw, ch;
                std::memcpy(&cx, pay + 0, 4);
                std::memcpy(&cy, pay + 4, 4);
                std::memcpy(&cw, pay + 8, 4);
                std::memcpy(&ch, pay + 12, 4);
                if (onCropFn) onCropFn(cx, cy, cw, ch);
                else          onCrop(cx, cy, cw, ch);
            }
            break;
        case proto::CLEAR_CROP:
            if (onClearCropFn) onClearCropFn(); else onClearCrop();
            break;
        case proto::CANCEL:
            if (onCancelFn) onCancelFn(); else onCancel();
            break;
        case proto::PAUSE:
            if (onPauseFn) onPauseFn(); else onPause();
            break;
        case proto::RESUME:
            if (onResumeFn) onResumeFn(); else onResume();
            break;
        default:
            spdlog::warn("SocketDisplayDriver: unknown command type 0x{:02x}", hdr.type);
            break;
        }

        m_recvBuf.erase(m_recvBuf.begin(),
                         m_recvBuf.begin() + static_cast<ptrdiff_t>(total));
    }
}

} // namespace anacapa
