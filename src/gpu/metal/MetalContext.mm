#ifdef ANACAPA_ENABLE_METAL

#include "MetalContext.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <spdlog/spdlog.h>

namespace anacapa {

// ---------------------------------------------------------------------------
// PIMPL — keeps all Objective-C types out of the header
// ---------------------------------------------------------------------------
struct MetalContext::Impl {
    id<MTLDevice>       device       = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary>      library      = nil;
    std::string         deviceName;
};

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------
bool MetalContext::isAvailable() {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d != nil;
}

std::unique_ptr<MetalContext> MetalContext::create(const std::string& metallibPath) {
    std::unique_ptr<MetalContext> ctx(new MetalContext());

    // Device
    ctx->m_impl->device = MTLCreateSystemDefaultDevice();
    if (!ctx->m_impl->device) {
        spdlog::error("MetalContext: no Metal device available");
        return nullptr;
    }
    ctx->m_impl->deviceName = [ctx->m_impl->device.name UTF8String];
    spdlog::info("MetalContext: using device '{}'", ctx->m_impl->deviceName);

    // Command queue
    ctx->m_impl->commandQueue = [ctx->m_impl->device newCommandQueue];
    if (!ctx->m_impl->commandQueue) {
        spdlog::error("MetalContext: failed to create command queue");
        return nullptr;
    }

    // Shader library — compiled at build time by xcrun metal
    NSString* path = [NSString stringWithUTF8String:metallibPath.c_str()];
    NSError*  err  = nil;
    ctx->m_impl->library = [ctx->m_impl->device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                                            error:&err];
    if (!ctx->m_impl->library) {
        spdlog::error("MetalContext: could not load metallib '{}': {}",
                      metallibPath,
                      err ? [[err localizedDescription] UTF8String] : "unknown error");
        return nullptr;
    }

    spdlog::info("MetalContext: shader library loaded from '{}'", metallibPath);
    return ctx;
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
MetalContext::MetalContext() : m_impl(std::make_unique<Impl>()) {}

MetalContext::~MetalContext() {
    // ARC handles release; just log
    spdlog::debug("MetalContext: destroyed");
}

bool MetalContext::isValid() const {
    return m_impl->device != nil
        && m_impl->commandQueue != nil
        && m_impl->library != nil;
}

std::string MetalContext::name() const { return m_impl->deviceName; }

void* MetalContext::device()       const { return (__bridge void*)m_impl->device; }
void* MetalContext::commandQueue() const { return (__bridge void*)m_impl->commandQueue; }
void* MetalContext::library()      const { return (__bridge void*)m_impl->library; }

} // namespace anacapa

#endif // ANACAPA_ENABLE_METAL
