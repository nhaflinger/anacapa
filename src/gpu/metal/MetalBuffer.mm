#ifdef ANACAPA_ENABLE_METAL

#import <Metal/Metal.h>
#include "MetalBuffer.h"

namespace anacapa {

void* metalBufferCreate(void* device, size_t byteSize) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> buf = [dev newBufferWithLength:byteSize
                                        options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buf;
}

void metalBufferRelease(void* buffer) {
    // Release the retained reference acquired in metalBufferCreate
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buffer;
    (void)buf;  // ARC releases here
}

void* metalBufferContents(void* buffer) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    return [buf contents];
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_METAL
