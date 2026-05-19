// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
// NOTE: This file was generated from oslversion.h.in for OSL 1.14.7

#ifndef OSLVERSION_H
#define OSLVERSION_H

#define OSL_VERSION_MAJOR 1
#define OSL_VERSION_MINOR 14
#define OSL_VERSION_PATCH 7
#define OSL_VERSION_TWEAK 0
#define OSL_VERSION (10000 * OSL_VERSION_MAJOR + \
                       100 * OSL_VERSION_MINOR + \
                             OSL_VERSION_PATCH)

#define OSL_LIBRARY_VERSION_MAJOR 1
#define OSL_LIBRARY_VERSION_MINOR 14
#define OSL_LIBRARY_VERSION_PATCH 7
#define OSL_LIBRARY_VERSION_TWEAK 0
#define OSL_LIBRARY_VERSION_RELEASE_TYPE

#define OSL_LIBRARY_VERSION_CODE (10000 * OSL_VERSION_MAJOR + \
                                    100 * OSL_VERSION_MINOR + \
                                          OSL_VERSION_PATCH)

#define OSL_MAKE_VERSION_STRING2(a,b,c,d,e) #a "." #b "." #c "." #d #e
#define OSL_MAKE_VERSION_STRING(a,b,c,d,e) OSL_MAKE_VERSION_STRING2(a,b,c,d,e)
#define OSL_LIBRARY_VERSION_STRING \
    OSL_MAKE_VERSION_STRING(OSL_LIBRARY_VERSION_MAJOR, \
                            OSL_LIBRARY_VERSION_MINOR, \
                            OSL_LIBRARY_VERSION_PATCH, \
                            OSL_LIBRARY_VERSION_TWEAK, \
                            OSL_LIBRARY_VERSION_RELEASE_TYPE)
#define OSL_INTRO_STRING "Open Shading Language " OSL_LIBRARY_VERSION_STRING
#define OSL_COPYRIGHT_STRING "Copyright Contributors to the Open Shading Language project."

#define OSO_FILE_VERSION_MAJOR 1
#define OSO_FILE_VERSION_MINOR 0

#define OSL_SUPPORTS_WEIGHTED_CLOSURE_COMPONENTS 1
#define OSL_SHADERGLOBALS_HAS_RENDERER_PTR 1

namespace OSL {
    inline namespace v1_14 { }
}
#define OSL_CUSTOM_OUTER_NAMESPACE 0

#define OSL_NAMESPACE_BEGIN namespace OSL { inline namespace v1_14 {
#define OSL_NAMESPACE_END } }
#define OSL_CURRENT_NAMESPACE OSL::v1_14

#define OSL_NS_BEGIN(ver) namespace OSL { namespace ver {
#define OSL_NS_END } }

// Legacy definitions, DEPRECATED(1.14)
#define OSL_NAMESPACE_ENTER OSL_NAMESPACE_BEGIN
#define OSL_NAMESPACE_EXIT OSL_NAMESPACE_END

#define OSL_BUILD_CPP 17
#define OSL_BUILD_CPP11 (17 >= 11)
#define OSL_BUILD_CPP14 (17 >= 14)
#define OSL_BUILD_CPP17 (17 >= 17)
#define OSL_BUILD_CPP20 (17 >= 20)
#define OSL_BUILD_CPP23 (17 >= 20)

#define OSL_SHADER_INSTALL_DIR ""

#define __OSL_CONCAT_INDIRECT(A, B) A ## B
#define __OSL_CONCAT(A, B) __OSL_CONCAT_INDIRECT(A,B)
#define __OSL_CONCAT3(A, B, C) __OSL_CONCAT(__OSL_CONCAT(A,B),C)
#define __OSL_CONCAT4(A, B, C, D) __OSL_CONCAT(__OSL_CONCAT3(A,B,C),D)
#define __OSL_CONCAT5(A, B, C, D, E) __OSL_CONCAT(__OSL_CONCAT4(A,B,C,D),E)
#define __OSL_CONCAT6(A, B, C, D, E, F) __OSL_CONCAT(__OSL_CONCAT5(A,B,C,D,E),F)
#define __OSL_CONCAT7(A, B, C, D, E, F, G) __OSL_CONCAT(__OSL_CONCAT6(A,B,C,D,E,F),G)
#define __OSL_CONCAT8(A, B, C, D, E, F, G, H) __OSL_CONCAT(__OSL_CONCAT7(A,B,C,D,E,F,G),H)
#define __OSL_CONCAT9(A, B, C, D, E, F, G, H, I) __OSL_CONCAT(__OSL_CONCAT8(A,B,C,D,E,F,G,H),I)
#define __OSL_CONCAT10(A, B, C, D, E, F, G, H, I, J) __OSL_CONCAT(__OSL_CONCAT9(A,B,C,D,E,F,G,H,I),J)

#if defined(__OSL_TARGET_ISA) && defined(__OSL_WIDTH)
    #define __OSL_WIDE_PVT __OSL_CONCAT5(b,__OSL_WIDTH,_,__OSL_TARGET_ISA,_pvt)
#endif

#define __OSL_INDIRECT_STRINGIFY(x) #x
#define __OSL_STRINGIFY(x) __OSL_INDIRECT_STRINGIFY(x)

#endif /* OSLVERSION_H */
