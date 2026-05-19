// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// NOTE: This file was generated from oslconfig.h.in for OSL 1.14.7

#pragma once

#define OSL_USE_OPTIX 0
#define OSL_BUILD_BATCHED 0
#define OSL_USE_BATCHED OSL_BUILD_BATCHED

#include <OSL/export.h>
#include <OSL/oslversion.h>
#include <OSL/platform.h>

// Imath 3.x
#define OSL_USING_IMATH 3
#include <Imath/ImathVec.h>
#include <Imath/ImathMatrix.h>
#include <Imath/ImathColor.h>

#define OIIO_IMATH_H_INCLUDED 1

#ifdef __CUDA_ARCH__
#    ifndef FMT_USE_INT128
#        define FMT_USE_INT128 0
#    endif
#endif

// All the things we need from OpenImageIO
#include <OpenImageIO/oiioversion.h>
#include <OpenImageIO/errorhandler.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/ustring.h>
#include <OpenImageIO/platform.h>
#include <OpenImageIO/span.h>

OSL_NAMESPACE_BEGIN

typedef float Float;

using Vec2     = Imath::Vec2<Float>;
using Vec3     = Imath::Vec3<Float>;
using Color3   = Imath::Color3<Float>;
using Matrix22 = Imath::Matrix22<Float>;
using Matrix33 = Imath::Matrix33<Float>;
using Matrix44 = Imath::Matrix44<Float>;

using OIIO::TextureSystem;
using OIIO::TextureOpt;

using OIIO::ErrorHandler;
using OIIO::ustring;
using OIIO::ustringhash;
using OIIO::string_view;
using OIIO::span;
using OIIO::cspan;

using OIIO::TypeDesc;
using OIIO::TypeUnknown;
using OIIO::TypeFloat;
using OIIO::TypeColor;
using OIIO::TypePoint;
using OIIO::TypeVector;
using OIIO::TypeNormal;
using OIIO::TypeMatrix;
using OIIO::TypeFloat4;
using OIIO::TypeString;
using OIIO::TypeInt;
using OIIO::TypeFloat2;
using OIIO::TypeVector2;
using OIIO::TypeVector4;
using OIIO::TypeUInt64;

using OIIO::Strutil::print;

template<typename Str, typename... Args>
OSL_NODISCARD inline std::string
fmtformat(const Str& fmt, Args&&... args)
{
#if OSL_CPLUSPLUS_VERSION >= 20 || FMT_VERSION >= 100000
    return ::fmt::vformat(fmt, ::fmt::make_format_args(args...));
#else
    return OIIO::Strutil::fmt::format(fmt, std::forward<Args>(args)...);
#endif
}

template<typename OutIt, typename... Args>
OSL_NODISCARD inline auto
fmtformat_to_n(OutIt& out, size_t n, string_view fmt, Args&&... args)
{
#if OSL_CPLUSPLUS_VERSION >= 20 || FMT_VERSION >= 100000
    std::string str = fmtformat(fmt, std::forward<Args>(args)...);
    return ::fmt::format_to_n(out, n, "{}", str);
#else
    return ::fmt::format_to_n(out, n, ::fmt::string_view{fmt.begin(), fmt.length()}, std::forward<Args>(args)...);
#endif
}

using ustringhash_pod = size_t;
using ustring_pod = const char*;

inline ustring
ustring_from(ustringhash_pod h)
{
    return ustring::from_hash(h);
}

inline ustring
ustring_from(ustringhash h)
{
    return ustring::from_hash(h.hash());
}

inline ustring
ustring_from(ustring u)
{
    return u;
}

OSL_HOSTDEVICE inline ustringhash
ustringhash_from(ustringhash u)
{
    return u;
}

OSL_HOSTDEVICE inline ustringhash
ustringhash_from(ustringhash_pod u)
{
#if OIIO_VERSION_GREATER_EQUAL(2, 4, 10)
    return ustringhash{u};
#else
    return OSL::bitcast<ustringhash>(u);
#endif
}

inline ustringhash
ustringhash_from(ustring u)
{
    ustringhash ret;
    if(!u.empty())
        ret = u.uhash();
    return ret;
}

using TypeDesc_pod = int64_t;
static_assert(sizeof(TypeDesc_pod) == sizeof(TypeDesc),
              "TypeDesc size differs from its POD counterpart");

OSL_HOSTDEVICE inline TypeDesc
TypeDesc_from(TypeDesc_pod type)
{
    return OSL::bitcast<OSL::TypeDesc>(type);
}

struct TraceOpt {
    float mindist;
    float maxdist;
    bool shade;
    ustringhash traceset;
    OSL_HOSTDEVICE TraceOpt() : mindist(0.0f), maxdist(1.0e30), shade(false) {}
    enum class LLVMMemberIndex { mindist = 0, maxdist, shade, traceset, count };
};

enum class SymArena {
    Unknown = 0,
    Absolute,
    Heap,
    Outputs,
    UserData,
    Interactive,
};

#define __OSL_EXPAND_PARAMETER_PACKS(EXPRESSION) (void((EXPRESSION)), ...);

namespace pvt {

template<int... IntegerListT>
using int_sequence = std::integer_sequence<int, IntegerListT...>;

template<int EndBeforeT>
using make_int_sequence = std::make_integer_sequence<int, EndBeforeT>;

template<bool... BoolListT>
using bool_sequence = std::integer_sequence<bool, BoolListT...>;

template<class... ListT> using conjunction = std::conjunction<ListT...>;

template<bool TestT, typename TypeT = std::true_type>
using enable_if_type = typename std::enable_if<TestT, TypeT>::type;

} // namespace pvt

template <template<int> class ConstantWrapperT, int... IntListT, typename FunctorT>
static OSL_FORCEINLINE OSL_HOSTDEVICE void static_foreach(pvt::int_sequence<IntListT...>, const FunctorT &iFunctor) {
     __OSL_EXPAND_PARAMETER_PACKS( iFunctor(ConstantWrapperT<IntListT>{}) );
}

template <template<int> class ConstantWrapperT, int N, typename FunctorT>
static OSL_FORCEINLINE OSL_HOSTDEVICE void static_foreach(const FunctorT &iFunctor) {
    static_foreach<ConstantWrapperT>(pvt::make_int_sequence<N>(), iFunctor);
}

template<int N>
using ConstIndex = std::integral_constant<int, N>;

#ifdef OSL_DEV
    #define OSL_DEV_ONLY(...) __VA_ARGS__
#else
    #define OSL_DEV_ONLY(...)
#endif

OSL_NAMESPACE_END

namespace std {
#ifndef OIIO_USTRING_HAS_STDHASH
template<> struct hash<OSL::ustring> {
    std::size_t operator()(OSL::ustring u) const noexcept { return u.hash(); }
};

template<> struct hash<OSL::ustringhash> {
    OSL_HOSTDEVICE constexpr std::size_t
    operator()(OSL::ustringhash u) const noexcept { return u.hash(); }
};
#endif

template<> struct less<OSL::ustringhash> {
    OSL_HOSTDEVICE constexpr bool operator()(OSL::ustringhash u,
                                             OSL::ustringhash v) const noexcept
    {
        return u.hash() < v.hash();
    }
};
}  // namespace std

#ifndef OIIO_HAS_USTRINGHASH_FORMATTER
FMT_BEGIN_NAMESPACE
template<> struct formatter<OIIO::ustringhash> : formatter<fmt::string_view, char> {
    template<typename FormatContext>
    auto format(const OIIO::ustringhash& h, FormatContext& ctx)
        -> decltype(ctx.out()) const
    {
        OIIO::ustring u(h);
        return formatter<fmt::string_view, char>::format({ u.data(), u.size() }, ctx);
    }
};
FMT_END_NAMESPACE
#endif
