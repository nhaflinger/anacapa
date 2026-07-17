#ifdef ANACAPA_ENABLE_MATERIALX
#include <gtest/gtest.h>
#include "shading/MaterialXCodegen.h"
#include "shading/MaterialXMaterial.h"
#include "shading/MaterialXMslAdapter.h"
#include "shading/MxGlslToCuda.h"

namespace {
std::string fixture(const char* name) {
    return std::string(ANACAPA_TEST_FIXTURES_DIR) + "/materialx/" + name;
}
}

TEST(MaterialXCodegen, GeneratesMslForProceduralBaseColor) {
    anacapa::MaterialXCodegen codegen;
    auto result = codegen.generateMsl(fixture("procedural.mtlx"), "base_color");
    ASSERT_TRUE(result.valid) << "base_color is driven by a mix() node chain — should be procedural";
    EXPECT_FALSE(result.source.empty());
    EXPECT_FALSE(result.functionName.empty());
}

TEST(MaterialXCodegen, GeneratesGlslForProceduralBaseColor) {
    anacapa::MaterialXCodegen codegen;
    auto result = codegen.generateGlsl(fixture("procedural.mtlx"), "base_color");
    ASSERT_TRUE(result.valid);
    EXPECT_NE(result.source.find("void main()"), std::string::npos);
    EXPECT_EQ(result.outputVarName, "anacapa_gen_base_color");
}

TEST(MaterialXCodegen, LiteralInputIsNotProcedural) {
    anacapa::MaterialXCodegen codegen;
    auto result = codegen.generateMsl(fixture("literal.mtlx"), "base_color");
    EXPECT_FALSE(result.valid) << "base_color is a plain literal value — should not generate";
}

TEST(MaterialXMaterial, ExposesGeneratedShaderForProceduralInput) {
    anacapa::MaterialXMaterial mat(fixture("procedural.mtlx"));
    const auto* gen = mat.generated("base_color");
    ASSERT_NE(gen, nullptr);
    EXPECT_TRUE(gen->valid);
    EXPECT_FALSE(gen->source.empty());
}

TEST(MaterialXMaterial, FallsBackToLiteralDefaultsForLiteralMaterial) {
    anacapa::MaterialXMaterial mat(fixture("literal.mtlx"));
    EXPECT_EQ(mat.generated("base_color"), nullptr);
    EXPECT_FLOAT_EQ(mat.metalness(), 1.f);
}

TEST(MaterialXMslAdapter, WrapperSignatureAlwaysTakesUv) {
    // Every wrapper shares one visible_function_table signature (Metal
    // requires this), so `uv` and `objPos` are parameters regardless of
    // whether the material actually uses <texcoord>/<position> — see
    // MaterialXMslAdapter.cpp's findTexcoordFields()/hasPositionObjectField()
    // comments for why (a real-scene regression: no procedural wood grain,
    // random black regions from garbage/uninitialized UV feeding pattern
    // nodes that DO use <texcoord>).
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateMsl(fixture("procedural.mtlx"), "base_color");
    ASSERT_TRUE(gen.valid);
    auto adapted = anacapa::buildMslAdapter(gen, "anacapa_mx_wrap_uvsig");
    ASSERT_TRUE(adapted.valid);
    EXPECT_NE(adapted.source.find("uint offset, float2 uv, float3 objPos)"), std::string::npos);
    // procedural.mtlx has no <texcoord>/<position> node, so no vd.texcoord_N
    // or vd.positionObject assignment should be emitted — regression control
    // for the next tests.
    EXPECT_EQ(adapted.source.find("vd.texcoord"), std::string::npos);
    EXPECT_EQ(adapted.source.find("vd.positionObject"), std::string::npos);
}

TEST(MaterialXMslAdapter, InitializesPositionObjectFieldFromRealObjPos) {
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateMsl(fixture("position_driven.mtlx"), "base_color");
    ASSERT_TRUE(gen.valid) << "base_color is driven by an object-space <position> — should be procedural";
    EXPECT_NE(gen.source.find("vec3 positionObject"), std::string::npos)
        << "sanity check: MaterialX's own generated VertexData should declare a positionObject field";

    auto adapted = anacapa::buildMslAdapter(gen, "anacapa_mx_wrap_postest");
    ASSERT_TRUE(adapted.valid);
    EXPECT_NE(adapted.source.find("vd.positionObject = objPos;"), std::string::npos)
        << "the real per-fragment object-space position must be wired into VertexData, not left uninitialized";
}

TEST(MaterialXMslAdapter, InitializesTexcoordFieldFromRealUv) {
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateMsl(fixture("texcoord_driven.mtlx"), "base_color");
    ASSERT_TRUE(gen.valid) << "base_color is driven by a <texcoord>-fed mix — should be procedural";
    EXPECT_NE(gen.source.find("texcoord_0"), std::string::npos)
        << "sanity check: MaterialX's own generated VertexData should declare a texcoord_0 field";

    auto adapted = anacapa::buildMslAdapter(gen, "anacapa_mx_wrap_uvtest");
    ASSERT_TRUE(adapted.valid);
    EXPECT_NE(adapted.source.find("vd.texcoord_0 = uv;"), std::string::npos)
        << "the real per-fragment UV must be wired into VertexData, not left uninitialized";
}

TEST(MaterialXMslAdapter, WrapsGeneratedShaderIntoVisibleFunction) {
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateMsl(fixture("procedural.mtlx"), "base_color");
    ASSERT_TRUE(gen.valid);
    ASSERT_FALSE(gen.outputVarName.empty());

    auto adapted = anacapa::buildMslAdapter(gen, "anacapa_mx_wrap_test0");
    ASSERT_TRUE(adapted.valid);
    EXPECT_NE(adapted.source.find("[[visible]] float4 anacapa_mx_wrap_test0"), std::string::npos);
    EXPECT_NE(adapted.source.find("GlobalContext ctx(vd"), std::string::npos);
    EXPECT_NE(adapted.source.find("return po." + gen.outputVarName + ";"), std::string::npos);
    // mix1_fg=(0,0,1), mix1_bg=(1,0,0), mix1_mix=0.5 -> 3+3+1 = 7 floats.
    // Exact count isn't the point here — it must match gen.uniforms.size()
    // component-for-component (PublicUniforms only; see MaterialXCodegen.cpp).
    size_t expected = 0;
    for (const auto& u : gen.uniforms) {
        if (u.type == "float" || u.type == "integer" || u.type == "boolean") expected += 1;
        else if (u.type == "vector2") expected += 2;
        else if (u.type == "vector3" || u.type == "color3") expected += 3;
        else if (u.type == "vector4" || u.type == "color4") expected += 4;
    }
    EXPECT_EQ(adapted.defaultUniformData.size(), expected);
}

TEST(MaterialXMslAdapter, WrapsFloatOutputCorrectly) {
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateMsl(fixture("procedural.mtlx"), "specular_roughness");
    ASSERT_TRUE(gen.valid);

    auto adapted = anacapa::buildMslAdapter(gen, "anacapa_mx_wrap_test1");
    ASSERT_TRUE(adapted.valid);
    // rough1_in=0.4, rough1_low=0.05, rough1_high=0.95 -> 3 floats.
    ASSERT_EQ(adapted.defaultUniformData.size(), 3u);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[0], 0.4f);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[1], 0.05f);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[2], 0.95f);
}

TEST(MxGlslToCuda, WrapsGeneratedShaderIntoDeviceFunction) {
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateGlsl(fixture("procedural.mtlx"), "base_color");
    ASSERT_TRUE(gen.valid);

    auto adapted = anacapa::buildCudaAdapter(gen, "anacapa_mx_cuda_wrap_test0");
    ASSERT_TRUE(adapted.valid);
    EXPECT_NE(adapted.source.find("__direct_callable__anacapa_mx_cuda_wrap_test0"), std::string::npos);
    EXPECT_NE(adapted.source.find("float2 cudaUv, float3 cudaObjPos)"), std::string::npos);
    EXPECT_NE(adapted.source.find("vec3 mix1_fg = vec3(uniformData[offset+0]"), std::string::npos);
    EXPECT_NE(adapted.source.find("return make_float4(" + gen.outputVarName), std::string::npos);
    EXPECT_EQ(adapted.source.find("#version"), std::string::npos);
    EXPECT_EQ(adapted.source.find("uniform vec3"), std::string::npos);
    // mix1_fg=(0,0,1), mix1_bg=(1,0,0), mix1_mix=0.5 -> 7 floats.
    size_t expected = 0;
    for (const auto& u : gen.uniforms) {
        if (u.type == "float" || u.type == "integer" || u.type == "boolean") expected += 1;
        else if (u.type == "vector2") expected += 2;
        else if (u.type == "vector3" || u.type == "color3") expected += 3;
        else if (u.type == "vector4" || u.type == "color4") expected += 4;
    }
    EXPECT_EQ(adapted.defaultUniformData.size(), expected);
    ASSERT_EQ(adapted.defaultUniformData.size(), 7u);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[0], 0.f);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[1], 0.f);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[2], 1.f);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[6], 0.5f);
}

TEST(MxGlslToCuda, NoMxDefineCollisionsWithPrelude) {
    // MaterialX's own generated boilerplate emits `#define mx_sin sin` etc,
    // which would silently collide with (and contradict) the CUDA-correct
    // aliases the prelude defines for the same names — regression guard for
    // that bug (see MxGlslToCuda.cpp's stripLinesStartingWith(body, "#define mx_")).
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateGlsl(fixture("procedural.mtlx"), "base_color");
    ASSERT_TRUE(gen.valid);
    auto adapted = anacapa::buildCudaAdapter(gen, "anacapa_mx_cuda_collision_test");
    ASSERT_TRUE(adapted.valid);
    EXPECT_EQ(adapted.source.find("#define mx_sin sin\n"), std::string::npos);
    EXPECT_EQ(adapted.source.find("#define mx_inversesqrt inversesqrt"), std::string::npos);
}

TEST(MxGlslToCuda, InitializesTexcoordFieldFromRealUv) {
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateGlsl(fixture("texcoord_driven.mtlx"), "base_color");
    ASSERT_TRUE(gen.valid);
    EXPECT_NE(gen.source.find("texcoord_0"), std::string::npos);

    auto adapted = anacapa::buildCudaAdapter(gen, "anacapa_mx_cuda_uvtest");
    ASSERT_TRUE(adapted.valid);
    // The GLSL interface block must be excised (not valid CUDA C++).
    EXPECT_EQ(adapted.source.find("in VertexData"), std::string::npos);
    // Wrapper takes CUDA's native float2/float3 at the ABI boundary...
    EXPECT_NE(adapted.source.find("float2 cudaUv, float3 cudaObjPos)"), std::string::npos);
    // ...converted to the prelude's own vec2 for internal use...
    EXPECT_NE(adapted.source.find("vec2 uv = vec2(cudaUv.x, cudaUv.y);"), std::string::npos);
    // ...and wired into a local vd standing in for the excised interface block.
    EXPECT_NE(adapted.source.find("vd.texcoord_0 = uv;"), std::string::npos);
}

TEST(MxGlslToCuda, InitializesPositionObjectFieldFromRealObjPos) {
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateGlsl(fixture("position_driven.mtlx"), "base_color");
    ASSERT_TRUE(gen.valid);
    EXPECT_NE(gen.source.find("vec3 positionObject"), std::string::npos);

    auto adapted = anacapa::buildCudaAdapter(gen, "anacapa_mx_cuda_postest");
    ASSERT_TRUE(adapted.valid);
    EXPECT_EQ(adapted.source.find("in VertexData"), std::string::npos);
    EXPECT_NE(adapted.source.find("vec3 objPos = vec3(cudaObjPos.x, cudaObjPos.y, cudaObjPos.z);"), std::string::npos);
    EXPECT_NE(adapted.source.find("vd.positionObject = objPos;"), std::string::npos);
}

TEST(MxGlslToCuda, WrapsFloatOutputCorrectly) {
    anacapa::MaterialXCodegen codegen;
    auto gen = codegen.generateGlsl(fixture("procedural.mtlx"), "specular_roughness");
    ASSERT_TRUE(gen.valid);

    auto adapted = anacapa::buildCudaAdapter(gen, "anacapa_mx_cuda_wrap_test1");
    ASSERT_TRUE(adapted.valid);
    ASSERT_EQ(adapted.defaultUniformData.size(), 3u);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[0], 0.4f);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[1], 0.05f);
    EXPECT_FLOAT_EQ(adapted.defaultUniformData[2], 0.95f);
}
#endif
