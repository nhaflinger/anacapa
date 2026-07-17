#include "MaterialXCodegen.h"

#ifdef ANACAPA_ENABLE_MATERIALX

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Definition.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>

#include <spdlog/spdlog.h>

namespace mx = MaterialX;

namespace anacapa {

struct MaterialXCodegen::Impl {
    mx::DocumentPtr libraryDoc;
    bool libraryLoaded = false;

    void ensureLibraryLoaded() {
        if (libraryLoaded) return;
        libraryDoc = mx::createDocument();
        try {
#ifdef ANACAPA_MATERIALX_LIBRARIES_DIR
            mx::FilePath librariesDir(ANACAPA_MATERIALX_LIBRARIES_DIR);
            mx::FileSearchPath searchPath;
            searchPath.append(librariesDir.getParentPath());
            mx::loadLibraries({"libraries"}, searchPath, libraryDoc);
            m_searchPath = searchPath;
#else
#error "ANACAPA_MATERIALX_LIBRARIES_DIR not defined — check CMakeLists.txt MaterialX linking block"
#endif
            libraryLoaded = true;
        } catch (const std::exception& e) {
            spdlog::error("MaterialXCodegen: failed to load MaterialX standard library: {}", e.what());
        }
    }

    mx::FileSearchPath m_searchPath;
};

MaterialXCodegen::MaterialXCodegen() : m_impl(std::make_unique<Impl>()) {
    m_impl->ensureLibraryLoaded();
}

MaterialXCodegen::~MaterialXCodegen() = default;

// Find any node of category "open_pbr_surface" or "standard_surface" (the
// two surface-shader terminals the Blender addon can emit) anywhere in the
// document — mirrors USDLoader.cpp's findOpenPBRShader(), same idea for the
// MaterialX-native document instead of a USD stage.
static mx::NodePtr findSurfaceShaderNode(mx::DocumentPtr doc) {
    for (mx::ElementPtr elem : doc->traverseTree()) {
        mx::NodePtr node = elem->asA<mx::Node>();
        if (!node) continue;
        const std::string& cat = node->getCategory();
        if (cat == "open_pbr_surface" || cat == "standard_surface") {
            return node;
        }
    }
    return nullptr;
}

// Resolve what `inputName` on the surface shader node is actually driven by:
// a literal value (nothing to generate), a direct node connection, or a
// nodegraph output connection. Returns the ElementPtr to pass to
// ShaderGenerator::generate(), or nullptr if the input is a plain literal.
static mx::ElementPtr resolveGenTarget(mx::NodePtr surfaceNode, const std::string& inputName) {
    mx::InputPtr input = surfaceNode->getInput(inputName);
    if (!input) return nullptr;

    // getConnectedOutput() resolves a direct nodename="..." connection (no
    // "nodegraph" attribute) by returning the target node's own <output>
    // child element, if it declares one (Blender's exporter emits these on
    // every node). That Output's parent is the Node itself, which
    // ShaderGraph::create() does not accept as an interface (only a
    // NodeGraph or Document parent) — it throws "no interface valid for
    // shader generation". Only trust getConnectedOutput() when it actually
    // resolved to a nodegraph-scoped output; otherwise fall through to the
    // node-connection path below, which synthesizes a proper NodeGraph
    // output for the connected node.
    if (mx::OutputPtr connectedOutput = input->getConnectedOutput()) {
        if (connectedOutput->getParent()->isA<mx::NodeGraph>() ||
            connectedOutput->getParent()->isA<mx::Document>()) {
            return connectedOutput;
        }
    }

    mx::NodePtr connectedNode = input->getConnectedNode();
    if (!connectedNode) {
        return nullptr;  // literal value — nothing procedural to generate
    }

    // Direct node connection (no intermediate nodegraph Output element) —
    // synthesize a temporary Output in the same graph scope pointing at it,
    // matching the pattern MaterialX's own tests use for single-output
    // generation (ShaderGenerator::generate() expects an Output/shaderref,
    // not a bare Node).
    mx::GraphElementPtr parentGraph = connectedNode->getParent()->asA<mx::GraphElement>();
    if (!parentGraph) return nullptr;

    const std::string tempName = "anacapa_gen_" + inputName;
    mx::OutputPtr tempOut = parentGraph->getOutput(tempName);
    if (!tempOut) {
        tempOut = parentGraph->addOutput(tempName, input->getType());
        tempOut->setConnectedNode(connectedNode);
    }
    return tempOut;
}

static MxGeneratedShader generateForTarget(MaterialXCodegen::Impl* impl,
                                            mx::ShaderGeneratorPtr generator,
                                            const std::string& mtlxPath,
                                            const std::string& inputName) {
    MxGeneratedShader result;
    if (!impl->libraryLoaded) {
        spdlog::error("MaterialXCodegen: library not loaded, cannot generate '{}'", mtlxPath);
        return result;
    }

    mx::DocumentPtr doc = mx::createDocument();
    doc->importLibrary(impl->libraryDoc);
    try {
        mx::readFromXmlFile(doc, mtlxPath);
    } catch (const std::exception& e) {
        spdlog::error("MaterialXCodegen: failed to read '{}': {}", mtlxPath, e.what());
        return result;
    }

    mx::NodePtr surfaceNode = findSurfaceShaderNode(doc);
    if (!surfaceNode) {
        spdlog::warn("MaterialXCodegen: no open_pbr_surface/standard_surface node in '{}'", mtlxPath);
        return result;
    }

    mx::ElementPtr genTarget = resolveGenTarget(surfaceNode, inputName);
    if (!genTarget) {
        // Not an error — the input is a literal or plain texture, which the
        // existing (non-MaterialX) GPU material path already handles.
        return result;
    }

    mx::GenContext context(generator);
    context.registerSourceCodeSearchPath(impl->m_searchPath);

    mx::ShaderPtr shader;
    try {
        shader = generator->generate("anacapa_mx_" + inputName, genTarget, context);
    } catch (const std::exception& e) {
        spdlog::error("MaterialXCodegen: generation failed for '{}' input '{}': {}",
                      mtlxPath, inputName, e.what());
        return result;
    }
    if (!shader) {
        spdlog::error("MaterialXCodegen: generator returned null for '{}' input '{}'", mtlxPath, inputName);
        return result;
    }

    const std::string& src = shader->getSourceCode(mx::Stage::PIXEL);
    if (src.empty()) {
        spdlog::error("MaterialXCodegen: empty source generated for '{}' input '{}'", mtlxPath, inputName);
        return result;
    }

    result.valid         = true;
    result.functionName  = "anacapa_mx_" + inputName;
    result.source        = src;
    result.outputVarName = genTarget->getName();

    // Only the "PublicUniforms" block corresponds 1:1 to GlobalContext's
    // constructor parameters (confirmed by inspecting generated MSL: the
    // struct's field list and the constructor's parameter list match
    // exactly). Other uniform blocks (e.g. per-nodedef enum/"type"
    // selectors baked in at generation time) are declared elsewhere and
    // are NOT constructor arguments — including them here would make
    // MaterialXMslAdapter's wrapper pass more arguments than the
    // constructor accepts.
    mx::ShaderStage& pixelStage = shader->getStage(mx::Stage::PIXEL);
    for (const auto& blockEntry : pixelStage.getUniformBlocks()) {
        const mx::VariableBlock& block = *blockEntry.second;
        if (block.getName() != "PublicUniforms") continue;
        for (mx::ShaderPort* port : block.getVariableOrder()) {
            MxUniform u;
            u.name  = port->getVariable();
            u.type  = port->getType().getName();
            u.value = port->getValue() ? port->getValue()->getValueString() : "";
            result.uniforms.push_back(std::move(u));
        }
    }

    return result;
}

MxGeneratedShader MaterialXCodegen::generateMsl(const std::string& mtlxPath, const std::string& inputName) {
    return generateForTarget(m_impl.get(), mx::MslShaderGenerator::create(), mtlxPath, inputName);
}

MxGeneratedShader MaterialXCodegen::generateGlsl(const std::string& mtlxPath, const std::string& inputName) {
    return generateForTarget(m_impl.get(), mx::GlslShaderGenerator::create(), mtlxPath, inputName);
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_MATERIALX
