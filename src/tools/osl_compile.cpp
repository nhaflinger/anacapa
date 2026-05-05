// osl_compile — compile an OSL shader source (.osl) to bytecode (.oso).
//
// Usage:
//   osl_compile <input.osl> <output.oso> [include_dir]
//
// Wraps anacapa's oslCompileShader() so it can be invoked as a CMake
// custom command during the build.  The optional include_dir is passed
// as the matDir argument to oslCompileShader() and controls which
// directory is put on the -I path (alongside the build-time _mx_stdlib.h
// directory set by ANACAPA_MX_STDLIB_DIR).  Defaults to the directory
// containing the input .osl file.

#ifdef ANACAPA_ENABLE_OSL

#include <anacapa/shading/OslMaterial.h>
#include <cstdio>
#include <string>

// Derive the directory part of a file path.
static std::string dirOf(const std::string& p) {
    auto slash = p.rfind('/');
    if (slash == std::string::npos) slash = p.rfind('\\');
    return (slash != std::string::npos) ? p.substr(0, slash) : ".";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: osl_compile <input.osl> <output.oso> [include_dir]\n");
        return 1;
    }
    const std::string oslPath = argv[1];
    const std::string osoPath = argv[2];
    const std::string matDir  = (argc > 3) ? argv[3] : dirOf(oslPath);

    bool ok = anacapa::oslCompileShader(oslPath, osoPath, matDir);
    if (!ok) {
        fprintf(stderr, "osl_compile: compilation failed: %s\n", argv[1]);
        return 1;
    }
    printf("osl_compile: %s → %s\n", argv[1], argv[2]);
    return 0;
}

#else  // !ANACAPA_ENABLE_OSL

int main() {
    fprintf(stderr,
        "osl_compile: built without OSL support (ANACAPA_ENABLE_OSL=OFF)\n");
    return 1;
}

#endif
