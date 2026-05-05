# Anacapa Linux Dev Machine Setup

Reference for building Anacapa on Linux (x86_64). Versions here match the
macOS development machine exactly to avoid cross-platform surprises.

---

## Toolchain

| Tool | Mac version | Linux requirement |
|------|-------------|-------------------|
| CMake | 3.30.3 | ≥ 3.22 (project minimum); install 3.30+ to match |
| C++ compiler | Apple Clang 17 (C++20) | GCC 13+ or Clang 17+ with full C++20 support |
| Python | 3.10.18 (Blender addon) | 3.10+ |
| CUDA (GPU backend) | N/A (Metal on Mac) | 12.x (see below) |

```bash
# Ubuntu 24.04
sudo apt install build-essential cmake git python3 python3-pip
# For Clang 17 specifically:
sudo apt install clang-17 lld-17
```

---

## Auto-fetched by CMake (FetchContent — no manual install needed)

These are identical on Mac and Linux. CMake downloads them at configure time.

| Library | Version | Purpose |
|---------|---------|---------|
| spdlog | **v1.14.1** | Structured logging |
| nlohmann/json | **v3.11.3** | JSON parsing (matassign, USD sidecars) |
| CLI11 | **v2.4.2** | Command-line argument parsing |
| GoogleTest | **v1.15.2** | Unit tests |
| OSL headers | **v1.14.7.0** | Open Shading Language headers (runtime libs separate — see below) |
| SDL2 | **release-2.30.3** | Viewer window/input (ANACAPA_ENABLE_VIEWER only) |
| glad | **v0.1.36** | OpenGL 3.3 core loader (viewer only) |
| Dear ImGui | **v1.91.1** | Viewer UI (viewer only) |
| stb | master | Image loader for viewer |

---

## System Dependencies (must install manually)

### OpenImageIO — 3.1.11.0
EXR output, texture loading. **This is the most common version mismatch.**

```bash
# Ubuntu 24.04 ships an older version — build from source to match exactly:
sudo apt install libilmbase-dev libopenexr-dev libboost-all-dev \
                 libtiff-dev libpng-dev libjpeg-dev libraw-dev
git clone https://github.com/AcademySoftwareFoundation/OpenImageIO.git
cd OpenImageIO && git checkout v3.1.11.0
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build --parallel && sudo cmake --install build
```

> **Note:** Mac uses `brew install openimageio` which currently gives 3.1.11.0.
> The apt package on Ubuntu 24.04 is 2.4.x — too old, build from source.

---

### OpenUSD — 25.02  (`PXR_VERSION 2502`)
Built from source using USD's own build script. Install to `~/usd` to match
the CMake preset (`USD_ROOT=$HOME/usd`).

```bash
sudo apt install libglew-dev libxt-dev libxrandr-dev nasm

git clone https://github.com/PixarAnimation/OpenUSD.git
cd OpenUSD && git checkout v25.02
python3 build_scripts/build_usd.py ~/usd
```

Build takes ~30 min. The CMake preset passes `USD_ROOT=$HOME/usd`.

---

### Open Shading Language — 1.14.7

**Runtime libs (`liboslexec.so`, `liboslcomp.so`) come from Blender 5.1.**
The headers are auto-fetched by CMake, so you only need the runtime.

Blender installs its shared libs at:
```
~/.config/blender/5.1/  # (not quite — see below)
# Actual location inside Blender's installation:
/path/to/blender-5.1/lib/liboslexec.so
/path/to/blender-5.1/lib/liboslcomp.so
```

Point the build at them:
```bash
cmake -DOSL_LIB_DIR=/path/to/blender-5.1/lib ...
```

Alternatively, build OSL 1.14.7 from source (heavier, but fully self-contained):
```bash
git clone https://github.com/AcademySoftwareFoundation/OpenShadingLanguage.git
cd OpenShadingLanguage && git checkout v1.14.7.0
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/osl \
      -DOSL_BUILD_TESTS=OFF
cmake --build build --parallel && cmake --install build
# Then: -DOSL_LIB_DIR=$HOME/osl/lib
```

---

### Alembic — 1.8.11

```bash
# Ubuntu may have this in apt:
sudo apt install libalembic-dev  # check version: dpkg -s libalembic-dev | grep Version

# If not 1.8.11, build from source:
git clone https://github.com/alembic/alembic.git
cd alembic && git checkout 1.8.11
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DUSE_HDF5=OFF -DALEMBIC_ILMBASE_LINK_STATIC=OFF
cmake --build build --parallel && sudo cmake --install build
```

---

### Intel OpenImageDenoise — 2.3.0
The Linux CMake preset uses **2.3.0** (installed to `~/oidn`), not 2.4.1 (Mac).

```bash
# Download pre-built binary from:
# https://github.com/OpenImageDenoise/oidn/releases/tag/v2.3.0
# Choose: oidn-2.3.0.x86_64.linux.tar.gz
tar xf oidn-2.3.0.x86_64.linux.tar.gz
mv oidn-2.3.0.x86_64.linux ~/oidn
```

The CMake preset passes:
```
-DOpenImageDenoise_DIR=$HOME/oidn/lib/cmake/OpenImageDenoise-2.3.0
```

---

### NVIDIA CUDA Toolkit (GPU backend only)

The Linux CMake preset targets **SM 86** (RTX 3080/3090/A6000).  
Adjust `CUDA_ARCH` for your specific card:

| Card | SM arch |
|------|---------|
| RTX 30xx | 86 |
| RTX 40xx | 89 |
| RTX 20xx / Titan V | 75 |
| A100 | 80 |

```bash
# Install CUDA 12.x from NVIDIA's repo (not apt's older version):
# https://developer.nvidia.com/cuda-downloads
# Choose: Linux → x86_64 → Ubuntu → 24.04 → deb (network)

sudo apt install cuda-toolkit-12-x
```

CUDA standard used: **C++17** (`CMAKE_CUDA_STANDARD 17`).
Runtime library: shared (`CMAKE_CUDA_RUNTIME_LIBRARY Shared`).

---

### Blender — 5.1.1
Required for the Blender addon and as the source of OSL runtime libraries.

Download from blender.org or Steam. On Linux without a GUI:
```bash
# Headless install for OSL libs only (no Steam needed):
wget https://download.blender.org/release/Blender5.1/blender-5.1.1-linux-x64.tar.xz
tar xf blender-5.1.1-linux-x64.tar.xz -C ~/
# OSL libs will be at: ~/blender-5.1.1-linux-x64/lib/liboslexec.so
```

---

### PySide6 — 6.11.0
Required for the matassign editor (upcoming tool, not yet built).

```bash
pip3 install PySide6==6.11.0
```

---

## Build Presets (Linux)

```bash
# Configure — basic (no GPU):
cmake --preset linux-x86_64

# Configure — with CUDA GPU backend:
cmake --preset linux-x86_64-cuda

# Configure — with CUDA + viewer:
cmake --preset linux-x86_64-cuda-viewer

# Build:
cmake --build build/Linux --parallel
```

The preset automatically sets:
- `ANACAPA_ENABLE_USD=ON`
- `ANACAPA_ENABLE_CUDA=ON`
- `CUDA_ARCH=86`
- `USD_ROOT=$HOME/usd`
- `OpenImageDenoise_DIR=$HOME/oidn/lib/cmake/OpenImageDenoise-2.3.0`

---

## Version Summary (quick reference)

| Dependency | Mac | Linux |
|-----------|-----|-------|
| CMake | 3.30.3 | 3.30.3 |
| OpenImageIO | 3.1.11.0 | 3.1.11.0 (build from source) |
| OpenUSD | 25.02 | 25.02 (build from source) |
| OSL runtime | 1.14.7 (from Blender) | 1.14.7 (from Blender or source) |
| Alembic | 1.8.11 | 1.8.11 |
| OpenImageDenoise | 2.4.1 | **2.3.0** (pre-built binary) |
| Blender | 5.1.1 | 5.1.1 |
| PySide6 | 6.11.0 | 6.11.0 |
| spdlog | v1.14.1 | v1.14.1 (FetchContent) |
| nlohmann/json | v3.11.3 | v3.11.3 (FetchContent) |
| CLI11 | v2.4.2 | v2.4.2 (FetchContent) |
| GoogleTest | v1.15.2 | v1.15.2 (FetchContent) |
| CUDA | N/A | 12.x, SM 86 |

> **OSL shader compilation note:** The `anacapa_marschner_hair.osl` shader
> uses `closure color marschner_hair(...) BUILTIN` syntax. If you're hitting
> compile errors on Linux, verify that your OSL runtime is exactly **1.14.7**
> — the BUILTIN closure declaration syntax and the `OSL_LIBRARY_VERSION_MINOR`
> guard in `OslMaterial.cpp` (line ~60) both depend on this version.
