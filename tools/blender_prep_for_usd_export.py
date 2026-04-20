"""
blender_prep_for_usd_export.py
──────────────────────────────
Prepares a Blender scene for clean USD export to Anacapa, then exports USD.

Usage:
    blender my_scene.blend --background --python tools/blender_prep_for_usd_export.py -- output.usda

The output path (after the -- separator) is required. Use .usda for ASCII
or .usdc for binary. The original .blend file is never modified.

What this script handles
────────────────────────
  ✓ Realize collection instances (linked duplicates / Alt+D collections)
  ✓ Realize particle system instances (Object / Collection render type) — converts
    each instanced particle to a standalone mesh object and removes the emitter
  ✓ Convert non-mesh types to mesh (curves, text, NURBS, metaballs)
  ✓ Apply all mesh modifiers in stack order (boolean, subdivision, mirror,
    array, solidify, bevel, screw, weld, decimate, etc.)
  ✓ Apply object scale (and optionally rotation / location)
  ✓ Remove render-hidden objects that serve only as boolean cutters
  ✓ Bake unsupported shader nodes into textures so they survive USD export:
      - Invert Color   → pixel-level inversion written to a new texture file
      - Hue/Saturation → HSV adjustment baked into texture
      - Bright/Contrast → baked into texture
      - RGB Curves, Color Ramp → Blender bake (requires UV + mesh)
  ✓ Convert Glass BSDF materials to Principled BSDF with opacity=0 so the
    USD exporter produces a valid UsdPreviewSurface (Glass BSDF has no direct
    USD equivalent and exports as an empty material shell otherwise)
  ✓ Report objects that could not be processed with an explanation

What requires manual attention (printed as warnings)
─────────────────────────────────────────────────────
  ✗ Particle hair (strand rendering) — complex; export separately
  ✗ Volume / VDB objects — fundamentally different USD prim type
  ✗ Grease Pencil objects — no mesh equivalent
  ✗ Library-linked objects that cannot be made local
  ✗ Shape keys — modifier application removes shape keys; user must decide
    whether to apply at current key mix or remove them first
  ✗ Armature / skeleton deformation — pose is applied but rig is left intact;
    verify the resulting mesh looks correct before export
  ✗ Complex procedural node graphs with no image texture source — these
    require a full Cycles bake and are not handled automatically
"""

import bpy
import os
import traceback
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Configuration — flip these if you want different behavior
# ---------------------------------------------------------------------------

# Apply object rotation and location in addition to scale.
# Scale should almost always be applied for correct USD normals.
# Rotation and location are safe to leave at False for most scenes.
APPLY_ROTATION = False
APPLY_LOCATION = False

# Remove objects that are hidden from the render AND have no children.
# These are usually boolean cutter helpers that shouldn't appear in USD.
REMOVE_RENDER_HIDDEN = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')

def select_only(obj):
    deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

def log(msg: str):
    print(f"[prep_usd] {msg}")


# ---------------------------------------------------------------------------
# Step 1: Realize collection instances
# ---------------------------------------------------------------------------

def realize_instances() -> int:
    """Convert collection instances (Alt+D linked duplicates) to real objects."""
    instance_objects = [
        obj for obj in bpy.context.scene.objects
        if obj.instance_type == 'COLLECTION' and obj.instance_collection is not None
    ]
    if not instance_objects:
        return 0

    deselect_all()
    for obj in instance_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = instance_objects[0]

    bpy.ops.object.duplicates_make_real(use_base_parent=True, use_hierarchy=True)
    log(f"Realized {len(instance_objects)} collection instance(s).")
    return len(instance_objects)


# ---------------------------------------------------------------------------
# Step 2: Realize particle system instances
# ---------------------------------------------------------------------------

def realize_particle_instances() -> Tuple[int, List[str]]:
    """
    Convert particle system instances (Object/Collection render_type) to real
    mesh objects.

    For every emitter that has a particle system whose render_type is 'OBJECT'
    or 'COLLECTION', this function evaluates the dependency graph to collect
    every individual instance transform, creates a standalone mesh object for
    each one (with the instance's world transform applied), then removes the
    original emitter object from the scene.

    Particle hair (strand rendering) is NOT handled and is left as-is.

    Returns (instances_created, emitter_names_removed).
    """
    # Find emitter objects that carry relevant particle systems
    emitters = []
    for obj in list(bpy.context.scene.objects):
        for ps in obj.particle_systems:
            if ps.settings.render_type in ('OBJECT', 'COLLECTION'):
                emitters.append(obj)
                break

    if not emitters:
        return 0, []

    log(f"  Found {len(emitters)} particle emitter(s): {[e.name for e in emitters]}")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    emitter_names = {obj.name for obj in emitters}

    # DepsgraphObjectInstance references are transient — they become invalid
    # as soon as the iterator advances or any scene change happens.
    # Extract mesh data and matrix immediately inside the loop.
    instance_data_by_emitter: dict = {name: [] for name in emitter_names}
    for inst in depsgraph.object_instances:
        if not inst.is_instance:
            continue
        if inst.parent is None:
            continue
        parent_name = inst.parent.name
        if parent_name not in instance_data_by_emitter:
            continue
        # Capture everything we need now, while inst is still valid
        eval_obj = inst.object.evaluated_get(depsgraph)
        new_mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
        matrix   = inst.matrix_world.copy()
        src_name = inst.object.name
        instance_data_by_emitter[parent_name].append((src_name, new_mesh, matrix))

    scene_collection = bpy.context.scene.collection
    created = 0
    emitters_removed: List[str] = []

    for emitter in emitters:
        data_list = instance_data_by_emitter.get(emitter.name, [])
        if not data_list:
            log(f"  No instances found for emitter '{emitter.name}' — skipping.")
            continue

        log(f"  Creating {len(data_list)} mesh object(s) from '{emitter.name}' particle system...")

        for i, (src_name, new_mesh, matrix) in enumerate(data_list):
            new_obj = bpy.data.objects.new(f"{src_name}_inst_{i:04d}", new_mesh)
            new_obj.matrix_world = matrix
            scene_collection.objects.link(new_obj)
            created += 1

        # The emitter itself is just the scatter surface — remove it
        emitter_name = emitter.name  # capture before remove() invalidates the ref
        log(f"  Removing emitter '{emitter_name}' from scene.")
        bpy.data.objects.remove(emitter, do_unlink=True)
        emitters_removed.append(emitter_name)

    return created, emitters_removed


# ---------------------------------------------------------------------------
# Step 3: Convert non-mesh types to mesh
# ---------------------------------------------------------------------------

NON_MESH_CONVERTIBLE = {'CURVE', 'SURFACE', 'META', 'FONT'}
SKIP_TYPES = {'VOLUME', 'GPENCIL', 'ARMATURE', 'LATTICE', 'EMPTY',
              'CAMERA', 'LIGHT', 'LIGHT_PROBE', 'SPEAKER'}

def convert_to_mesh() -> Tuple[int, List[str]]:
    """Convert curves, text, metaballs, NURBS to mesh. Returns (count, skipped_names)."""
    converted = 0
    skipped: List[str] = []

    # Snapshot name+type before the loop — convert() can invalidate any
    # object reference, including ones later in the same iteration.
    candidates = [(obj.name, obj.type) for obj in bpy.context.scene.objects]

    for obj_name, obj_type in candidates:
        if obj_type in NON_MESH_CONVERTIBLE:
            # Look up the live object by name; skip if already converted/removed
            obj = bpy.context.scene.objects.get(obj_name)
            if obj is None:
                log(f"  '{obj_name}' already removed (converted as part of another object).")
                continue
            try:
                select_only(obj)
                bpy.ops.object.convert(target='MESH')
                log(f"Converted '{obj_name}' ({obj_type} → MESH).")
                converted += 1
            except Exception as e:
                skipped.append(f"'{obj_name}' ({obj_type}): {e}")
        elif obj_type in SKIP_TYPES:
            pass  # expected non-geometry — silently ignore
        elif obj_type == 'POINTCLOUD':
            skipped.append(f"'{obj_name}' (POINTCLOUD): not supported in USD export pipeline")

    return converted, skipped


# ---------------------------------------------------------------------------
# Step 3: Apply modifiers
# ---------------------------------------------------------------------------

def has_shape_keys(obj) -> bool:
    return obj.data is not None and hasattr(obj.data, 'shape_keys') \
           and obj.data.shape_keys is not None \
           and len(obj.data.shape_keys.key_blocks) > 1  # >1 because Basis always exists

def safe_name(s) -> str:
    """Decode a potentially non-UTF-8 string safely for printing."""
    if isinstance(s, bytes):
        return s.decode('utf-8', errors='replace')
    try:
        # Force through bytes round-trip to catch bad internal encodings
        return s.encode('utf-8', errors='replace').decode('utf-8')
    except Exception:
        return repr(s)

def apply_modifiers() -> Tuple[int, List[str]]:
    """Apply all modifiers on all mesh objects. Returns (modifier_count, warnings)."""
    applied = 0
    warnings: List[str] = []

    for obj in list(bpy.context.scene.objects):
        if obj.type != 'MESH':
            continue
        if not obj.modifiers:
            continue

        # Shape keys block most modifier applications — warn and skip
        if has_shape_keys(obj):
            mod_names = [safe_name(m.name) for m in obj.modifiers]
            warnings.append(
                f"'{obj.name}': skipped modifiers {mod_names} — object has shape keys. "
                "Apply or remove shape keys manually before re-running."
            )
            continue

        # Make mesh data single-user so modifiers can be applied.
        # Shared (multi-user) mesh data blocks cannot have modifiers applied.
        if obj.data.users > 1:
            obj.data = obj.data.copy()
            log(f"  Made mesh data single-user for '{obj.name}'.")

        select_only(obj)

        for mod in list(obj.modifiers):
            mod_label = safe_name(mod.name)
            # Skip modifiers that are disabled (unchecked in the stack)
            if not mod.show_render:
                continue
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
                applied += 1
            except Exception as e:
                warnings.append(
                    f"'{obj.name}' modifier '{mod_label}': {e}"
                )
                # Remove the failed modifier so we don't block subsequent ones
                try:
                    obj.modifiers.remove(mod)
                    warnings[-1] += " (modifier removed)"
                except Exception:
                    pass

    return applied, warnings


# ---------------------------------------------------------------------------
# Step 4: Apply transforms
# ---------------------------------------------------------------------------

def apply_transforms() -> int:
    """Apply scale (and optionally rotation/location) to all mesh objects."""
    count = 0
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        # Make mesh data single-user — transform_apply aborts on shared data blocks
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        select_only(obj)
        bpy.ops.object.transform_apply(
            location=APPLY_LOCATION,
            rotation=APPLY_ROTATION,
            scale=True,
        )
        count += 1

    log(f"Applied transforms on {count} mesh object(s) "
        f"(scale=True, rotation={APPLY_ROTATION}, location={APPLY_LOCATION}).")
    return count


# ---------------------------------------------------------------------------
# Step 5: Bake unsupported shader nodes into textures
#
# UsdPreviewSurface only natively represents: Image Texture, UV Map,
# UsdTransform2d (Mapping), and Normal Map. Any Blender node between an
# Image Texture and a BSDF socket input that isn't one of those is silently
# dropped by Blender's USD exporter, producing wrong colors in the render.
#
# This step detects the most common unsupported intermediate nodes and applies
# their effect directly to the image pixel data, writing a new "_baked"
# texture alongside the original. The material node tree is then rewired to
# connect the baked texture directly to the BSDF, with the unsupported nodes
# removed.
#
# Handled analytically (no render pass needed):
#   - Invert Color      (ShaderNodeInvert) — any fac value
#   - Hue/Saturation    (ShaderNodeHueSaturation)
#   - Bright/Contrast   (ShaderNodeBrightContrast)
#   - Color Ramp        (ShaderNodeValToRGB) — per-pixel when driven by image;
#                         constant-folded when Fac is a constant
#   - Mix / Mix RGB     (ShaderNodeMix / ShaderNodeMixRGB) — image+constant blend
#                         or constant-folded when both inputs are constants
#   - Bump              (ShaderNodeBump) — Sobel-filter bake to normal map
#
# Not handled (emits a warning):
#   - RGB Curves, procedural textures with no image source,
#     and Mix nodes where both inputs are textured.
# ---------------------------------------------------------------------------

# Nodes whose type IDs are supported natively by UsdPreviewSurface / USD export.
# Everything else in the path from Image Texture → BSDF socket is unsupported.
_USD_PASSTHROUGH_TYPES = {
    'TEX_IMAGE',           # UsdUVTexture
    'UVMAP',               # UsdPrimvarReader_float2
    'MAPPING',             # UsdTransform2d
    'NORMAL_MAP',          # UsdPreviewSurface normal input
    'OUTPUT_MATERIAL',
    'BSDF_PRINCIPLED',
    'GROUP_INPUT',
    'GROUP_OUTPUT',
}


def _image_file_exists(img) -> bool:
    """Return True if the image has a readable file on disk."""
    if not img.filepath:
        return False
    path = bpy.path.abspath(img.filepath)
    return os.path.isfile(path)


def _image_has_pixels(img) -> bool:
    """Return True if the image has pixel data accessible in memory."""
    try:
        return img.size[0] > 0 and img.size[1] > 0 and len(img.pixels) > 0
    except Exception:
        return False


def _load_pixels_rgba(img) -> list:
    """Return a flat list of [r,g,b,a, ...] floats from a Blender Image."""
    # Only reload from disk if the file actually exists — avoids errors for
    # images whose source is in read-only caches or relative to a blend file
    # that isn't in the current working directory.
    if _image_file_exists(img):
        try:
            img.reload()
        except Exception:
            pass
    pixels = list(img.pixels)
    return pixels


def _save_baked_image(src_img, pixels_rgba: list, suffix: str,
                      out_dir: str) -> str:
    """
    Create a new Blender Image with modified pixels and save it to disk.
    Always saves into out_dir (next to the USD file) so we never try to
    write into read-only locations like BlenderKit's asset cache.
    Returns the absolute file path of the saved image.
    """
    w, h = src_img.size
    baked = bpy.data.images.new(
        name=src_img.name + suffix,
        width=w, height=h,
        alpha=True, float_buffer=src_img.is_float
    )
    baked.pixels[:] = pixels_rgba
    baked.update()

    # Derive a filename from the original image name, sanitised for the filesystem.
    src_basename = os.path.splitext(os.path.basename(
        bpy.path.abspath(src_img.filepath) or src_img.name
    ))[0]
    safe_basename = "".join(c if c.isalnum() or c in ('_', '-') else '_'
                            for c in src_basename)
    textures_dir = os.path.join(out_dir, "textures")
    os.makedirs(textures_dir, exist_ok=True)
    save_path = os.path.join(textures_dir, safe_basename + suffix + ".png")

    baked.filepath_raw = save_path
    baked.file_format = 'PNG'
    baked.save()
    return save_path


def _invert_pixels(pixels: list, fac: float = 1.0) -> list:
    """Invert R, G, B channels with blend factor; leave A unchanged."""
    out = list(pixels)
    for i in range(0, len(out), 4):
        out[i]   = out[i]   * (1.0 - fac) + (1.0 - out[i])   * fac
        out[i+1] = out[i+1] * (1.0 - fac) + (1.0 - out[i+1]) * fac
        out[i+2] = out[i+2] * (1.0 - fac) + (1.0 - out[i+2]) * fac
    return out


def _evaluate_color_ramp(ramp_node, fac: float) -> list:
    """Evaluate a Color Ramp node at scalar fac ∈ [0,1]. Returns [r,g,b,a]."""
    ramp = ramp_node.color_ramp
    elements = sorted(ramp.elements, key=lambda e: e.position)
    if not elements:
        return [0.0, 0.0, 0.0, 1.0]
    fac = max(0.0, min(1.0, fac))
    if ramp.interpolation == 'CONSTANT':
        col = elements[0].color[:]
        for e in elements:
            if e.position <= fac:
                col = e.color[:]
        return list(col)
    # LINEAR (and approximation for EASE/CARDINAL/B_SPLINE)
    if fac <= elements[0].position:
        return list(elements[0].color[:])
    if fac >= elements[-1].position:
        return list(elements[-1].color[:])
    for i in range(len(elements) - 1):
        e0, e1 = elements[i], elements[i + 1]
        if e0.position <= fac <= e1.position:
            span = e1.position - e0.position
            t = (fac - e0.position) / span if span > 1e-8 else 0.0
            c0, c1 = e0.color[:], e1.color[:]
            return [c0[j] * (1 - t) + c1[j] * t for j in range(4)]
    return list(elements[-1].color[:])


def _apply_color_ramp_pixels(grey_pixels: list, ramp_node) -> list:
    """Apply a Color Ramp to a greyscale pixel buffer (reads R channel as fac).
    Returns a new RGBA pixel list."""
    out = []
    for i in range(0, len(grey_pixels), 4):
        fac = grey_pixels[i]  # R channel as 0-1 scalar
        col = _evaluate_color_ramp(ramp_node, fac)
        out.extend(col[:4])
    return out


def _mix_color(col1, col2, fac: float, blend_type: str) -> list:
    """Evaluate one MIX_RGB / Mix blend between two [r,g,b] colour lists."""
    def clamp(v): return max(0.0, min(1.0, v))
    r1, g1, b1 = col1[0], col1[1], col1[2]
    r2, g2, b2 = col2[0], col2[1], col2[2]
    if blend_type in ('MIX', 'MIX'):
        r = r1 + fac * (r2 - r1)
        g = g1 + fac * (g2 - g1)
        b = b1 + fac * (b2 - b1)
    elif blend_type == 'ADD':
        r = r1 + fac * r2
        g = g1 + fac * g2
        b = b1 + fac * b2
    elif blend_type == 'MULTIPLY':
        r = r1 * (1 - fac + fac * r2)
        g = g1 * (1 - fac + fac * g2)
        b = b1 * (1 - fac + fac * b2)
    elif blend_type == 'SUBTRACT':
        r = r1 - fac * r2
        g = g1 - fac * g2
        b = b1 - fac * b2
    elif blend_type == 'SCREEN':
        r = 1 - (1 - r1) * (1 - fac * r2)
        g = 1 - (1 - g1) * (1 - fac * g2)
        b = 1 - (1 - b1) * (1 - fac * b2)
    elif blend_type == 'OVERLAY':
        def ov(a, b_v):
            return 2*a*b_v if a < 0.5 else 1 - 2*(1-a)*(1-b_v)
        r = r1 + fac * (ov(r1, r2) - r1)
        g = g1 + fac * (ov(g1, g2) - g1)
        b = b1 + fac * (ov(b1, b2) - b1)
    else:
        r = r1 + fac * (r2 - r1)
        g = g1 + fac * (g2 - g1)
        b = b1 + fac * (b2 - b1)
    return [clamp(r), clamp(g), clamp(b), 1.0]


def _mix_pixels_with_constant(pixels: list, const_rgba: list,
                               fac: float, blend_type: str,
                               const_is_b: bool) -> list:
    """Mix each pixel with a constant colour.
    const_is_b=True  → pixel is A, const is B  (result blends pixel→const)
    const_is_b=False → const is A, pixel is B  (result blends const→pixel)"""
    out = list(pixels)
    for i in range(0, len(out), 4):
        px = [out[i], out[i+1], out[i+2]]
        if const_is_b:
            col = _mix_color(px, const_rgba, fac, blend_type)
        else:
            col = _mix_color(const_rgba, px, fac, blend_type)
        out[i], out[i+1], out[i+2] = col[0], col[1], col[2]
    return out


def _height_to_normal_pixels(pixels: list, w: int, h: int,
                              strength: float = 1.0) -> list:
    """Convert a height map (reads R channel) to a tangent-space normal map
    using a 3×3 Sobel filter.  Returns RGBA pixels encoded as (n+1)/2."""
    try:
        import numpy as np
        arr = np.array(pixels, dtype=np.float32).reshape(h, w, 4)
        # Luminance as height
        lum = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1]
               + 0.0722 * arr[:, :, 2])
        # Pad with wrap for Sobel
        p = np.pad(lum, 1, mode='wrap')
        # Sobel X
        dx = (p[0:-2, 2:] + 2*p[1:-1, 2:] + p[2:, 2:]
              - p[0:-2, 0:-2] - 2*p[1:-1, 0:-2] - p[2:, 0:-2]) * strength
        # Sobel Y
        dy = (p[2:, 0:-2] + 2*p[2:, 1:-1] + p[2:, 2:]
              - p[0:-2, 0:-2] - 2*p[0:-2, 1:-1] - p[0:-2, 2:]) * strength
        nx = -dx
        ny = -dy
        nz = np.ones_like(nx)
        length = np.sqrt(nx*nx + ny*ny + nz*nz)
        length = np.maximum(length, 1e-8)
        nx, ny, nz = nx/length, ny/length, nz/length
        out = np.stack([nx*0.5+0.5, ny*0.5+0.5, nz*0.5+0.5,
                        np.ones_like(nx)], axis=-1)
        return out.flatten().tolist()
    except ImportError:
        # Pure-Python fallback (slow — only practical for small textures)
        if w * h > 256 * 256:
            return None  # signal caller to skip
        def lum_at(xx, yy):
            idx = (yy % h * w + xx % w) * 4
            return 0.2126*pixels[idx] + 0.7152*pixels[idx+1] + 0.0722*pixels[idx+2]
        out = []
        for py in range(h):
            for px in range(w):
                dx = (lum_at(px+1, py-1) + 2*lum_at(px+1, py) + lum_at(px+1, py+1)
                      - lum_at(px-1, py-1) - 2*lum_at(px-1, py) - lum_at(px-1, py+1)) * strength
                dy = (lum_at(px-1, py+1) + 2*lum_at(px, py+1) + lum_at(px+1, py+1)
                      - lum_at(px-1, py-1) - 2*lum_at(px, py-1) - lum_at(px+1, py-1)) * strength
                nx, ny, nz = -dx, -dy, 1.0
                ln = (nx*nx + ny*ny + nz*nz) ** 0.5
                out += [nx/ln*0.5+0.5, ny/ln*0.5+0.5, nz/ln*0.5+0.5, 1.0]
        return out


def _hue_sat_val_pixels(pixels: list, hue: float, sat: float, val: float) -> list:
    """Apply Blender's Hue/Saturation/Value adjustment (HSV shift)."""
    import colorsys
    out = list(pixels)
    for i in range(0, len(out), 4):
        r, g, b = out[i], out[i+1], out[i+2]
        h, s, v = colorsys.rgb_to_hsv(max(0, min(1, r)),
                                       max(0, min(1, g)),
                                       max(0, min(1, b)))
        h = (h + hue - 0.5) % 1.0
        s = max(0.0, min(1.0, s * sat))
        v = max(0.0, min(1.0, v * val))
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
        out[i], out[i+1], out[i+2] = r2, g2, b2
    return out


def _bright_contrast_pixels(pixels: list, bright: float, contrast: float) -> list:
    """Apply Blender's Bright/Contrast node formula."""
    # Blender formula: out = (in + bright) * (1 + contrast) - contrast * 0.5
    out = list(pixels)
    for i in range(0, len(out), 4):
        for c in range(3):
            v = out[i + c]
            v = (v + bright) * (1.0 + contrast) - contrast * 0.5
            out[i + c] = max(0.0, min(1.0, v))
    return out


def _find_upstream_image_tex(node, visited=None):
    """
    Walk backwards from `node` through its inputs to find the first
    TEX_IMAGE node.  Returns (image_tex_node, socket_name_on_that_node)
    or (None, None) if not found.
    """
    if visited is None:
        visited = set()
    if node.name in visited:
        return None, None
    visited.add(node.name)

    for inp in node.inputs:
        for link in inp.links:
            src = link.from_node
            if src.type == 'TEX_IMAGE':
                return src, link.from_socket.name
            result = _find_upstream_image_tex(src, visited)
            if result[0]:
                return result
    return None, None


def _get_unsupported_chain(from_node, to_socket, visited=None):
    """
    Collect unsupported nodes in the path from `from_node` toward `to_socket`.
    Returns a list of (node, input_link) pairs in order closest-to-image first.
    """
    if visited is None:
        visited = set()
    result = []
    for inp in from_node.inputs:
        for link in inp.links:
            src = link.from_node
            if src.name in visited:
                continue
            visited.add(src.name)
            if src.type not in _USD_PASSTHROUGH_TYPES:
                result.append(src)
            result.extend(_get_unsupported_chain(src, None, visited))
    return result


def bake_unsupported_nodes(out_dir: str) -> Tuple[int, List[str]]:
    """
    Walk all materials. For each BSDF input that has an unsupported node
    between it and its source Image Texture, apply the operation analytically
    and rewire the material.
    Returns (baked_count, warnings).
    """
    baked = 0
    warnings: List[str] = []

    for mat in bpy.data.materials:
        if not mat.use_nodes or not mat.node_tree:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find the Principled BSDF node (or output surface node)
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf:
            continue

        # Check each color/value input socket on the BSDF
        for sock in bsdf.inputs:
            if not sock.links:
                continue
            link = sock.links[0]
            src_node = link.from_node

            # Log what we find for every non-trivial connection
            if src_node.type not in _USD_PASSTHROUGH_TYPES:
                log(f"  Material '{mat.name}', socket '{sock.name}': "
                    f"unsupported node '{src_node.name}' (type={src_node.type})")

            # If already a supported node, nothing to do
            if src_node.type in _USD_PASSTHROUGH_TYPES:
                continue

            # ----------------------------------------------------------------
            # Constant-fold: VALTORGB / MIX(+MIX_RGB) with no upstream image
            # ----------------------------------------------------------------
            img_node, _ = _find_upstream_image_tex(src_node)
            if img_node is None or img_node.image is None:
                folded = False

                if src_node.type == 'VALTORGB':
                    fac_inp = src_node.inputs.get('Fac')
                    if fac_inp and not fac_inp.links:
                        result = _evaluate_color_ramp(src_node, fac_inp.default_value)
                        try:
                            if hasattr(sock, 'default_value'):
                                dv = sock.default_value
                                if hasattr(dv, '__len__') and len(dv) >= 3:
                                    sock.default_value = (result[0], result[1], result[2], 1.0)
                                else:
                                    sock.default_value = result[0]
                        except Exception:
                            pass
                        links.remove(link)
                        nodes.remove(src_node)
                        log(f"  Material '{mat.name}', socket '{sock.name}': "
                            f"Color Ramp constant-folded → ({result[0]:.3f},{result[1]:.3f},{result[2]:.3f})")
                        folded = True

                elif src_node.type in ('MIX_RGB', 'MIX'):
                    # Determine input socket names (Blender 4.x MIX vs old MIX_RGB)
                    a_names = ('Color1', 'A')
                    b_names = ('Color2', 'B')
                    f_names = ('Fac', 'Factor')
                    fac_inp  = next((src_node.inputs[n] for n in f_names if n in src_node.inputs), None)
                    col1_inp = next((src_node.inputs[n] for n in a_names if n in src_node.inputs), None)
                    col2_inp = next((src_node.inputs[n] for n in b_names if n in src_node.inputs), None)
                    if (fac_inp and not fac_inp.links and
                            col1_inp and not col1_inp.links and
                            col2_inp and not col2_inp.links):
                        fac  = float(fac_inp.default_value)
                        col1 = list(col1_inp.default_value)[:4]
                        col2 = list(col2_inp.default_value)[:4]
                        bt   = getattr(src_node, 'blend_type', 'MIX')
                        result = _mix_color(col1, col2, fac, bt)
                        try:
                            if hasattr(sock, 'default_value'):
                                dv = sock.default_value
                                if hasattr(dv, '__len__') and len(dv) >= 3:
                                    sock.default_value = (result[0], result[1], result[2], 1.0)
                                else:
                                    sock.default_value = result[0]
                        except Exception:
                            pass
                        links.remove(link)
                        nodes.remove(src_node)
                        log(f"  Material '{mat.name}', socket '{sock.name}': "
                            f"Mix constant-folded → ({result[0]:.3f},{result[1]:.3f},{result[2]:.3f})")
                        folded = True

                if not folded:
                    warnings.append(
                        f"Material '{mat.name}', socket '{sock.name}': "
                        f"unsupported node '{src_node.type}' with no Image Texture "
                        f"source — skipped (manual bake required)."
                    )
                continue

            src_img = img_node.image
            if src_img.size[0] == 0 or src_img.size[1] == 0:
                warnings.append(
                    f"Material '{mat.name}': image '{src_img.name}' has zero size — skipped."
                )
                continue
            if not src_img.packed_file and not _image_file_exists(src_img) \
                    and not _image_has_pixels(src_img):
                warnings.append(
                    f"Material '{mat.name}': image '{src_img.name}' has no pixel data "
                    f"(file not found: '{bpy.path.abspath(src_img.filepath)}') — skipped."
                )
                continue

            # ----------------------------------------------------------------
            # BUMP: special path — bake Height map → normal map texture,
            # then wire TEX_IMAGE → NORMAL_MAP node → bsdf Normal socket.
            # ----------------------------------------------------------------
            if src_node.type == 'BUMP':
                height_inp = src_node.inputs.get('Height')
                strength_val = float(src_node.inputs['Strength'].default_value
                                     if 'Strength' in src_node.inputs else 1.0)
                bump_img_node = None
                if height_inp and height_inp.links:
                    hn = height_inp.links[0].from_node
                    if hn.type == 'TEX_IMAGE':
                        bump_img_node = hn
                    else:
                        bump_img_node, _ = _find_upstream_image_tex(hn)

                if bump_img_node is None or bump_img_node.image is None:
                    warnings.append(
                        f"Material '{mat.name}', socket '{sock.name}': "
                        f"BUMP with no upstream image — skipped."
                    )
                    continue

                bump_src = bump_img_node.image
                if bump_src.size[0] == 0 or bump_src.size[1] == 0:
                    warnings.append(
                        f"Material '{mat.name}': BUMP image has zero size — skipped.")
                    continue

                bw, bh = bump_src.size
                norm_pixels = _height_to_normal_pixels(
                    _load_pixels_rgba(bump_src), bw, bh, strength_val)
                if norm_pixels is None:
                    warnings.append(
                        f"Material '{mat.name}': BUMP image too large for "
                        f"pure-Python bake ({bw}×{bh}) — install numpy or reduce resolution.")
                    continue

                try:
                    save_path = _save_baked_image(bump_src, norm_pixels, '_baked_NormalFromBump', out_dir)
                except Exception as e:
                    warnings.append(f"Material '{mat.name}': BUMP bake save failed: {e}")
                    continue

                norm_img = bpy.data.images.load(save_path)
                norm_img.colorspace_settings.name = 'Non-Color'

                new_tex = nodes.new('ShaderNodeTexImage')
                new_tex.image = norm_img
                new_tex.location = src_node.location[0] - 300, src_node.location[1]

                # Copy UV links from the height image node to the new tex node
                for old_lnk in bump_img_node.inputs['Vector'].links:
                    links.new(old_lnk.from_socket, new_tex.inputs['Vector'])

                nm_node = nodes.new('ShaderNodeNormalMap')
                nm_node.location = src_node.location[0], src_node.location[1]
                links.new(new_tex.outputs['Color'], nm_node.inputs['Color'])
                links.new(nm_node.outputs['Normal'], sock)

                nodes.remove(src_node)
                log(f"  Material '{mat.name}', socket '{sock.name}': "
                    f"BUMP baked → {os.path.basename(save_path)}")
                baked += 1
                continue

            # Collect the chain of unsupported nodes between img_node and bsdf
            _CHAIN_INPUT_NAMES = {'Color', 'Image', 'Value', '', 'Fac', 'Height', 'Factor', 'A', 'B'}
            chain = []
            node = src_node
            while node and node.type not in _USD_PASSTHROUGH_TYPES:
                chain.append(node)
                next_node = None
                for inp in node.inputs:
                    if inp.links and inp.name in _CHAIN_INPUT_NAMES:
                        cand = inp.links[0].from_node
                        if cand.type != 'TEX_IMAGE':
                            next_node = cand
                            break
                if next_node is None or next_node == img_node:
                    break
                node = next_node
            chain.reverse()  # image → bsdf order

            # Apply each node in the chain analytically
            pixels = _load_pixels_rgba(src_img)
            applied_ops = []
            unhandled = []

            for n in chain:
                if n.type == 'INVERT':
                    fac = float(n.inputs['Fac'].default_value) if 'Fac' in n.inputs else 1.0
                    pixels = _invert_pixels(pixels, fac)
                    applied_ops.append(f'Invert(fac={fac:.2f})')
                elif n.type == 'HUE_SAT':
                    h = n.inputs['Hue'].default_value        if 'Hue'        in n.inputs else 0.5
                    s = n.inputs['Saturation'].default_value if 'Saturation' in n.inputs else 1.0
                    v = n.inputs['Value'].default_value      if 'Value'      in n.inputs else 1.0
                    pixels = _hue_sat_val_pixels(pixels, h, s, v)
                    applied_ops.append(f'HueSat(h={h:.2f},s={s:.2f},v={v:.2f})')
                elif n.type == 'BRIGHTCONTRAST':
                    br = n.inputs['Bright'].default_value   if 'Bright'   in n.inputs else 0.0
                    co = n.inputs['Contrast'].default_value if 'Contrast' in n.inputs else 0.0
                    pixels = _bright_contrast_pixels(pixels, br, co)
                    applied_ops.append(f'BrightContrast(br={br:.2f},co={co:.2f})')
                elif n.type == 'VALTORGB':
                    # Apply color ramp using the upstream greyscale image as Fac
                    pixels = _apply_color_ramp_pixels(pixels, n)
                    applied_ops.append('ColorRamp')
                elif n.type in ('MIX_RGB', 'MIX'):
                    a_names = ('Color1', 'A')
                    b_names = ('Color2', 'B')
                    f_names = ('Fac', 'Factor')
                    fac_inp  = next((n.inputs[nm] for nm in f_names if nm in n.inputs), None)
                    col1_inp = next((n.inputs[nm] for nm in a_names if nm in n.inputs), None)
                    col2_inp = next((n.inputs[nm] for nm in b_names if nm in n.inputs), None)
                    fac_v = float(fac_inp.default_value) if fac_inp and not fac_inp.links else 0.5
                    bt = getattr(n, 'blend_type', 'MIX')
                    # Determine which input is the texture (already loaded in pixels)
                    # and which is a constant.  We assume the first linked-to-image
                    # input is whichever fed into this node via the chain.
                    # If col1 has no links its value is constant; pixel is "A".
                    if col1_inp and not col1_inp.links:
                        const = list(col1_inp.default_value)[:4]
                        pixels = _mix_pixels_with_constant(pixels, const, fac_v, bt, const_is_b=False)
                    elif col2_inp and not col2_inp.links:
                        const = list(col2_inp.default_value)[:4]
                        pixels = _mix_pixels_with_constant(pixels, const, fac_v, bt, const_is_b=True)
                    else:
                        unhandled.append(f'{n.type}(both-inputs-textured)')
                        continue
                    applied_ops.append(f'Mix({bt},fac={fac_v:.2f})')
                else:
                    unhandled.append(n.type)

            if unhandled:
                warnings.append(
                    f"Material '{mat.name}', socket '{sock.name}': "
                    f"node(s) {unhandled} cannot be baked analytically — "
                    f"manual Cycles bake required. Partial ops applied: {applied_ops or 'none'}."
                )
                if not applied_ops:
                    continue  # nothing changed, don't write a useless texture

            # Save the baked texture
            suffix = "_baked_" + "_".join(applied_ops).replace('/', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '').replace('.', '')[:40]
            try:
                save_path = _save_baked_image(src_img, pixels, suffix, out_dir)
            except Exception as e:
                warnings.append(
                    f"Material '{mat.name}': failed to save baked texture: {e}"
                )
                continue

            # Load the saved image as a new Blender Image and rewire
            baked_img = bpy.data.images.load(save_path)
            baked_img.colorspace_settings.name = src_img.colorspace_settings.name

            # Create a new Image Texture node pointing at the baked image
            new_tex = nodes.new('ShaderNodeTexImage')
            new_tex.image = baked_img
            new_tex.location = img_node.location[0] + 300, img_node.location[1]

            # Copy UV/Mapping links from the original image node to the new one
            for old_link in img_node.inputs['Vector'].links:
                links.new(old_link.from_socket, new_tex.inputs['Vector'])

            # Remove old chain nodes and rewire baked tex → bsdf socket
            links.new(new_tex.outputs['Color'], sock)
            for n in chain:
                nodes.remove(n)

            log(f"  Material '{mat.name}', socket '{sock.name}': "
                f"baked [{', '.join(applied_ops)}] → {os.path.basename(save_path)}")
            baked += 1

    return baked, warnings


# ---------------------------------------------------------------------------
# Step 6: Convert Glass BSDF materials to Principled BSDF with opacity=0
#
# Blender's Glass BSDF node has no equivalent in UsdPreviewSurface. When the
# USD exporter encounters a material whose output is connected to a Glass BSDF
# (or a Mix Shader combining Glass with anything), it produces an empty material
# shell with outputs:surface = None. Anacapa then gets a grey Lambertian fallback
# and the windows/glass objects are opaque, blocking all light.
#
# The UsdPreviewSurface convention for glass is: opacity=0, IOR from the glass
# node. Blender's USD exporter translates that correctly to a dielectric shader.
#
# This step:
#   1. Finds any material whose surface output connects (directly or via Mix
#      Shader) to a Glass BSDF node.
#   2. Replaces the material node tree with a minimal Principled BSDF wired as:
#        opacity = 0, IOR = (Glass BSDF IOR), roughness = (Glass BSDF roughness),
#        base color = (Glass BSDF color), transmission = 1.
#   3. Logs the conversion so the artist can verify the result.
# ---------------------------------------------------------------------------

def convert_glass_materials() -> Tuple[int, List[str]]:
    """
    Detect materials driven by a Glass BSDF and replace them with a
    Principled BSDF configured as glass (opacity=0, IOR, transmission=1).
    Returns (converted_count, warnings).
    """
    converted = 0
    warnings: List[str] = []

    for mat in bpy.data.materials:
        if not mat.use_nodes or not mat.node_tree:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find the Material Output node
        output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if not output:
            continue

        # Walk the Surface input to find a Glass BSDF, possibly through Mix Shaders
        surface_links = output.inputs['Surface'].links
        if not surface_links:
            continue

        def find_glass_bsdf(node, depth=0):
            """Recursively find a BSDF_GLASS node reachable from this node."""
            if depth > 5:
                return None
            if node.type == 'BSDF_GLASS':
                return node
            # Mix Shader: check both shader inputs
            if node.type == 'MIX_SHADER':
                for inp in node.inputs:
                    if inp.name in ('Shader', 'Shader_001') or inp.name.startswith('Shader'):
                        for lnk in inp.links:
                            result = find_glass_bsdf(lnk.from_node, depth + 1)
                            if result:
                                return result
            return None

        src_node = surface_links[0].from_node
        glass_node = find_glass_bsdf(src_node)
        if glass_node is None:
            continue

        # Read glass parameters
        ior = glass_node.inputs['IOR'].default_value if 'IOR' in glass_node.inputs else 1.5
        roughness = glass_node.inputs['Roughness'].default_value if 'Roughness' in glass_node.inputs else 0.0
        color = tuple(glass_node.inputs['Color'].default_value[:3]) if 'Color' in glass_node.inputs else (1.0, 1.0, 1.0)

        # Clear existing nodes and build a minimal Principled BSDF glass setup
        nodes.clear()

        out_node = nodes.new('ShaderNodeOutputMaterial')
        out_node.location = (300, 0)

        pbsdf = nodes.new('ShaderNodeBsdfPrincipled')
        pbsdf.location = (0, 0)

        # Glass via UsdPreviewSurface convention: opacity=0, IOR, Transmission=1
        pbsdf.inputs['Base Color'].default_value    = (*color, 1.0)
        pbsdf.inputs['Roughness'].default_value     = roughness
        pbsdf.inputs['IOR'].default_value           = ior
        pbsdf.inputs['Alpha'].default_value         = 0.0   # opacity=0 → glass in USD

        # Blender 4.x uses 'Transmission Weight'; older versions use 'Transmission'
        for trans_name in ('Transmission Weight', 'Transmission'):
            if trans_name in pbsdf.inputs:
                pbsdf.inputs[trans_name].default_value = 1.0
                break

        links.new(pbsdf.outputs['BSDF'], out_node.inputs['Surface'])

        # Set blend mode so Blender's viewport also shows it as transparent
        mat.blend_method = 'BLEND'

        log(f"  Converted glass material '{mat.name}': "
            f"IOR={ior:.2f} roughness={roughness:.3f} color={color}")
        converted += 1

    return converted, warnings


# ---------------------------------------------------------------------------
# Step 8: Strip custom properties that the USD exporter cannot serialize
#
# Blender stores various addon metadata, node editor state, and other
# per-object/per-material data as custom properties (RNA ID properties).
# The USD exporter tries to write all of them as userProperties:* attributes
# but emits "Couldn't determine USD type name for array property" warnings
# for any type it doesn't know how to represent (e.g. node-tree references,
# ID-array properties, Python objects stored as bytes, etc.).
#
# We strip known-problematic property names and any property whose value is
# a type the USD exporter can't represent before export.
# ---------------------------------------------------------------------------

# Property names known to cause "Couldn't determine USD type" warnings.
# These are typically Blender-internal or addon-specific bookkeeping values.
_UNSUPPORTED_PROP_NAMES = {
    'em_nodes',       # node editor / addon node metadata
    'cycles',         # Cycles render settings stored on objects
    'cycles_visibility',
}

# Supported ID-property value types for USD export (scalar or homogeneous arrays)
_USD_SUPPORTED_TYPES = (bool, int, float, str, bytes)


def _prop_is_exportable(val) -> bool:
    """Return True if the property value can be represented in USD."""
    if isinstance(val, _USD_SUPPORTED_TYPES):
        return True
    if isinstance(val, (list, tuple)) and val:
        return all(isinstance(v, (int, float)) for v in val)
    # IDPropertyArray from bpy (mathutils vectors etc.) are fine
    try:
        import idprop
        if isinstance(val, idprop.types.IDPropertyArray):
            return True
    except ImportError:
        pass
    return False


def strip_unsupported_custom_props() -> int:
    """
    Remove custom properties that the USD exporter cannot serialize from all
    objects, meshes, materials, and lights in the scene.
    Returns the number of properties removed.
    """
    removed = 0
    targets = list(bpy.data.objects) + list(bpy.data.meshes) + \
              list(bpy.data.materials) + list(bpy.data.lights)

    for datablock in targets:
        keys_to_remove = []
        for key in datablock.keys():
            if key.startswith('_'):          # internal RNA, skip
                continue
            if key in _UNSUPPORTED_PROP_NAMES:
                keys_to_remove.append(key)
                continue
            try:
                val = datablock[key]
                if not _prop_is_exportable(val):
                    keys_to_remove.append(key)
            except Exception:
                keys_to_remove.append(key)  # can't read it → can't export it

        for key in keys_to_remove:
            try:
                del datablock[key]
                log(f"  Removed unsupported property '{key}' from '{datablock.name}'.")
                removed += 1
            except Exception as e:
                log(f"  Could not remove '{key}' from '{datablock.name}': {e}")

    return removed


# ---------------------------------------------------------------------------
# Step 9: Remove render-hidden cutter helpers
# ---------------------------------------------------------------------------

def remove_render_hidden() -> Tuple[int, List[str]]:
    """Remove objects hidden from render that have no children."""
    removed = 0
    kept: List[str] = []

    # Build parent → children map
    children_of = {obj.name: [] for obj in bpy.context.scene.objects}
    for obj in bpy.context.scene.objects:
        if obj.parent:
            children_of.setdefault(obj.parent.name, []).append(obj.name)

    for obj in list(bpy.context.scene.objects):
        if obj.hide_render and not children_of.get(obj.name):
            log(f"Removing render-hidden object '{obj.name}' ({obj.type}).")
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        elif obj.hide_render:
            kept.append(obj.name)

    return removed, kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  Anacapa USD Export Prep")
    print("=" * 60)

    # Ensure we're in object mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # --- Step 1: Collection instances ---
    log("Step 1: Realizing collection instances...")
    n_instances = realize_instances()

    # --- Step 2: Particle system instances ---
    log("Step 2: Realizing particle system instances (Object/Collection render type)...")
    n_particle_instances, particle_emitters_removed = realize_particle_instances()

    # --- Step 3: Convert non-mesh ---
    log("Step 3: Converting non-mesh objects to mesh...")
    n_converted, convert_skipped = convert_to_mesh()

    # --- Step 4: Apply modifiers ---
    log("Step 4: Applying modifiers...")
    n_mods, mod_warnings = apply_modifiers()

    # --- Step 5: Apply transforms ---
    log("Step 5: Applying transforms...")
    n_transforms = apply_transforms()

    # --- Step 6: Bake unsupported shader nodes ---
    log("Step 6: Baking unsupported shader nodes into textures...")
    # We need the output dir to save baked textures alongside the USD.
    # Parse it early here just to get the directory.
    import sys, argparse as _ap
    _argv = sys.argv
    _script_args = _argv[_argv.index("--") + 1:] if "--" in _argv else []
    _parser = _ap.ArgumentParser(add_help=False)
    _parser.add_argument("output")
    _parsed, _ = _parser.parse_known_args(_script_args)
    _bake_dir = os.path.dirname(os.path.abspath(_parsed.output)) if _parsed.output else "."
    n_baked, bake_warnings = bake_unsupported_nodes(_bake_dir)

    # --- Step 7: Convert Glass BSDF materials ---
    log("Step 7: Converting Glass BSDF materials to Principled BSDF (opacity=0)...")
    n_glass, glass_warnings = convert_glass_materials()

    # --- Step 8: Strip unsupported custom properties ---
    log("Step 8: Stripping unsupported custom properties...")
    n_props_removed = strip_unsupported_custom_props()

    # --- Step 9: Remove hidden cutters ---
    if REMOVE_RENDER_HIDDEN:
        log("Step 9: Removing render-hidden helper objects...")
        n_removed, hidden_kept = remove_render_hidden()
    else:
        n_removed = 0
        hidden_kept = []

    # --- Summary ---
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Collection instances realized : {n_instances}")
    print(f"  Particle instances created    : {n_particle_instances}"
          + (f" (removed emitters: {', '.join(particle_emitters_removed)})"
             if particle_emitters_removed else ""))
    print(f"  Objects converted to mesh     : {n_converted}")
    print(f"  Modifiers applied             : {n_mods}")
    print(f"  Objects with transforms fixed : {n_transforms}")
    print(f"  Materials with baked nodes    : {n_baked}")
    print(f"  Glass materials converted     : {n_glass}")
    print(f"  Unsupported custom props strip: {n_props_removed}")
    print(f"  Render-hidden objects removed : {n_removed}")

    if hidden_kept:
        print()
        print("  Render-hidden objects with children (kept):")
        for name in hidden_kept:
            print(f"    - {name}")

    all_warnings = convert_skipped + mod_warnings + bake_warnings + glass_warnings
    if all_warnings:
        print()
        print("  Warnings — manual attention required:")
        for w in all_warnings:
            print(f"    ! {w}")

    print()
    print("  Known limitations (not handled by this script):")
    print("    - Particle hair (strand rendering) — must be baked/exported separately")
    print("    - Volume / VDB objects")
    print("    - Grease Pencil objects")
    print("    - Library-linked objects that cannot be made local")

    # --- Export USD ---
    import sys, argparse
    # Arguments after '--' on the Blender command line land in sys.argv
    # after the '--' separator.
    argv = sys.argv
    script_args = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(
        prog="blender_prep_for_usd_export.py",
        description="Prep Blender scene and export to USD for Anacapa."
    )
    parser.add_argument("output",
                        help="Output USD file path (e.g. scene.usda or scene.usdc)")
    args = parser.parse_args(script_args)

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path) or "."
    if not os.path.isdir(out_dir):
        print()
        print(f"  ERROR: output directory does not exist: {out_dir}")
        print("=" * 60)
        sys.exit(1)
    print()
    print(f"  Resolved output path : {out_path}")
    print(f"  Directory exists     : {os.path.isdir(os.path.dirname(out_path))}")
    print(f"  Blender version      : {bpy.app.version_string}")
    print(f"  Exporting USD...")

    result = bpy.ops.wm.usd_export(
        filepath=out_path,
        export_animation=False,
        export_hair=False,
        export_uvmaps=True,
        export_normals=True,
        export_materials=True,
        use_instancing=False,
        generate_preview_surface=True,
        generate_materialx_network=True,
        export_textures_mode='NEW',
        overwrite_textures=False,
    )
    print(f"  Operator result      : {result}")

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  Done. ({size} bytes written to {out_path})")
    else:
        print(f"  ERROR: output file was not created at {out_path}")
        # Check if it landed next to the blend file instead
        blend_dir = os.path.dirname(bpy.data.filepath)
        alt = os.path.join(blend_dir, os.path.basename(out_path))
        if os.path.exists(alt):
            print(f"  (file found at {alt} instead — path was resolved relative to blend file)")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Inject ND_image_* nodes into OpenPBR subgraphs that are missing textures
    # (Blender emits OpenPBR terminals with literal values; textures only exist
    #  in the UsdPreviewSurface subgraph — we bridge them here so the graph is
    #  self-contained before OSL/MaterialX evaluation.)
    # -----------------------------------------------------------------------
    if os.path.exists(out_path):
        _inject_materialx_textures(out_path)

    # -----------------------------------------------------------------------
    # Extract MaterialX node graphs → JSON sidecar
    # -----------------------------------------------------------------------
    if os.path.exists(out_path):
        _extract_materialx_sidecar(out_path)


# ---------------------------------------------------------------------------
# MaterialX texture injector
# ---------------------------------------------------------------------------
# Maps UsdPreviewSurface input → (OpenPBR input, ND_image node type, SdfType)
_PREVIEW_TO_OPENPBR_TEX = [
    # (preview_input,   openpbr_input,         nd_image_type,       output_name)
    ('diffuseColor',   'base_color',           'ND_image_color3',   'out'),
    ('roughness',      'specular_roughness',   'ND_image_float',    'out'),
    ('metallic',       'base_metalness',       'ND_image_float',    'out'),
    ('emissiveColor',  'emission_color',       'ND_image_color3',   'out'),
    # Normal handled separately below
]


def _inject_materialx_textures(usd_path):
    """
    Post-process the exported USD to inject ND_image_* nodes into OpenPBR
    subgraphs where Blender left literal values instead of texture connections.

    For each material that has both an ND_open_pbr_surface_surfaceshader and a
    UsdPreviewSurface shader, we inspect the UsdPreviewSurface subgraph for
    UsdUVTexture connections.  For each texture found on a mapped input, if the
    corresponding OpenPBR input has no connection, we create:
      - ND_texcoord_vector2  (shared per material, created once)
      - ND_place2d_vector2   (if the UsdTransform2d has non-identity values)
      - ND_image_<type>      (one per texture)
    and wire them into the OpenPBR terminal.

    The stage is saved in-place.  This makes the OpenPBR subgraph self-contained
    so OSL/MaterialX evaluation does not need to fall back to UsdPreviewSurface.
    """
    try:
        from pxr import Usd, UsdShade, Sdf, Gf
    except ImportError:
        print("  [mtlx inject] pxr not available — skipping.")
        return

    print()
    print("  Injecting MaterialX texture nodes into OpenPBR subgraphs...")

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("  [mtlx inject] Could not open stage — skipping.")
        return

    OPEN_PBR_ID  = "ND_open_pbr_surface_surfaceshader"
    PREVIEW_ID   = "UsdPreviewSurface"
    UVTEX_ID     = "UsdUVTexture"
    TRANSFORM_ID = "UsdTransform2d"
    injected_total = 0

    for mat_prim in stage.Traverse():
        if not mat_prim.IsA(UsdShade.Material):
            continue

        # --- find OpenPBR terminal ---
        openpbr = None
        for desc in Usd.PrimRange(mat_prim):
            if not desc.IsA(UsdShade.Shader):
                continue
            id_attr = desc.GetAttribute("info:id")
            if id_attr and id_attr.IsValid():
                tok = id_attr.Get()
                sid = tok.GetString() if hasattr(tok, 'GetString') else str(tok)
                if sid == OPEN_PBR_ID:
                    openpbr = UsdShade.Shader(desc)
                    break
        if openpbr is None:
            continue

        # --- find UsdPreviewSurface shader ---
        preview = None
        for desc in Usd.PrimRange(mat_prim):
            if not desc.IsA(UsdShade.Shader):
                continue
            id_attr = desc.GetAttribute("info:id")
            if id_attr and id_attr.IsValid():
                tok = id_attr.Get()
                sid = tok.GetString() if hasattr(tok, 'GetString') else str(tok)
                if sid == PREVIEW_ID:
                    preview = UsdShade.Shader(desc)
                    break
        if preview is None:
            continue

        mat_path = mat_prim.GetPath()
        injected = 0

        # Shared ND_texcoord_vector2 prim (one per material, created on demand)
        texcoord_prim_path = mat_path.AppendChild("mtlx_texcoord")
        texcoord_output    = None   # filled lazily

        def _ensure_texcoord():
            nonlocal texcoord_output
            if texcoord_output is not None:
                return texcoord_output
            tc = UsdShade.Shader.Define(stage, texcoord_prim_path)
            tc.CreateIdAttr("ND_texcoord_vector2")
            tc.CreateInput("index", Sdf.ValueTypeNames.Int).Set(0)
            texcoord_output = tc.CreateOutput("out", Sdf.ValueTypeNames.Float2)
            return texcoord_output

        def _get_asset_path_str(prim, input_name):
            """Return resolved (or raw) file path from a shader prim's asset input."""
            attr = prim.GetAttribute(f"inputs:{input_name}")
            if not attr:
                return None
            val = attr.Get()
            if val is None:
                return None
            resolved = val.GetResolvedPath() if hasattr(val, 'GetResolvedPath') else ""
            asset    = val.GetAssetPath()    if hasattr(val, 'GetAssetPath')    else str(val)
            return resolved if resolved else asset if asset else None

        # ---- colour / scalar inputs ----
        for preview_inp_name, openpbr_inp_name, nd_type, out_name in _PREVIEW_TO_OPENPBR_TEX:
            # Skip if OpenPBR input is already connected (to ND_image or NodeGraph)
            openpbr_inp = openpbr.GetInput(openpbr_inp_name)
            if openpbr_inp and openpbr_inp.HasConnectedSource():
                continue

            preview_inp = preview.GetInput(preview_inp_name)
            if not preview_inp or not preview_inp.HasConnectedSource():
                continue

            srcs = preview_inp.GetConnectedSources()
            if not srcs:
                continue
            uvtex_prim, _ = _follow_connection(srcs[0])
            if not uvtex_prim or not uvtex_prim.IsValid():
                continue

            uvtex = UsdShade.Shader(uvtex_prim)
            uv_id_attr = uvtex_prim.GetAttribute("info:id")
            uv_id = ""
            if uv_id_attr and uv_id_attr.IsValid():
                tok = uv_id_attr.Get()
                uv_id = tok.GetString() if hasattr(tok, 'GetString') else str(tok)
            if uv_id != UVTEX_ID:
                continue

            # Read file path
            file_path = _get_asset_path_str(uvtex_prim, "file")
            if not file_path:
                continue

            # Check for UsdTransform2d on the st input
            place2d_output = None
            st_inp = uvtex.GetInput("st")
            if st_inp and st_inp.HasConnectedSource():
                st_srcs = st_inp.GetConnectedSources()
                if st_srcs:
                    xf_prim, _ = _follow_connection(st_srcs[0])
                    xf = UsdShade.Shader(xf_prim)
                    xf_id_attr = xf_prim.GetAttribute("info:id")
                    xf_id = ""
                    if xf_id_attr and xf_id_attr.IsValid():
                        tok = xf_id_attr.Get()
                        xf_id = tok.GetString() if hasattr(tok, 'GetString') else str(tok)
                    if xf_id == TRANSFORM_ID:
                        # Read scale / translation / rotation
                        sc_inp  = xf.GetInput("scale")
                        tr_inp  = xf.GetInput("translation")
                        rot_inp = xf.GetInput("rotation")
                        sc  = sc_inp.Get()  if sc_inp  else Gf.Vec2f(1, 1)
                        tr  = tr_inp.Get()  if tr_inp  else Gf.Vec2f(0, 0)
                        rot = rot_inp.Get() if rot_inp else 0.0
                        if sc  is None: sc  = Gf.Vec2f(1, 1)
                        if tr  is None: tr  = Gf.Vec2f(0, 0)
                        if rot is None: rot = 0.0

                        is_identity = (abs(sc[0]-1)<1e-5 and abs(sc[1]-1)<1e-5
                                       and abs(tr[0])<1e-5 and abs(tr[1])<1e-5
                                       and abs(rot)<1e-5)
                        if not is_identity:
                            # Create ND_place2d_vector2
                            p2d_path = mat_path.AppendChild(
                                f"mtlx_place2d_{preview_inp_name}")
                            p2d = UsdShade.Shader.Define(stage, p2d_path)
                            p2d.CreateIdAttr("ND_place2d_vector2")
                            p2d.CreateInput("scale",  Sdf.ValueTypeNames.Float2).Set(sc)
                            p2d.CreateInput("offset", Sdf.ValueTypeNames.Float2).Set(tr)
                            p2d.CreateInput("rotate", Sdf.ValueTypeNames.Float).Set(rot)
                            tc_out = _ensure_texcoord()
                            p2d.CreateInput("texcoord",
                                            Sdf.ValueTypeNames.Float2).ConnectToSource(tc_out)
                            place2d_output = p2d.CreateOutput("out", Sdf.ValueTypeNames.Float2)

            # Create ND_image_* node
            img_prim_path = mat_path.AppendChild(f"mtlx_image_{preview_inp_name}")
            img_shader = UsdShade.Shader.Define(stage, img_prim_path)
            img_shader.CreateIdAttr(nd_type)
            img_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(file_path))

            # Determine output SdfType from nd_type
            if nd_type == 'ND_image_color3':
                out_type = Sdf.ValueTypeNames.Color3f
            elif nd_type == 'ND_image_float':
                out_type = Sdf.ValueTypeNames.Float
            else:
                out_type = Sdf.ValueTypeNames.Float3

            # Wire UV
            uv_src = place2d_output if place2d_output else _ensure_texcoord()
            img_shader.CreateInput("texcoord", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_src)

            img_out = img_shader.CreateOutput("out", out_type)

            # Connect to OpenPBR input
            if openpbr_inp is None:
                openpbr_inp = openpbr.CreateInput(openpbr_inp_name, out_type)
            openpbr_inp.ConnectToSource(img_out)
            injected += 1

        # ---- normal map ----
        norm_inp_preview = preview.GetInput("normal")
        norm_inp_openpbr = openpbr.GetInput("geometry_normal")
        if (norm_inp_preview and norm_inp_preview.HasConnectedSource()
                and (norm_inp_openpbr is None or not norm_inp_openpbr.HasConnectedSource())):
            srcs = norm_inp_preview.GetConnectedSources()
            if srcs:
                uvtex_prim, _ = _follow_connection(srcs[0])
                uvtex = UsdShade.Shader(uvtex_prim)
                file_path = _get_asset_path_str(uvtex_prim, "file")
                if file_path:
                        img_prim_path = mat_path.AppendChild("mtlx_image_normal")
                        img_shader = UsdShade.Shader.Define(stage, img_prim_path)
                        img_shader.CreateIdAttr("ND_image_vector3")
                        img_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                            Sdf.AssetPath(file_path))
                        tc_out = _ensure_texcoord()
                        img_shader.CreateInput("texcoord",
                                               Sdf.ValueTypeNames.Float2).ConnectToSource(tc_out)
                        img_out = img_shader.CreateOutput("out", Sdf.ValueTypeNames.Float3)

                        nm_path = mat_path.AppendChild("mtlx_normalmap")
                        nm = UsdShade.Shader.Define(stage, nm_path)
                        nm.CreateIdAttr("ND_normalmap_vector3")
                        nm.CreateInput("in", Sdf.ValueTypeNames.Float3).ConnectToSource(img_out)
                        # Read scale from UsdUVTexture scale/bias if present
                        sc_attr = uvtex_prim.GetAttribute("inputs:scale")
                        sc_val = sc_attr.Get() if sc_attr else None
                        nm_scale = float(sc_val[0]) if sc_val is not None else 2.0
                        nm.CreateInput("scale", Sdf.ValueTypeNames.Float).Set(nm_scale)
                        nm_out = nm.CreateOutput("out", Sdf.ValueTypeNames.Float3)

                        if norm_inp_openpbr is None:
                            norm_inp_openpbr = openpbr.CreateInput(
                                "geometry_normal", Sdf.ValueTypeNames.Float3)
                        norm_inp_openpbr.ConnectToSource(nm_out)
                        injected += 1

        if injected:
            print(f"  [mtlx inject] {mat_prim.GetPath()}: injected {injected} texture node(s)")
            injected_total += injected

    if injected_total == 0:
        print("  [mtlx inject] No missing texture connections found.")
    else:
        stage.Save()
        print(f"  [mtlx inject] Saved stage with {injected_total} new node(s).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# MaterialX JSON sidecar extractor
# ---------------------------------------------------------------------------
def _get_shader_input_value(inp):
    """Return a JSON-serialisable value for a UsdShadeInput, or None."""
    try:
        val = inp.Get()
    except Exception:
        return None
    if val is None:
        return None
    # GfVec* → list
    if hasattr(val, '__len__') and not isinstance(val, str):
        try:
            return list(val)
        except Exception:
            pass
    # SdfAssetPath → string
    if hasattr(val, 'resolvedPath'):
        return val.resolvedPath or val.path
    if hasattr(val, 'path'):
        return val.path
    # TfToken → string
    if hasattr(val, 'GetString'):
        return val.GetString()
    # scalars
    try:
        return float(val)
    except (TypeError, ValueError):
        pass
    return str(val)


def _follow_connection(src_info):
    """Extract (prim, source_output_name) from a UsdShadeConnectionSourceInfo.

    Blender's bundled USD wraps each element in an extra list, so
    GetConnectedSources() returns [[ConnectionSourceInfo], ...] rather than
    [ConnectionSourceInfo, ...].  Unwrap one level if needed.
    """
    # Unwrap list-of-list if present
    if isinstance(src_info, list):
        if not src_info:
            return None, None
        src_info = src_info[0]
    # src_info is now a ConnectionSourceInfo object
    try:
        prim = src_info.source.GetPrim()
        out_name = (src_info.sourceName.GetString()
                    if hasattr(src_info.sourceName, 'GetString')
                    else str(src_info.sourceName))
        return prim, out_name
    except Exception:
        return None, None


def _resolve_through_nodegraph(prim, output_name):
    """Transparently walk through NodeGraph prims to reach the actual shader
    that drives the named output.  NodeGraphs are purely structural in
    MaterialX and carry no per-node parameters we care about.

    Returns the resolved Shader prim (may be prim itself if it is already a
    Shader, or if the NodeGraph cannot be resolved).
    """
    from pxr import UsdShade
    depth = 0
    while prim and prim.IsValid() and prim.IsA(UsdShade.NodeGraph) and depth < 8:
        depth += 1
        ng = UsdShade.NodeGraph(prim)
        # Find the named output on the NodeGraph (strip namespace prefix if any)
        bare_name = output_name.split(":")[-1] if output_name else ""
        out = ng.GetOutput(bare_name) if bare_name else None
        if out is None:
            # Fall back: take the first output
            outs = ng.GetOutputs()
            out = outs[0] if outs else None
        if out is None or not out.HasConnectedSource():
            break
        src_infos = out.GetConnectedSources()
        if not src_infos:
            break
        inner_prim, inner_out = _follow_connection(src_infos[0])
        if not inner_prim or not inner_prim.IsValid():
            break
        prim, output_name = inner_prim, inner_out
    return prim


def _collect_node(shader_prim, visited):
    """Recursively collect a shader node and all nodes connected to its inputs.
    NodeGraph prims are resolved transparently — the JSON only contains actual
    Shader nodes with concrete parameters."""
    from pxr import UsdShade
    path_str = str(shader_prim.GetPath())
    if path_str in visited:
        return visited[path_str]

    node_id = ""
    id_attr = shader_prim.GetAttribute("info:id")
    if id_attr and id_attr.IsValid():
        tok = id_attr.Get()
        node_id = tok.GetString() if hasattr(tok, 'GetString') else str(tok)

    node_data = {
        "path": path_str,
        "id": node_id,
        "inputs": {},
        "connected_nodes": {},
    }
    visited[path_str] = node_data  # register early to break cycles

    sh = UsdShade.Shader(shader_prim)
    for inp in sh.GetInputs():
        name = inp.GetBaseName()
        if not inp.HasConnectedSource():
            val = _get_shader_input_value(inp)
            if val is not None:
                node_data["inputs"][name] = val
            continue
        src_info_list = inp.GetConnectedSources()
        if not src_info_list:
            continue
        connected_prim, source_out = _follow_connection(src_info_list[0])
        if not connected_prim or not connected_prim.IsValid():
            continue
        # Pass through any NodeGraph wrappers to reach the concrete shader
        connected_prim = _resolve_through_nodegraph(connected_prim, source_out)
        if connected_prim and connected_prim.IsValid():
            child_node = _collect_node(connected_prim, visited)
            node_data["connected_nodes"][name] = child_node["path"]

    return node_data


def _extract_materialx_sidecar(usd_path):
    """
    Walk all materials in the exported USD looking for ND_open_pbr_surface_surfaceshader
    nodes (Blender 4.x/5.x MaterialX terminal).  For each material found, traverse the
    full node graph and write the result to <usd_path>.materials.json alongside the USD.
    """
    import json
    try:
        from pxr import Usd, UsdShade, Sdf, Tf
    except ImportError:
        print("  [MaterialX sidecar] pxr not available in this Python — skipping.")
        return

    print()
    print("  Extracting MaterialX node graphs...")

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("  [MaterialX sidecar] Could not open stage — skipping.")
        return

    materials_data = {}   # material_path → node dict list
    visited_nodes = {}    # shared across all materials to avoid duplicate work

    OPEN_PBR_ID = "ND_open_pbr_surface_surfaceshader"

    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        mat = UsdShade.Material(prim)
        mat_path = str(prim.GetPath())

        # Find any ND_open_pbr_surface_surfaceshader descendant
        openpbr_shader = None
        for desc in Usd.PrimRange(prim):
            if not desc.IsA(UsdShade.Shader):
                continue
            id_attr = desc.GetAttribute("info:id")
            if not (id_attr and id_attr.IsValid()):
                continue
            tok = id_attr.Get()
            node_id = tok.GetString() if hasattr(tok, 'GetString') else str(tok)
            if node_id == OPEN_PBR_ID:
                openpbr_shader = desc
                break

        if openpbr_shader is None:
            continue

        root_node = _collect_node(openpbr_shader, visited_nodes)

        # Collect all nodes reachable from this material's root
        # (visited_nodes already has them; gather the unique subtree)
        def _gather_subtree(node, acc):
            p = node["path"]
            if p in acc:
                return
            acc[p] = node
            for child_path in node["connected_nodes"].values():
                if child_path in visited_nodes:
                    _gather_subtree(visited_nodes[child_path], acc)

        subtree = {}
        _gather_subtree(root_node, subtree)

        materials_data[mat_path] = {
            "root": root_node["path"],
            "nodes": subtree,
        }

    if not materials_data:
        print("  [MaterialX sidecar] No ND_open_pbr_surface_surfaceshader materials found.")
        return

    sidecar_path = usd_path + ".materials.json"
    with open(sidecar_path, "w") as f:
        json.dump(materials_data, f, indent=2)

    print(f"  [MaterialX sidecar] Wrote {len(materials_data)} material(s) → {sidecar_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
