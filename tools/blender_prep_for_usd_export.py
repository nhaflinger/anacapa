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
    vl_objects = set(bpy.context.view_layer.objects)
    candidates = [(obj.name, obj.type) for obj in bpy.context.scene.objects]

    for obj_name, obj_type in candidates:
        if obj_type in NON_MESH_CONVERTIBLE:
            # Look up the live object by name; skip if already converted/removed
            obj = bpy.context.scene.objects.get(obj_name)
            if obj is None:
                log(f"  '{obj_name}' already removed (converted as part of another object).")
                continue
            if obj not in vl_objects:
                log(f"  Skipping '{obj_name}' — not in active View Layer.")
                continue
            try:
                # For curves/NURBS, promote preview resolution to render resolution
                # before converting so the mesh gets the intended tessellation detail.
                if obj_type in ('CURVE', 'SURFACE') and obj.data is not None:
                    curve = obj.data
                    # resolution_u controls preview; render_resolution_u = 0 means
                    # "use resolution_u". Force them equal so convert() uses render detail.
                    if hasattr(curve, 'render_resolution_u') and curve.render_resolution_u > 0:
                        curve.resolution_u = curve.render_resolution_u
                    for spline in getattr(curve, 'splines', []):
                        if hasattr(spline, 'render_resolution_u') and spline.render_resolution_u > 0:
                            spline.resolution_u = spline.render_resolution_u
                    # Enforce a minimum tessellation quality.  A resolution_u of 2
                    # (common when artists set a low preview value) produces visibly
                    # faceted or straight-line geometry after mesh conversion.
                    MIN_CURVE_RES = 24
                    if curve.resolution_u < MIN_CURVE_RES:
                        log(f"  Bumping '{obj_name}' resolution_u: {curve.resolution_u} → {MIN_CURVE_RES}")
                        curve.resolution_u = MIN_CURVE_RES
                    for spline in getattr(curve, 'splines', []):
                        if hasattr(spline, 'resolution_u') and spline.resolution_u < MIN_CURVE_RES:
                            spline.resolution_u = MIN_CURVE_RES
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

    vl_objects = set(bpy.context.view_layer.objects)

    for obj in list(bpy.context.scene.objects):
        if obj.type != 'MESH':
            continue
        if not obj.modifiers:
            continue
        # Objects not in the active View Layer cannot be selected — skip them.
        # This happens with objects inside disabled/excluded collections.
        if obj not in vl_objects:
            log(f"  Skipping '{obj.name}' — not in active View Layer.")
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
    """Apply scale (and optionally rotation/location) to all mesh objects.

    Selects all eligible objects at once and calls transform_apply once —
    much faster than per-object calls when the scene has thousands of objects.
    """
    vl_objects = set(bpy.context.view_layer.objects)

    # Make mesh data single-user where needed, collect eligible objects
    eligible = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        if obj not in vl_objects:
            continue
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        eligible.append(obj)

    if not eligible:
        log("No mesh objects to apply transforms on.")
        return 0

    # Batch-select all eligible objects and apply in one call
    deselect_all()
    for obj in eligible:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = eligible[0]
    bpy.ops.object.transform_apply(
        location=APPLY_LOCATION,
        rotation=APPLY_ROTATION,
        scale=True,
    )

    count = len(eligible)
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

# Nodes that are NOT natively exported to USD but ARE handled by the
# MaterialX export step — suppress bake-step warnings for these.
_MATERIALX_HANDLED_TYPES = {
    # Color
    'VALTORGB', 'INVERT', 'MIX_RGB', 'MIX', 'HUE_SAT', 'BRIGHTCONTRAST',
    'GAMMA', 'CURVE_RGB', 'RGBTOBW', 'BLACKBODY', 'WAVELENGTH', 'LIGHT_FALLOFF',
    # Math / Converter
    'MATH', 'CLAMP', 'MAP_RANGE', 'FLOAT_CURVE',
    # Texture
    'TEX_NOISE', 'TEX_WAVE', 'TEX_VORONOI', 'TEX_MUSGRAVE', 'TEX_GRADIENT',
    'TEX_CHECKER', 'TEX_BRICK', 'TEX_ENVIRONMENT', 'TEX_MAGIC', 'TEX_GABOR',
    'TEX_WHITE_NOISE', 'TEX_SKY', 'TEX_IES',
    # Vector
    'BUMP', 'NORMAL_MAP', 'DISPLACEMENT', 'VECTOR_DISPLACEMENT',
    'VECTOR_MATH', 'VECTOR_ROTATE', 'VECTOR_TRANSFORM', 'CURVE_VEC', 'NORMAL',
    'TANGENT',
    # Shader
    'BSDF_DIFFUSE', 'BSDF_GLOSSY', 'BSDF_METALLIC', 'BSDF_GLASS',
    'BSDF_REFRACTION', 'BSDF_TRANSPARENT', 'BSDF_TRANSLUCENT', 'BSDF_TOON',
    'BSDF_SHEEN', 'BSDF_HAIR', 'BSDF_HAIR_PRINCIPLED', 'BSDF_RAY_PORTAL',
    'EMISSION', 'BACKGROUND', 'SUBSURFACE_SCATTERING', 'HOLDOUT',
    'MIX_SHADER', 'ADD_SHADER', 'EEVEE_SPECULAR', 'SHADER_TO_RGB',
    'VOLUME_PRINCIPLED', 'VOLUME_ABSORPTION', 'VOLUME_SCATTER', 'VOLUME_COEFFICIENTS',
    # Input
    'RGB', 'VALUE', 'VERTEX_COLOR', 'ATTRIBUTE', 'FRESNEL', 'LAYER_WEIGHT',
    'AMBIENT_OCCLUSION', 'NEW_GEOMETRY', 'GEOMETRY', 'OBJECT_INFO',
    'CAMERA', 'LIGHT_PATH', 'HAIR_INFO', 'PARTICLE_INFO', 'POINT_INFO',
    'VOLUME_INFO', 'WIREFRAME', 'BEVEL', 'RAYCAST', 'UV_ALONG_STROKE',
    # Separate / Combine
    'SEPXYZ', 'SEPRGB', 'SEPARATE_XYZ', 'SEPARATE_COLOR',
    'COMBXYZ', 'COMBRGB', 'COMBINE_XYZ', 'COMBINE_COLOR',
    # Structural
    'GROUP', 'REROUTE', 'FRAME',
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

            # Log only nodes not handled by either USD export or MaterialX export
            if src_node.type not in _USD_PASSTHROUGH_TYPES \
                    and src_node.type not in _MATERIALX_HANDLED_TYPES:
                log(f"  Material '{mat.name}', socket '{sock.name}': "
                    f"unsupported node '{src_node.name}' (type={src_node.type}) "
                    f"— not handled by USD or MaterialX export")

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
                    # Procedural-only chains (no image source) are handled by
                    # the MaterialX export step — no need to warn here.
                    pass
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
    parser.add_argument("--sky-texture", default="",
                        help="Path to equirectangular sky/environment image to use as "
                             "the DomeLight texture (overrides whatever Blender exports). "
                             "Use this when the scene has a sky mesh instead of a World HDRI.")
    parser.add_argument("--sun-intensity", type=float, default=0.0,
                        help="Add a DistantLight sun to the exported USD with this intensity. "
                             "0 (default) = no sun added.")
    parser.add_argument("--sun-angle", type=float, default=0.53,
                        help="Angular diameter of the sun disk in degrees (default 0.53 = real sun).")
    parser.add_argument("--sun-color", default="1.0,0.95,0.8",
                        help="Sun color as R,G,B floats in [0,1] (default '1.0,0.95,0.8').")
    parser.add_argument("--sun-elevation", type=float, default=45.0,
                        help="Sun elevation above the horizon in degrees (default 45).")
    parser.add_argument("--sun-azimuth", type=float, default=135.0,
                        help="Sun azimuth in degrees, 0=+Z, 90=+X (default 135).")
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
    # Patch DomeLight texture: if Blender exported a near-black solid-color EXR
    # as the DomeLight, replace it with --sky-texture (if given) or search the
    # World shader for a sky/environment image.
    # -----------------------------------------------------------------------
    if os.path.exists(out_path):
        _patch_dome_light_texture(out_path, out_dir,
                                  explicit_sky=args.sky_texture)

    # -----------------------------------------------------------------------
    # Add sun DistantLight if requested
    # -----------------------------------------------------------------------
    if os.path.exists(out_path) and args.sun_intensity > 0.0:
        try:
            sun_color = tuple(float(x) for x in args.sun_color.split(","))
            if len(sun_color) != 3:
                raise ValueError("--sun-color must be three comma-separated floats")
        except Exception as e:
            print(f"  [sun] Bad --sun-color value '{args.sun_color}': {e} — skipping sun.")
            sun_color = None
        if sun_color is not None:
            _add_sun_light(out_path,
                           intensity=args.sun_intensity,
                           angle=args.sun_angle,
                           color_rgb=sun_color,
                           elevation_deg=args.sun_elevation,
                           azimuth_deg=args.sun_azimuth)

    # -----------------------------------------------------------------------
    # Inject ND_image_* nodes into OpenPBR subgraphs that are missing textures
    # (Blender emits OpenPBR terminals with literal values; textures only exist
    #  in the UsdPreviewSurface subgraph — we bridge them here so the graph is
    #  self-contained before OSL/MaterialX evaluation.)
    # -----------------------------------------------------------------------
    if os.path.exists(out_path):
        _inject_materialx_textures(out_path)

    # -----------------------------------------------------------------------
    # Export full MaterialX node graphs (.mtlx) — one file per material,
    # preserving procedural nodes (Noise, ColorRamp, Math, etc.)
    # -----------------------------------------------------------------------
    mtlx_dir = os.path.join(os.path.dirname(out_path), "materials")
    _export_materialx_graphs(mtlx_dir)

    # -----------------------------------------------------------------------
    # Extract MaterialX node graphs → JSON sidecar
    # -----------------------------------------------------------------------
    if os.path.exists(out_path):
        _extract_materialx_sidecar(out_path)


# ---------------------------------------------------------------------------
# MaterialX node graph exporter  (Blender → .mtlx)
# ---------------------------------------------------------------------------
#
# Translates Blender shader node trees to MaterialX documents and writes one
# .mtlx file per material into <out_dir>/<material_name>.mtlx.
#
# Supported Blender node types and their MaterialX equivalents:
#   BSDF_PRINCIPLED      → ND_open_pbr_surface_surfaceshader  (terminal)
#   TEX_IMAGE            → ND_image_color3 / ND_image_float
#   TEX_NOISE            → ND_fractal3d_float (+ ND_mix for detail blend)
#   TEX_WAVE             → ND_fractal3d_float approximation
#   TEX_VORONOI          → ND_cellnoise3d_float
#   TEX_MUSGRAVE         → ND_fractal3d_float
#   TEX_GRADIENT         → ND_texcoord + ND_dotproduct approximation
#   TEX_CHECKER          → ND_checkerboard_color3
#   TEX_BRICK            → ND_checkerboard_color3 (approximation)
#   VALTORGB             → ND_ramp_color3 / custom ramp baked to texture
#   RGBTOBW              → ND_luminance_float
#   MIX_RGB / MIX        → ND_mix / ND_multiply / ND_add etc.
#   MATH                 → ND_multiply_float, ND_add_float, ND_clamp_float, etc.
#   INVERT               → ND_subtract_float / ND_subtract_color3
#   HUE_SAT              → ND_hsvtorgb / ND_rgbtohsv chain
#   BRIGHTCONTRAST       → ND_multiply_float + ND_add_float chain
#   NORMAL_MAP           → ND_normalmap_vector3
#   MAPPING              → ND_place2d_vector2 (UV) / ND_transformpoint (3D)
#   UVMAP                → ND_texcoord_vector2
#   SEPXYZ / COMBXYZ     → ND_extract / ND_combine3
#   SEPRGB / COMBRGB     → ND_extract / ND_combine3
#   GAMMA                → ND_power_float
#   CURVE_RGB            → ND_ramp_color3 (sampled)
#   OUTPUT_MATERIAL      → (root — not emitted)
#   GROUP                → inlined recursively
# ---------------------------------------------------------------------------

# Maps Blender blend_type → MaterialX node name suffix
_MX_BLEND_TYPE = {
    'MIX':        'mix',
    'ADD':        'add',
    'SUBTRACT':   'subtract',
    'MULTIPLY':   'multiply',
    'SCREEN':     'screen',
    'OVERLAY':    'overlay',
    'DODGE':      'dodge',
    'BURN':       'burn',
    'DARKEN':     'minimum',
    'LIGHTEN':    'maximum',
    'DIFFERENCE': 'difference',
    'DIVIDE':     'divide',
    'EXCLUSION':  'exclusion',
    'COLOR':      'mix',  # approximate
    'HUE':        'mix',
    'SATURATION': 'mix',
    'VALUE':      'mix',
    'LINEAR_LIGHT': 'mix',
    'SOFT_LIGHT': 'mix',
    'POWER':      'power',
}

# Maps Blender math operation → (mx_node_type, n_inputs)
_MX_MATH_OP = {
    'ADD':           ('add',      2),
    'SUBTRACT':      ('subtract', 2),
    'MULTIPLY':      ('multiply', 2),
    'DIVIDE':        ('divide',   2),
    'POWER':         ('power',    2),
    'SQRT':          ('sqrt',     1),
    'ABSOLUTE':      ('absval',   1),
    'MINIMUM':       ('min',      2),
    'MAXIMUM':       ('max',      2),
    'ROUND':         ('round',    1),
    'FLOOR':         ('floor',    1),
    'CEIL':          ('ceil',     1),
    'FRACT':         ('fract',    1),
    'MODULO':        ('modulo',   2),
    'SINE':          ('sin',      1),
    'COSINE':        ('cos',      1),
    'TANGENT':       ('tan',      1),
    'ARCSINE':       ('asin',     1),
    'ARCCOSINE':     ('acos',     1),
    'ARCTANGENT':    ('atan2',    1),
    'ARCTAN2':       ('atan2',    2),
    'LOGARITHM':     ('ln',       1),
    'EXPONENT':      ('exp',      1),
    'SIGN':          ('sign',     1),
    'CLAMP':         ('clamp',    1),  # uses separate min/max inputs
    'SNAP':          ('floor',    1),  # approximate
    'WRAP':          ('modulo',   2),  # approximate
    'PINGPONG':      ('modulo',   2),  # approximate
    'SMOOTH_MIN':    ('min',      2),  # approximate
    'SMOOTH_MAX':    ('max',      2),  # approximate
    'COMPARE':       ('min',      2),  # approximate — step not in stdlib
    'MULTIPLY_ADD':  ('multiply', 2),  # only captures multiply part; add wired separately
    'TRUNC':         ('floor',    1),
    'GREATER_THAN':  ('min',      2),  # approximate
    'LESS_THAN':     ('min',      2),  # approximate
}


class _MtlxBuilder:
    """
    Converts one Blender material node tree to a MaterialX document.

    Usage:
        builder = _MtlxBuilder(mat, out_dir, mx_lib_path)
        builder.build()   # writes <out_dir>/<mat.name>.mtlx
    """

    def __init__(self, mat, out_dir: str, mx_lib_path: str):
        self.mat          = mat
        self.out_dir      = out_dir
        self.mx_lib_path  = mx_lib_path
        self._doc         = None
        self._ng          = None   # NodeGraph element
        self._mat_node    = None   # Material element
        self._node_cache: dict = {}  # blender node.name → mx Output
        self._counter     = 0

    def _uid(self, prefix='n'):
        self._counter += 1
        return f"{prefix}{self._counter}"

    def build(self):
        import MaterialX as mx
        import os

        mat = self.mat
        nodes = mat.node_tree.nodes

        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is None:
            return  # nothing to export

        doc = mx.createDocument()
        # Do NOT import the stdlib into the document — stdlib definitions are
        # implicit and loaded by the reader from its own search path.  Importing
        # them here causes every .mtlx file to contain thousands of duplicate
        # node definitions.

        self._doc = doc
        safe_name = mx.createValidName(mat.name)

        # NodeGraph wraps the full shading network.
        # Terminal shader (open_pbr_surface) lives inside the NodeGraph.
        # An output port on the NodeGraph bridges it to the document scope.
        ng = doc.addNodeGraph(f"NG_{safe_name}")
        self._ng = ng

        # Translate the BSDF node (terminal — placed inside the NodeGraph)
        surface_out = self._translate_node(bsdf)
        if surface_out is None:
            return

        # Expose the surface shader through the NodeGraph's output port.
        # Only set 'output' attribute for multi-output nodes (same rule as _connect).
        ng_out = ng.addOutput('surface', 'surfaceshader')
        src_node = surface_out.getParent()
        ng_out.setNodeName(src_node.getName())
        if len(src_node.getOutputs()) > 1:
            ng_out.setAttribute('output', surface_out.getName())

        # surfacematerial at document scope wires to the NodeGraph output
        mat_node = doc.addNode('surfacematerial', f"M_{safe_name}", 'material')
        surf_inp = mat_node.addInput('surfaceshader', 'surfaceshader')
        surf_inp.setAttribute('nodegraph', ng.getName())
        surf_inp.setAttribute('output', 'surface')

        # Write file
        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(self.out_dir, safe_name + '.mtlx')
        mx.writeToXmlFile(doc, out_path)
        return out_path

    # ------------------------------------------------------------------ #
    #  Main dispatch                                                        #
    # ------------------------------------------------------------------ #

    def _translate_node(self, bl_node):
        """Translate a Blender node and return its primary mx Output.
        Results are cached so nodes used by multiple downstream sockets
        are not duplicated."""
        import MaterialX as mx
        key = bl_node.name
        if key in self._node_cache:
            return self._node_cache[key]

        out = None
        t = bl_node.type

        # ── Shader / BSDF ──────────────────────────────────────────────
        if t == 'BSDF_PRINCIPLED':
            out = self._tx_principled(bl_node)
        elif t in ('BSDF_DIFFUSE',):
            out = self._tx_bsdf_diffuse(bl_node)
        elif t in ('BSDF_GLOSSY', 'BSDF_METALLIC'):
            out = self._tx_bsdf_glossy(bl_node)
        elif t == 'BSDF_GLASS':
            out = self._tx_bsdf_glass(bl_node)
        elif t == 'BSDF_REFRACTION':
            out = self._tx_bsdf_refraction(bl_node)
        elif t in ('BSDF_TRANSPARENT', 'BSDF_TRANSLUCENT'):
            out = self._tx_bsdf_transparent(bl_node)
        elif t in ('BSDF_TOON', 'BSDF_SHEEN'):
            out = self._tx_bsdf_diffuse(bl_node)    # rough approximation
        elif t in ('BSDF_HAIR', 'BSDF_HAIR_PRINCIPLED'):
            out = self._tx_bsdf_glossy(bl_node)     # rough approximation
        elif t == 'BSDF_RAY_PORTAL':
            out = self._tx_bsdf_transparent(bl_node)
        elif t == 'EMISSION':
            out = self._tx_emission(bl_node)
        elif t == 'BACKGROUND':
            out = self._tx_emission(bl_node)         # same structure
        elif t == 'SUBSURFACE_SCATTERING':
            out = self._tx_sss(bl_node)
        elif t == 'HOLDOUT':
            out = self._const_color3(0.0, 0.0, 0.0)
        elif t == 'MIX_SHADER':
            out = self._tx_mix_shader(bl_node)
        elif t == 'ADD_SHADER':
            out = self._tx_add_shader(bl_node)
        elif t in ('EEVEE_SPECULAR', 'SHADER_TO_RGB'):
            # Eevee-only — fall back to diffuse approximation
            out = self._tx_bsdf_diffuse(bl_node)
        # ── Texture ────────────────────────────────────────────────────
        elif t == 'TEX_IMAGE':
            out = self._tx_image(bl_node)
        elif t == 'TEX_NOISE':
            out = self._tx_noise(bl_node)
        elif t == 'TEX_WAVE':
            out = self._tx_wave(bl_node)
        elif t == 'TEX_VORONOI':
            out = self._tx_voronoi(bl_node)
        elif t in ('TEX_MUSGRAVE',):
            out = self._tx_musgrave(bl_node)
        elif t == 'TEX_GRADIENT':
            out = self._tx_gradient(bl_node)
        elif t in ('TEX_CHECKER',):
            out = self._tx_checker(bl_node)
        elif t == 'TEX_BRICK':
            out = self._tx_checker(bl_node)          # approximation
        elif t == 'TEX_ENVIRONMENT':
            out = self._tx_image(bl_node)            # treat as image lookup
        elif t in ('TEX_MAGIC', 'TEX_GABOR'):
            out = self._tx_noise(bl_node)            # approximate with noise
        elif t in ('TEX_WHITE_NOISE',):
            out = self._tx_whitenoise(bl_node)
        elif t in ('TEX_SKY', 'TEX_IES'):
            out = self._const_color3(0.5, 0.5, 0.5) # can't evaluate analytically
        # ── Color ──────────────────────────────────────────────────────
        elif t == 'VALTORGB':
            out = self._tx_color_ramp(bl_node)
        elif t == 'RGBTOBW':
            out = self._tx_rgbtobw(bl_node)
        elif t in ('MIX_RGB', 'MIX'):
            out = self._tx_mix(bl_node)
        elif t == 'INVERT':
            out = self._tx_invert(bl_node)
        elif t == 'HUE_SAT':
            out = self._tx_hue_sat(bl_node)
        elif t == 'BRIGHTCONTRAST':
            out = self._tx_bright_contrast(bl_node)
        elif t == 'GAMMA':
            out = self._tx_gamma(bl_node)
        elif t == 'CURVE_RGB':
            out = self._tx_curve_rgb(bl_node)
        elif t == 'BLACKBODY':
            out = self._tx_blackbody(bl_node)
        elif t == 'WAVELENGTH':
            out = self._const_color3(0.5, 0.5, 0.5) # spectral — not expressible
        elif t == 'LIGHT_FALLOFF':
            out = self._const_float(1.0)             # quadratic falloff = 1 at distance 1
        # ── Vector ─────────────────────────────────────────────────────
        elif t == 'NORMAL_MAP':
            out = self._tx_normal_map(bl_node)
        elif t == 'BUMP':
            out = self._tx_bump(bl_node)
        elif t == 'DISPLACEMENT':
            out = self._tx_displacement(bl_node)
        elif t == 'VECTOR_DISPLACEMENT':
            out = self._tx_vector_displacement(bl_node)
        elif t == 'MAPPING':
            out = self._tx_mapping(bl_node)
        elif t == 'VECTOR_MATH':
            out = self._tx_vector_math(bl_node)
        elif t == 'VECTOR_ROTATE':
            out = self._tx_vector_rotate(bl_node)
        elif t in ('VECTOR_TRANSFORM',):
            out = self._tx_vector_transform(bl_node)
        elif t == 'CURVE_VEC':
            out = self._tx_curve_vec(bl_node)
        elif t == 'NORMAL':
            out = self._tx_normal_const(bl_node)
        elif t == 'TANGENT':
            out = self._tx_tangent(bl_node)
        # ── Math / Converter ───────────────────────────────────────────
        elif t == 'MATH':
            out = self._tx_math(bl_node)
        elif t == 'CLAMP':
            out = self._tx_clamp(bl_node)
        elif t == 'MAP_RANGE':
            out = self._tx_map_range(bl_node)
        elif t == 'FLOAT_CURVE':
            out = self._tx_curve_rgb(bl_node)        # same approximation
        # ── Input ──────────────────────────────────────────────────────
        elif t in ('UVMAP', 'TEX_COORD'):
            out = self._tx_texcoord(bl_node)
        elif t == 'RGB':
            out = self._tx_rgb_input(bl_node)
        elif t == 'VALUE':
            out = self._tx_value_input(bl_node)
        elif t == 'VERTEX_COLOR':
            out = self._tx_vertex_color(bl_node)
        elif t == 'ATTRIBUTE':
            out = self._tx_attribute(bl_node)
        elif t == 'FRESNEL':
            out = self._tx_fresnel(bl_node)
        elif t == 'LAYER_WEIGHT':
            out = self._tx_layer_weight(bl_node)
        elif t in ('AMBIENT_OCCLUSION',):
            out = self._const_float(1.0)             # baked AO unsupported at gen time
        elif t in ('NEW_GEOMETRY', 'GEOMETRY'):
            out = self._tx_geometry(bl_node)
        elif t == 'OBJECT_INFO':
            out = self._const_color3(0.5, 0.5, 0.5) # object-level, not shadeable
        elif t in ('CAMERA', 'LIGHT_PATH', 'HAIR_INFO', 'PARTICLE_INFO',
                   'POINT_INFO', 'VOLUME_INFO', 'WIREFRAME', 'BEVEL',
                   'RAYCAST', 'UV_ALONG_STROKE'):
            # Runtime-only render state — emit neutral constants
            out = self._const_float(0.0)
        # ── Separate / Combine ─────────────────────────────────────────
        elif t in ('SEPXYZ', 'SEPRGB', 'SEPARATE_XYZ', 'SEPARATE_COLOR'):
            out = self._tx_separate(bl_node)
        elif t in ('COMBXYZ', 'COMBRGB', 'COMBINE_XYZ', 'COMBINE_COLOR'):
            out = self._tx_combine(bl_node)
        # ── Volume ─────────────────────────────────────────────────────
        elif t in ('VOLUME_PRINCIPLED', 'VOLUME_ABSORPTION', 'VOLUME_SCATTER',
                   'VOLUME_COEFFICIENTS'):
            # Volumes need separate handling outside surface shaders
            out = self._const_color3(0.0, 0.0, 0.0)
        # ── Structural ─────────────────────────────────────────────────
        elif t == 'GROUP':
            out = self._tx_group(bl_node)
        elif t in ('OUTPUT_MATERIAL', 'OUTPUT_WORLD', 'OUTPUT_LIGHT',
                   'OUTPUT_AOV', 'OUTPUT_LINESTYLE', 'GROUP_OUTPUT'):
            out = None
        elif t == 'GROUP_INPUT':
            out = self._const_color3(0.5, 0.5, 0.5)
        elif t in ('FRAME', 'REROUTE'):
            if bl_node.inputs and bl_node.inputs[0].links:
                out = self._translate_node(bl_node.inputs[0].links[0].from_node)
            else:
                out = self._const_color3(0.5, 0.5, 0.5)
        elif t == 'SCRIPT':
            # OSL script node — contents opaque to us; emit grey and warn
            print(f"  [mtlx] WARNING: SCRIPT node '{bl_node.name}' "
                  f"cannot be translated — using grey constant")
            out = self._const_color3(0.5, 0.5, 0.5)
        else:
            # Truly unknown — warn
            print(f"  [mtlx] WARNING: unsupported node type '{t}' "
                  f"({bl_node.name}) — using grey constant")
            out = self._const_color3(0.5, 0.5, 0.5)

        self._node_cache[key] = out
        return out

    # ------------------------------------------------------------------ #
    #  Socket input helpers                                                 #
    # ------------------------------------------------------------------ #

    def _get_input_output(self, bl_socket):
        """Return mx Output connected to bl_socket, or None if unconnected."""
        if not bl_socket.links:
            return None
        from_node = bl_socket.links[0].from_node
        from_sock = bl_socket.links[0].from_socket

        # Separate XYZ / Separate Color: _tx_separate always creates index-0 extract,
        # but the downstream socket may request X(0), Y(1), or Z(2) / R(0), G(1), B(2).
        # Create a per-channel extract node keyed by (node_name, channel_index).
        if from_node.type in ('SEPXYZ', 'SEPARATE_XYZ', 'SEPRGB', 'SEPARATE_COLOR'):
            sock_name = from_sock.name.upper()
            is_color = from_node.type in ('SEPRGB', 'SEPARATE_COLOR')
            channel_map = {'X': 0, 'Y': 1, 'Z': 2, 'R': 0, 'G': 1, 'B': 2}
            idx = channel_map.get(sock_name, 0)
            cache_key = (id(from_node), idx)
            if not hasattr(self, '_sep_channel_cache'):
                self._sep_channel_cache = {}
            if cache_key not in self._sep_channel_cache:
                # Get (or create) the source vector/color input
                src_sock_name = 'Color' if is_color else 'Vector'
                src_sock = from_node.inputs.get(src_sock_name) or next(
                    (from_node.inputs[s] for s in ('Color', 'Vector', 'Image')
                     if s in from_node.inputs), None)
                mx_type = 'color3' if is_color else 'vector3'
                if src_sock is None:
                    self._sep_channel_cache[cache_key] = self._const_float(0.0)
                else:
                    if is_color:
                        col_out, col_c = self._socket_color3(src_sock)
                    else:
                        col_out, col_c = self._socket_vector3(src_sock)
                    ext = self._ng.addNode('extract', self._uid('ext'), 'float')
                    inp = ext.addInput('in', mx_type)
                    if col_out:
                        self._connect(inp, col_out)
                    else:
                        inp.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
                    ext.addInput('index', 'integer').setValue(idx)
                    self._sep_channel_cache[cache_key] = ext.addOutput('out', 'float')
            return self._sep_channel_cache[cache_key]

        mx_out = self._translate_node(from_node)
        # If the from_node has multiple outputs, select by socket name
        if mx_out and hasattr(mx_out, 'getParent'):
            node_elem = mx_out.getParent()
            named = node_elem.getOutput(from_sock.name.lower().replace(' ', '_'))
            if named:
                return named
        return mx_out

    def _socket_color3(self, bl_socket, default=(0.5, 0.5, 0.5)):
        """Wire a color3 input from a socket. Returns (mx_output_or_None, const_value)."""
        out = self._get_input_output(bl_socket)
        if out:
            return out, None
        dv = bl_socket.default_value
        try:
            return None, (float(dv[0]), float(dv[1]), float(dv[2]))
        except Exception:
            return None, default

    def _socket_float(self, bl_socket, default=0.0):
        """Wire a float input from a socket."""
        out = self._get_input_output(bl_socket)
        if out:
            out = self._to_float(out)  # coerce color3/vector3 → float via luminance
            return out, None
        try:
            return None, float(bl_socket.default_value)
        except Exception:
            return None, default

    def _socket_vector3(self, bl_socket, default=(0.0, 0.0, 0.0)):
        out = self._get_input_output(bl_socket)
        if out:
            return out, None
        dv = bl_socket.default_value
        try:
            return None, (float(dv[0]), float(dv[1]), float(dv[2]))
        except Exception:
            return None, default

    def _to_float(self, src_out):
        """Ensure src_out is a float Output.
        If it is already float, return it unchanged.
        If it is color3, insert luminance→extract to get a scalar.
        If it is vector3, extract the X component."""
        try:
            src_type = src_out.getType()
        except Exception:
            return src_out
        if src_type == 'float':
            return src_out
        if src_type in ('color3', 'color4'):
            lum = self._ng.addNode('luminance', self._uid('lum'), 'color3')
            self._connect(lum.addInput('in', 'color3'), src_out)
            ext = self._ng.addNode('extract', self._uid('ext'), 'float')
            ext.addInput('in', 'color3').setNodeName(lum.getName())
            ext.addInput('index', 'integer').setValue(0)
            return ext.addOutput('out', 'float')
        if src_type in ('vector3', 'vector2', 'vector4'):
            ext = self._ng.addNode('extract', self._uid('ext'), 'float')
            self._connect(ext.addInput('in', src_type), src_out)
            ext.addInput('index', 'integer').setValue(0)
            return ext.addOutput('out', 'float')
        return src_out

    def _to_color3(self, src_out):
        """Ensure src_out is a color3 Output.
        If it is already color3, return it unchanged.
        If it is float, replicate into all three channels via combine3."""
        try:
            src_type = src_out.getType()
        except Exception:
            return src_out
        if src_type == 'color3':
            return src_out
        if src_type == 'float':
            comb = self._ng.addNode('combine3', self._uid('f2c'), 'color3')
            for ch in ('in1', 'in2', 'in3'):
                self._connect(comb.addInput(ch, 'float'), src_out)
            return comb.addOutput('out', 'color3')
        return src_out

    def _wire_input(self, mx_node, inp_name, inp_type, bl_socket, default_val):
        """Set an input on mx_node from either a connection or a constant.
        Automatically inserts type-conversion nodes when the source type
        doesn't match the target type (e.g. color3 → float)."""
        import MaterialX as mx
        src_out, const = None, None
        if hasattr(bl_socket, 'links'):
            if inp_type == 'color3':
                src_out, const = self._socket_color3(bl_socket,
                    default_val if default_val else (0.5, 0.5, 0.5))
            elif inp_type == 'float':
                src_out, const = self._socket_float(bl_socket,
                    default_val if default_val is not None else 0.0)
            elif inp_type == 'vector3':
                src_out, const = self._socket_vector3(bl_socket,
                    default_val if default_val else (0.0, 0.0, 0.0))
            else:
                src_out, const = self._socket_color3(bl_socket,
                    default_val if default_val else (0.5, 0.5, 0.5))
        elif isinstance(bl_socket, mx.Output):
            src_out = bl_socket

        # Insert type-conversion nodes when needed
        if src_out is not None:
            if inp_type == 'float':
                src_out = self._to_float(src_out)
            elif inp_type == 'color3':
                src_out = self._to_color3(src_out)

        inp = mx_node.addInput(inp_name, inp_type)
        if src_out is not None:
            self._connect(inp, src_out)
        elif const is not None:
            if inp_type == 'float':
                inp.setValue(float(const))
            elif inp_type in ('color3', 'vector3'):
                inp.setValueString(f'{const[0]}, {const[1]}, {const[2]}')
            else:
                inp.setValue(str(const))

    # ------------------------------------------------------------------ #
    #  Constant helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _connect(mx_input, mx_output):
        """Wire mx_input to mx_output.
        Only sets the 'output' attribute for multi-output nodes — single-output
        nodes must NOT have it or MaterialX logs 'Multi-output type expected'."""
        node = mx_output.getParent()
        mx_input.setNodeName(node.getName())
        if len(node.getOutputs()) > 1:
            mx_input.setAttribute('output', mx_output.getName())

    # ------------------------------------------------------------------ #

    def _const_color3(self, r, g, b):
        import MaterialX as mx
        n = self._ng.addNode('constant', self._uid('const'), 'color3')
        n.addInput('value', 'color3').setValueString(f'{r}, {g}, {b}')
        return n.addOutput('out', 'color3')

    def _const_float(self, v):
        n = self._ng.addNode('constant', self._uid('const'), 'float')
        n.addInput('value', 'float').setValue(float(v))
        return n.addOutput('out', 'float')

    def _const_vector2(self, x, y):
        n = self._ng.addNode('constant', self._uid('const'), 'vector2')
        n.addInput('value', 'vector2').setValueString(f'{x}, {y}')
        return n.addOutput('out', 'vector2')

    def _const_vector3(self, x, y, z):
        n = self._ng.addNode('constant', self._uid('const'), 'vector3')
        n.addInput('value', 'vector3').setValueString(f'{x}, {y}, {z}')
        return n.addOutput('out', 'vector3')

    def _texcoord(self):
        """Return (or create) the shared UV texcoord node."""
        if '_texcoord' not in self._node_cache:
            n = self._ng.addNode('texcoord', self._uid('uv'), 'vector2')
            n.addInput('index', 'integer').setValue(0)
            out = n.addOutput('out', 'vector2')
            self._node_cache['_texcoord'] = out
        return self._node_cache['_texcoord']

    # ------------------------------------------------------------------ #
    #  Node translators                                                     #
    # ------------------------------------------------------------------ #

    def _tx_principled(self, n):
        """Principled BSDF → ND_open_pbr_surface_surfaceshader."""
        import MaterialX as mx

        node = self._ng.addNode(
            'open_pbr_surface', self._uid('openpbr'), 'surfaceshader')

        def _wire(mx_name, sock_name, mx_type, default):
            sock = n.inputs.get(sock_name)
            if sock is None:
                return
            self._wire_input(node, mx_name, mx_type, sock, default)

        _wire('base_color',          'Base Color',         'color3',  (0.8, 0.8, 0.8))
        node.addInput('base_weight', 'float').setValue(1.0)
        _wire('base_metalness',      'Metallic',           'float',   0.0)
        _wire('specular_roughness',  'Roughness',          'float',   0.5)
        _wire('specular_ior',        'IOR',                'float',   1.5)
        _wire('transmission_weight', 'Transmission Weight','float',   0.0)
        _wire('coat_weight',         'Coat Weight',        'float',   0.0)
        _wire('coat_roughness',      'Coat Roughness',     'float',   0.03)
        _wire('emission_luminance',  'Emission Strength',  'float',   0.0)
        _wire('emission_color',      'Emission Color',     'color3',  (1.0, 1.0, 1.0))
        _wire('geometry_normal',     'Normal',             'vector3', (0.0, 0.0, 1.0))

        return node.addOutput('out', 'surfaceshader')

    def _tx_image(self, n, force_color3=False):
        """Image Texture → ND_image_color3 or ND_image_float.

        force_color3=True forces a color3 node regardless of Blender's colorspace
        setting.  Use this for normal map images: they are always 3-channel RGB
        even when the colorspace is set to 'Non-Color' (which only means 'do not
        apply sRGB→linear decode', not 'the image is single-channel').
        """
        import os
        import shutil
        img = n.image
        if img is None:
            return self._const_color3(0.5, 0.5, 0.5)

        # Resolve file path — try abspath first, then look relative to blend file dir.
        filepath = bpy.path.abspath(img.filepath) if img.filepath else ''
        if filepath and not os.path.exists(filepath):
            blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else ''
            alt = os.path.join(blend_dir, img.filepath.lstrip('/\\'))
            if os.path.exists(alt):
                filepath = alt

        # If still not found and the image has packed data, unpack it using img.save()
        # which writes the raw pixel data without any render color management.
        # Never use save_render() — it applies Blender's display/render transforms.
        if (not filepath or not os.path.exists(filepath)) and img.packed_file:
            tex_dir = os.path.abspath(os.path.join(self.out_dir, '..', 'textures'))
            os.makedirs(tex_dir, exist_ok=True)
            # Derive filename from img.filepath (may be relative like //textures/Foo.png)
            # or fall back to img.name. Keep the original extension.
            raw_name = os.path.basename(img.filepath) if img.filepath else img.name
            if not raw_name:
                raw_name = 'texture.png'
            if not any(raw_name.lower().endswith(e)
                       for e in ('.png', '.jpg', '.jpeg', '.exr', '.tiff', '.tga')):
                raw_name += '.png'
            packed_path = os.path.join(tex_dir, raw_name)
            try:
                # Write raw packed bytes directly — no color management, no source
                # path lookup. img.packed_file.data is the original file content.
                with open(packed_path, 'wb') as fh:
                    fh.write(img.packed_file.data)
                filepath = packed_path
            except Exception:
                return self._const_color3(0.5, 0.5, 0.5)

        if not filepath or not os.path.exists(filepath):
            return self._const_color3(0.5, 0.5, 0.5)

        # Copy the texture to <out_dir>/../textures/ so the absolute path stays valid
        # when the .mtlx is compiled into OSL, regardless of where it was originally.
        tex_dir = os.path.abspath(os.path.join(self.out_dir, '..', 'textures'))
        os.makedirs(tex_dir, exist_ok=True)
        tex_name = os.path.basename(filepath)
        dest = os.path.join(tex_dir, tex_name)
        if not os.path.exists(dest) or os.path.getmtime(filepath) > os.path.getmtime(dest):
            shutil.copy2(filepath, dest)
        filepath = os.path.abspath(dest)

        is_color = (img.colorspace_settings.name not in ('Non-Color', 'Raw', 'Linear'))
        # force_color3: always emit a color3 node (needed for normal maps which are
        # stored as Non-Color in Blender but are still 3-channel RGB images).
        mx_type = 'color3' if (is_color or force_color3) else 'float'
        node_id = 'image'

        mx_node = self._ng.addNode(node_id, self._uid('img'), mx_type)
        file_inp = mx_node.addInput('file', 'filename')
        file_inp.setValueString(filepath)
        # Set colorspace so the OSL texture() call applies sRGB→linear for color maps.
        # Without this, img2_file_colorspace="" and OIIO returns raw sRGB values
        # which are used as if they were linear — producing washed-out colors.
        # Note: force_color3 images (normal maps) intentionally keep no colorspace
        # attribute — the image is linear and must NOT be sRGB-decoded.
        if is_color:
            file_inp.setAttribute('colorspace', 'srgb_texture')

        # UV / texcoord
        uv_sock = n.inputs.get('Vector')
        uv_out  = self._get_input_output(uv_sock) if uv_sock else None
        if uv_out is None:
            uv_out = self._texcoord()
        uv_inp = mx_node.addInput('texcoord', 'vector2')
        self._connect(uv_inp, uv_out)

        return mx_node.addOutput('out', mx_type)

    def _tx_noise(self, n):
        """Noise Texture → ND_fractal3d_float (best available approximation)."""
        scale   = float(n.inputs['Scale'].default_value)   if 'Scale'   in n.inputs else 5.0
        detail  = float(n.inputs['Detail'].default_value)  if 'Detail'  in n.inputs else 2.0
        rough   = float(n.inputs['Roughness'].default_value) if 'Roughness' in n.inputs else 0.5

        mx_node = self._ng.addNode('fractal3d', self._uid('noise'), 'float')
        mx_node.addInput('octaves',    'integer').setValue(max(1, int(detail)))
        mx_node.addInput('lacunarity', 'float').setValue(2.0)
        mx_node.addInput('diminish',   'float').setValue(rough)

        # Apply scale via position transform
        pos_n = self._ng.addNode('multiply', self._uid('nscale'), 'vector3')
        tc    = self._ng.addNode('position', self._uid('pos'), 'vector3')
        tc.addInput('space', 'string').setValueString('object')
        pi = pos_n.addInput('in1', 'vector3')
        pi.setNodeName(tc.getName())
        pos_n.addInput('in2', 'vector3').setValueString(f'{scale}, {scale}, {scale}')
        pos_out = pos_n.addOutput('out', 'vector3')

        noise_pos = mx_node.addInput('position', 'vector3')
        noise_pos.setNodeName(pos_n.getName())

        return mx_node.addOutput('out', 'float')

    def _tx_wave(self, n):
        """Wave Texture → sin-based bands/rings matching Blender's formula:
        output = 0.5 + 0.5 * sin(2π * (coord + distortion * noise + phase))
        where coord is an axis component (BANDS) or distance from origin (RINGS).
        """
        kTwoPi = 6.28318530717958647692

        scale        = float(n.inputs['Scale'].default_value)           if 'Scale'            in n.inputs else 5.0
        distortion   = float(n.inputs['Distortion'].default_value)      if 'Distortion'       in n.inputs else 0.0
        detail       = float(n.inputs['Detail'].default_value)          if 'Detail'           in n.inputs else 2.0
        detail_scale = float(n.inputs['Detail Scale'].default_value)    if 'Detail Scale'     in n.inputs else 1.0
        detail_rough = float(n.inputs['Detail Roughness'].default_value) if 'Detail Roughness' in n.inputs else 0.5
        phase        = float(n.inputs['Phase Offset'].default_value)    if 'Phase Offset'     in n.inputs else 0.0

        wave_type  = getattr(n, 'wave_type',       'BANDS')
        bands_dir  = getattr(n, 'bands_direction', 'X')

        # Object-space position scaled by Scale
        pos_n = self._ng.addNode('position', self._uid('wpos'), 'vector3')
        pos_n.addInput('space', 'string').setValueString('object')

        sc_n = self._ng.addNode('multiply', self._uid('wsc'), 'vector3')
        sc_n.addInput('in1', 'vector3').setNodeName(pos_n.getName())
        sc_n.addInput('in2', 'vector3').setValueString(f'{scale}, {scale}, {scale}')

        # Derive scalar coordinate from position
        if wave_type == 'RINGS':
            mag_n = self._ng.addNode('magnitude', self._uid('wmag'), 'float')
            mag_n.addInput('in', 'vector3').setNodeName(sc_n.getName())
            coord_node = mag_n
        else:
            axis_map = {'X': 0, 'Y': 1, 'Z': 2}
            if bands_dir == 'DIAGONAL':
                # sum X+Y+Z components
                ex = [self._ng.addNode('extract', self._uid('wex'), 'float') for _ in range(3)]
                for i, e in enumerate(ex):
                    e.addInput('in', 'vector3').setNodeName(sc_n.getName())
                    e.addInput('index', 'integer').setValue(i)
                a01 = self._ng.addNode('add', self._uid('wadd'), 'float')
                a01.addInput('in1', 'float').setNodeName(ex[0].getName())
                a01.addInput('in2', 'float').setNodeName(ex[1].getName())
                a012 = self._ng.addNode('add', self._uid('wadd'), 'float')
                a012.addInput('in1', 'float').setNodeName(a01.getName())
                a012.addInput('in2', 'float').setNodeName(ex[2].getName())
                coord_node = a012
            else:
                idx = axis_map.get(bands_dir, 0)
                ext_n = self._ng.addNode('extract', self._uid('wext'), 'float')
                ext_n.addInput('in', 'vector3').setNodeName(sc_n.getName())
                ext_n.addInput('index', 'integer').setValue(idx)
                coord_node = ext_n

        # Optional distortion: coord += distortion * fractal3d(pos * detail_scale)
        if abs(distortion) > 1e-6:
            dsc_n = self._ng.addNode('multiply', self._uid('wdsc'), 'vector3')
            dsc_n.addInput('in1', 'vector3').setNodeName(pos_n.getName())
            dsc_n.addInput('in2', 'vector3').setValueString(
                f'{detail_scale}, {detail_scale}, {detail_scale}')

            frac_n = self._ng.addNode('fractal3d', self._uid('wfrac'), 'float')
            frac_n.addInput('octaves',    'integer').setValue(max(1, int(detail)))
            frac_n.addInput('lacunarity', 'float').setValue(2.0)
            frac_n.addInput('diminish',   'float').setValue(detail_rough)
            frac_n.addInput('position',   'vector3').setNodeName(dsc_n.getName())

            dmul_n = self._ng.addNode('multiply', self._uid('wdmul'), 'float')
            dmul_n.addInput('in1', 'float').setNodeName(frac_n.getName())
            dmul_n.addInput('in2', 'float').setValue(float(distortion))

            dadd_n = self._ng.addNode('add', self._uid('wdadd'), 'float')
            dadd_n.addInput('in1', 'float').setNodeName(coord_node.getName())
            dadd_n.addInput('in2', 'float').setNodeName(dmul_n.getName())
            coord_node = dadd_n

        # Phase offset
        if abs(phase) > 1e-8:
            ph_n = self._ng.addNode('add', self._uid('wph'), 'float')
            ph_n.addInput('in1', 'float').setNodeName(coord_node.getName())
            ph_n.addInput('in2', 'float').setValue(float(phase))
            coord_node = ph_n

        # Multiply by 2π then sin
        pi2_n = self._ng.addNode('multiply', self._uid('w2pi'), 'float')
        pi2_n.addInput('in1', 'float').setNodeName(coord_node.getName())
        pi2_n.addInput('in2', 'float').setValue(kTwoPi)

        sin_n = self._ng.addNode('sin', self._uid('wsin'), 'float')
        sin_n.addInput('in', 'float').setNodeName(pi2_n.getName())

        # Remap [-1,1] → [0,1]: out = sin * 0.5 + 0.5
        mh_n = self._ng.addNode('multiply', self._uid('wmh'), 'float')
        mh_n.addInput('in1', 'float').setNodeName(sin_n.getName())
        mh_n.addInput('in2', 'float').setValue(0.5)

        ah_n = self._ng.addNode('add', self._uid('wah'), 'float')
        ah_n.addInput('in1', 'float').setNodeName(mh_n.getName())
        ah_n.addInput('in2', 'float').setValue(0.5)

        return ah_n.addOutput('out', 'float')

    def _tx_voronoi(self, n):
        """Voronoi Texture → ND_cellnoise3d_float."""
        scale = float(n.inputs['Scale'].default_value) if 'Scale' in n.inputs else 5.0

        tc    = self._ng.addNode('position', self._uid('pos'), 'vector3')
        tc.addInput('space', 'string').setValueString('object')

        scale_n = self._ng.addNode('multiply', self._uid('vscale'), 'vector3')
        scale_n.addInput('in1', 'vector3').setNodeName(tc.getName())
        scale_n.addInput('in2', 'vector3').setValueString(f'{scale}, {scale}, {scale}')

        mx_node = self._ng.addNode('cellnoise3d', self._uid('voronoi'), 'float')
        mx_node.addInput('position', 'vector3').setNodeName(scale_n.getName())

        return mx_node.addOutput('out', 'float')

    def _tx_musgrave(self, n):
        """Musgrave Texture → ND_fractal3d_float."""
        return self._tx_noise(n)

    def _tx_gradient(self, n):
        """Gradient Texture → UV-based gradient.

        Blender's Gradient Texture uses Generated coordinates by default
        (object bounding-box 0..1).  For LINEAR mode this is coord.x (U),
        but in practice most materials author vertical gradients by relying
        on the object's UV V axis.

        Mapping:
          LINEAR    → UV V (index=1) — vertical gradient, most common use
          QUADRATIC → UV V (index=1)
          EASING    → UV V (index=1)
          DIAGONAL  → UV U + V summed
          RADIAL    → atan2(V-0.5, U-0.5) normalised to [0,1]
          SPHERICAL / QUADRATIC_SPHERE → length from (0.5,0.5) normalised

        If the Vector socket is explicitly connected, follow the connection
        instead of using the default texcoord.
        """
        gradient_type = getattr(n, 'gradient_type', 'LINEAR')

        # --- Vector input ---
        vec_sock = n.inputs.get('Vector') if n.inputs else None
        vec_out  = None
        vec_c    = None
        vec_type = 'vector2'  # default: UV texcoord
        tc_node  = None

        if vec_sock and vec_sock.links:
            # Explicit Vector input — trace it; detect actual output type
            vec_out, vec_c = self._socket_vector3(vec_sock)
            if vec_out is not None:
                try:
                    vec_type = vec_out.getType()
                except Exception:
                    vec_type = 'vector3'
            else:
                vec_type = 'vector3'
        else:
            # Default: use UV texcoord (vector2)
            tc_node = self._ng.addNode('texcoord', self._uid('uv'), 'vector2')
            tc_node.addInput('index', 'integer').setValue(0)
            vec_type = 'vector2'

        def _extract_uv(idx):
            """Extract component idx from UV texcoord or traced vector."""
            e = self._ng.addNode('extract', self._uid('ext'), 'float')
            if tc_node is not None:
                e.addInput('in', 'vector2').setNodeName(tc_node.getName())
            else:
                inp = e.addInput('in', vec_type)
                if vec_out:
                    self._connect(inp, vec_out)
                elif vec_c:
                    if vec_type == 'vector2':
                        inp.setValueString(f'{vec_c[0]}, {vec_c[1]}')
                    else:
                        inp.setValueString(f'{vec_c[0]}, {vec_c[1]}, {vec_c[2]}')
            e.addInput('index', 'integer').setValue(idx)
            return e.addOutput('out', 'float')

        if gradient_type in ('LINEAR', 'QUADRATIC', 'EASING', 'QUADRATIC_SPHERE'):
            # Blender's Gradient Texture LINEAR mode uses the X component of its
            # input vector as the factor.  When no Vector is connected, Blender
            # uses Generated coordinates where X varies horizontally — use index 0.
            return _extract_uv(0)

        elif gradient_type == 'DIAGONAL':
            u = _extract_uv(0)
            v = _extract_uv(1)
            add_n = self._ng.addNode('add', self._uid('gdiag'), 'float')
            add_n.addInput('in1', 'float').setNodeName(u.getParent().getName())
            add_n.addInput('in2', 'float').setNodeName(v.getParent().getName())
            mul_n = self._ng.addNode('multiply', self._uid('gdiagm'), 'float')
            mul_n.addInput('in1', 'float').setNodeName(add_n.getName())
            mul_n.addInput('in2', 'float').setValue(0.5)
            return mul_n.addOutput('out', 'float')

        elif gradient_type == 'RADIAL':
            # atan2(v-0.5, u-0.5) / (2π) + 0.5  →  [0,1]
            kInv2Pi = 0.15915494309189534
            u = _extract_uv(0)
            v = _extract_uv(1)
            su = self._ng.addNode('subtract', self._uid('gru'), 'float')
            su.addInput('in1', 'float').setNodeName(u.getParent().getName())
            su.addInput('in2', 'float').setValue(0.5)
            sv = self._ng.addNode('subtract', self._uid('grv'), 'float')
            sv.addInput('in1', 'float').setNodeName(v.getParent().getName())
            sv.addInput('in2', 'float').setValue(0.5)
            at = self._ng.addNode('atan2', self._uid('grat'), 'float')
            at.addInput('in1', 'float').setNodeName(sv.getName())
            at.addInput('in2', 'float').setNodeName(su.getName())
            sc = self._ng.addNode('multiply', self._uid('grsc'), 'float')
            sc.addInput('in1', 'float').setNodeName(at.getName())
            sc.addInput('in2', 'float').setValue(kInv2Pi)
            off = self._ng.addNode('add', self._uid('groff'), 'float')
            off.addInput('in1', 'float').setNodeName(sc.getName())
            off.addInput('in2', 'float').setValue(0.5)
            return off.addOutput('out', 'float')

        else:  # SPHERICAL fallback
            u = _extract_uv(0)
            v = _extract_uv(1)
            su = self._ng.addNode('subtract', self._uid('gsu'), 'float')
            su.addInput('in1', 'float').setNodeName(u.getParent().getName())
            su.addInput('in2', 'float').setValue(0.5)
            sv = self._ng.addNode('subtract', self._uid('gsv'), 'float')
            sv.addInput('in1', 'float').setNodeName(v.getParent().getName())
            sv.addInput('in2', 'float').setValue(0.5)
            u2 = self._ng.addNode('multiply', self._uid('gsu2'), 'float')
            u2.addInput('in1', 'float').setNodeName(su.getName())
            u2.addInput('in2', 'float').setNodeName(su.getName())
            v2 = self._ng.addNode('multiply', self._uid('gsv2'), 'float')
            v2.addInput('in1', 'float').setNodeName(sv.getName())
            v2.addInput('in2', 'float').setNodeName(sv.getName())
            s = self._ng.addNode('add', self._uid('gss'), 'float')
            s.addInput('in1', 'float').setNodeName(u2.getName())
            s.addInput('in2', 'float').setNodeName(v2.getName())
            sq = self._ng.addNode('sqrt', self._uid('gssq'), 'float')
            sq.addInput('in', 'float').setNodeName(s.getName())
            cl = self._ng.addNode('clamp', self._uid('gscl'), 'float')
            cl.addInput('in', 'float').setNodeName(sq.getName())
            cl.addInput('low', 'float').setValue(0.0)
            cl.addInput('high', 'float').setValue(1.0)
            return cl.addOutput('out', 'float')

    def _tx_checker(self, n):
        """Checker/Brick Texture → ND_checkerboard_color3."""
        scale = 5.0
        if 'Scale' in n.inputs:
            scale = float(n.inputs['Scale'].default_value)

        uv_out = self._texcoord()
        # Scale UV
        scale_n = self._ng.addNode('multiply', self._uid('ckscale'), 'vector2')
        scale_n.addInput('in1', 'vector2').setNodeName(uv_out.getParent().getName())
        scale_n.addInput('in2', 'vector2').setValueString(f'{scale}, {scale}')

        mx_node = self._ng.addNode('checkerboard', self._uid('checker'), 'color3')
        mx_node.addInput('color1', 'color3').setValueString('0.8, 0.8, 0.8')
        mx_node.addInput('color2', 'color3').setValueString('0.2, 0.2, 0.2')
        mx_node.addInput('uvtiling', 'vector2').setNodeName(scale_n.getName())

        return mx_node.addOutput('out', 'color3')

    def _tx_color_ramp(self, n):
        """Color Ramp (VALTORGB) — sampled into a ramp_color3 node.
        Falls back to mix if only 2 stops."""
        ramp = n.color_ramp
        elems = sorted(ramp.elements, key=lambda e: e.position)

        fac_sock = n.inputs.get('Fac')
        fac_out, fac_const = self._socket_float(fac_sock, 0.5) \
                             if fac_sock else (None, 0.5)

        if len(elems) == 2:
            # Simple 2-stop → ND_mix
            c0 = elems[0].color
            c1 = elems[1].color
            mx_node = self._ng.addNode('mix', self._uid('ramp'), 'color3')
            mx_node.addInput('bg', 'color3').setValueString(
                f'{c0[0]}, {c0[1]}, {c0[2]}')
            mx_node.addInput('fg', 'color3').setValueString(
                f'{c1[0]}, {c1[1]}, {c1[2]}')
            fac_inp = mx_node.addInput('mix', 'float')
            if fac_out:
                self._connect(fac_inp, fac_out)
            else:
                fac_inp.setValue(float(fac_const))
            return mx_node.addOutput('out', 'color3')
        else:
            # Multi-stop — evaluate ramp at 8 positions, build mix chain
            # Sample the ramp at uniform intervals and chain ND_mix nodes
            samples = 8
            def sample(t):
                t = max(0.0, min(1.0, t))
                if t <= elems[0].position:
                    c = elems[0].color
                elif t >= elems[-1].position:
                    c = elems[-1].color
                else:
                    for i in range(len(elems)-1):
                        e0, e1 = elems[i], elems[i+1]
                        if e0.position <= t <= e1.position:
                            span = e1.position - e0.position
                            s = (t - e0.position) / span if span > 1e-8 else 0.0
                            c = [e0.color[j]*(1-s) + e1.color[j]*s for j in range(4)]
                            return c
                    c = elems[-1].color
                return list(c)

            # Build a lookup via piecewise mix chain
            step = 1.0 / (samples - 1)
            # Start with constant at t=0
            prev_col = sample(0.0)
            prev_out = self._const_color3(*prev_col[:3])
            for i in range(1, samples):
                t = i * step
                col = sample(t)
                # blend factor: how far into this segment is fac?
                seg_lo = (i-1) * step
                seg_hi = t
                # Normalize fac into [0,1] for this segment
                if fac_out or fac_const is not None:
                    # clamp((fac - seg_lo) / step, 0, 1)
                    sub_n = self._ng.addNode('subtract', self._uid('rsub'), 'float')
                    fac_src = sub_n.addInput('in1', 'float')
                    if fac_out:
                        self._connect(fac_src, fac_out)
                    else:
                        fac_src.setValue(float(fac_const))
                    sub_n.addInput('in2', 'float').setValue(float(seg_lo))
                    div_n = self._ng.addNode('divide', self._uid('rdiv'), 'float')
                    div_n.addInput('in1', 'float').setNodeName(sub_n.getName())
                    div_n.addInput('in2', 'float').setValue(float(step))
                    clamp_n = self._ng.addNode('clamp', self._uid('rclamp'), 'float')
                    clamp_n.addInput('in', 'float').setNodeName(div_n.getName())
                    clamp_n.addInput('low', 'float').setValue(0.0)
                    clamp_n.addInput('high', 'float').setValue(1.0)
                    seg_fac_out = clamp_n.addOutput('out', 'float')
                else:
                    seg_fac_out = None

                mix_n = self._ng.addNode('mix', self._uid('rmix'), 'color3')
                bg_inp = mix_n.addInput('bg', 'color3')
                self._connect(bg_inp, prev_out)
                mix_n.addInput('fg', 'color3').setValueString(
                    f'{col[0]}, {col[1]}, {col[2]}')
                fac_i = mix_n.addInput('mix', 'float')
                if seg_fac_out:
                    self._connect(fac_i, seg_fac_out)
                else:
                    fac_i.setValue(float(t))
                prev_out = mix_n.addOutput('out', 'color3')

            return prev_out

    def _tx_rgbtobw(self, n):
        """RGB to BW → luminance node (color3→color3) then extract index 0.
        MaterialX's luminance node sets all channels to Rec.709 luminance,
        so extracting any channel gives the scalar grey value."""
        col_sock = n.inputs.get('Color')
        col_out, col_const = self._socket_color3(col_sock, (0.5, 0.5, 0.5)) \
                             if col_sock else (None, (0.5, 0.5, 0.5))

        lum = self._ng.addNode('luminance', self._uid('lum'), 'color3')
        c_inp = lum.addInput('in', 'color3')
        if col_out:
            self._connect(c_inp, col_out)
        else:
            c_inp.setValueString(f'{col_const[0]}, {col_const[1]}, {col_const[2]}')

        # extract index 0 (R channel) — all channels are equal after luminance
        ext = self._ng.addNode('extract', self._uid('ext'), 'float')
        ext.addInput('in', 'color3').setNodeName(lum.getName())
        ext.addInput('index', 'integer').setValue(0)
        return ext.addOutput('out', 'float')

    def _tx_mix(self, n):
        """MIX_RGB / MIX → ND_mix or blend-type specific node."""
        bt = getattr(n, 'blend_type', 'MIX')
        mx_op = _MX_BLEND_TYPE.get(bt, 'mix')

        # Blender 4.x MIX node uses A/B; older MIX_RGB uses Color1/Color2
        a_name = 'A' if 'A' in n.inputs else 'Color1'
        b_name = 'B' if 'B' in n.inputs else 'Color2'
        f_name = 'Factor' if 'Factor' in n.inputs else 'Fac'

        col1_out, col1_c = self._socket_color3(n.inputs[a_name], (0.5,0.5,0.5)) \
                           if a_name in n.inputs else (None, (0.5,0.5,0.5))
        col2_out, col2_c = self._socket_color3(n.inputs[b_name], (0.5,0.5,0.5)) \
                           if b_name in n.inputs else (None, (0.5,0.5,0.5))
        fac_out,  fac_c  = self._socket_float(n.inputs[f_name], 0.5) \
                           if f_name in n.inputs else (None, 0.5)

        if mx_op == 'mix':
            mx_node = self._ng.addNode('mix', self._uid('mix'), 'color3')
            bg = mx_node.addInput('bg', 'color3')
            fg = mx_node.addInput('fg', 'color3')
            fc = mx_node.addInput('mix', 'float')
        else:
            mx_node = self._ng.addNode(mx_op, self._uid(mx_op), 'color3')
            bg = mx_node.addInput('in1', 'color3')
            fg = mx_node.addInput('in2', 'color3')
            fc = None

        def _set_color(inp, out, const):
            if out:
                self._connect(inp, out)
            elif const:
                inp.setValueString(f'{const[0]}, {const[1]}, {const[2]}')

        _set_color(bg, col1_out, col1_c)
        _set_color(fg, col2_out, col2_c)
        if fc is not None:
            if fac_out:
                self._connect(fc, fac_out)
            else:
                fc.setValue(float(fac_c))

        return mx_node.addOutput('out', 'color3')

    def _tx_math(self, n):
        """Math node → appropriate float arithmetic node."""
        op = n.operation if hasattr(n, 'operation') else 'ADD'
        mx_op, n_inputs = _MX_MATH_OP.get(op, ('add', 2))

        val1_out, val1_c = self._socket_float(n.inputs[0], 0.5) if len(n.inputs) > 0 else (None, 0.5)
        val2_out, val2_c = self._socket_float(n.inputs[1], 0.5) if len(n.inputs) > 1 else (None, 0.5)

        mx_node = self._ng.addNode(mx_op, self._uid(mx_op), 'float')

        def _set_float(inp_name, out, const):
            inp = mx_node.addInput(inp_name, 'float')
            if out:
                self._connect(inp, out)
            else:
                inp.setValue(float(const))

        _set_float('in1', val1_out, val1_c)
        if n_inputs >= 2:
            _set_float('in2', val2_out, val2_c)
        if op == 'CLAMP':
            mx_node.addInput('low',  'float').setValue(0.0)
            mx_node.addInput('high', 'float').setValue(1.0)

        return mx_node.addOutput('out', 'float')

    def _tx_invert(self, n):
        """Invert → ND_subtract."""
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (0.5,0.5,0.5))
        fac = float(n.inputs['Fac'].default_value) if 'Fac' in n.inputs else 1.0

        one = self._const_color3(1.0, 1.0, 1.0)
        sub = self._ng.addNode('subtract', self._uid('inv'), 'color3')
        sub.addInput('in1', 'color3').setNodeName(one.getParent().getName())
        in2 = sub.addInput('in2', 'color3')
        if col_out:
            self._connect(in2, col_out)
        else:
            in2.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')

        if abs(fac - 1.0) > 1e-4:
            # blend: result = orig * (1-fac) + inv * fac
            mix = self._ng.addNode('mix', self._uid('invmix'), 'color3')
            bg = mix.addInput('bg', 'color3')
            if col_out:
                self._connect(bg, col_out)
            else:
                bg.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
            mix.addInput('fg', 'color3').setNodeName(sub.getName())
            mix.addInput('mix', 'float').setValue(fac)
            return mix.addOutput('out', 'color3')

        return sub.addOutput('out', 'color3')

    def _tx_hue_sat(self, n):
        """HueSat → ND_hsvtorgb(ND_rgbtohsv + adjust)."""
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (0.5,0.5,0.5))
        h = float(n.inputs['Hue'].default_value)        if 'Hue'        in n.inputs else 0.5
        s = float(n.inputs['Saturation'].default_value) if 'Saturation' in n.inputs else 1.0
        v = float(n.inputs['Value'].default_value)      if 'Value'      in n.inputs else 1.0

        # Convert to HSV
        hsv_n = self._ng.addNode('rgbtohsv', self._uid('tohsv'), 'color3')
        c_inp = hsv_n.addInput('in', 'color3')
        if col_out:
            self._connect(c_inp, col_out)
        else:
            c_inp.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')

        # Multiply by HSV adjustment
        mul_n = self._ng.addNode('multiply', self._uid('hsvmul'), 'color3')
        mul_n.addInput('in1', 'color3').setNodeName(hsv_n.getName())
        mul_n.addInput('in2', 'color3').setValueString(f'{h*2.0}, {s}, {v}')

        # Convert back
        rgb_n = self._ng.addNode('hsvtorgb', self._uid('fromhsv'), 'color3')
        rgb_n.addInput('in', 'color3').setNodeName(mul_n.getName())
        return rgb_n.addOutput('out', 'color3')

    def _tx_bright_contrast(self, n):
        """BrightContrast → multiply + add chain."""
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (0.5,0.5,0.5))
        br = float(n.inputs['Bright'].default_value)   if 'Bright'   in n.inputs else 0.0
        co = float(n.inputs['Contrast'].default_value) if 'Contrast' in n.inputs else 0.0

        mul_n = self._ng.addNode('multiply', self._uid('bcmul'), 'color3')
        c_inp = mul_n.addInput('in1', 'color3')
        if col_out:
            self._connect(c_inp, col_out)
        else:
            c_inp.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        mul_n.addInput('in2', 'color3').setValueString(
            f'{1.0+co}, {1.0+co}, {1.0+co}')

        add_n = self._ng.addNode('add', self._uid('bcadd'), 'color3')
        add_n.addInput('in1', 'color3').setNodeName(mul_n.getName())
        adj = br - co * 0.5
        add_n.addInput('in2', 'color3').setValueString(f'{adj}, {adj}, {adj}')

        return add_n.addOutput('out', 'color3')

    def _tx_normal_map(self, n):
        """Normal Map → ND_normalmap_float.
        The normalmap nodedef requires 'in' as vector3 (not color3), so we
        decode the image color3 via extract+combine3 to get a vector3.
        The mx_normalmap_float OSL function does the 0-1→(-1..1) decode
        internally, so we pass raw encoded values unchanged.

        Normal map images are stored as 'Non-Color' in Blender (no sRGB decode),
        which makes _tx_image emit a 'float' (single-channel) image node.  But
        normal maps are always 3-channel RGB, so we must force color3 loading.
        If _socket_color3 returns a float output, we re-create the image as
        color3 using _tx_image(force_color3=True)."""
        color_sock = n.inputs.get('Color')
        col_out, col_c = self._socket_color3(color_sock, (0.5, 0.5, 1.0))

        # If the image node was emitted as float (Non-Color colorspace), its output
        # type is 'float' but we need 'color3' for the R/G/B channel extraction.
        if col_out is not None and col_out.getType() == 'float':
            if color_sock and color_sock.links:
                src_node = color_sock.links[0].from_node
                if src_node.type == 'TEX_IMAGE':
                    col_out = self._tx_image(src_node, force_color3=True)
        strength = float(n.inputs['Strength'].default_value) \
                   if 'Strength' in n.inputs else 1.0

        # Decode color3 → vector3 via extract(R,G,B) + combine3
        if col_out:
            ex_r = self._ng.addNode('extract', self._uid('nmex'), 'float')
            self._connect(ex_r.addInput('in', 'color3'), col_out)
            ex_r.addInput('index', 'integer').setValue(0)
            ex_g = self._ng.addNode('extract', self._uid('nmex'), 'float')
            self._connect(ex_g.addInput('in', 'color3'), col_out)
            ex_g.addInput('index', 'integer').setValue(1)
            ex_b = self._ng.addNode('extract', self._uid('nmex'), 'float')
            self._connect(ex_b.addInput('in', 'color3'), col_out)
            ex_b.addInput('index', 'integer').setValue(2)
            comb = self._ng.addNode('combine3', self._uid('nmcomb'), 'vector3')
            comb.addInput('in1', 'float').setNodeName(ex_r.getName())
            comb.addInput('in2', 'float').setNodeName(ex_g.getName())
            comb.addInput('in3', 'float').setNodeName(ex_b.getName())
            vec_out = comb.addOutput('out', 'vector3')
        else:
            vec_out = None

        nm_n = self._ng.addNode('normalmap', self._uid('nmap'), 'vector3')
        v_inp = nm_n.addInput('in', 'vector3')
        if vec_out:
            self._connect(v_inp, vec_out)
        else:
            v_inp.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        nm_n.addInput('scale', 'float').setValue(strength)
        return nm_n.addOutput('out', 'vector3')

    def _tx_bump(self, n):
        """Bump → ND_heighttonormal_vector3.
        Converts a scalar height map to a tangent-space normal vector.
        Inputs: Height (float), Strength (float), Distance (float).
        """
        height_sock    = n.inputs.get('Height')
        strength_sock  = n.inputs.get('Strength')
        distance_sock  = n.inputs.get('Distance')

        height_out, height_c = self._socket_float(height_sock, 0.5) \
                               if height_sock else (None, 0.5)
        strength = float(strength_sock.default_value) \
                   if strength_sock and not strength_sock.links else \
                   (float(strength_sock.default_value) if strength_sock else 1.0)
        distance = float(distance_sock.default_value) \
                   if distance_sock and not distance_sock.links else \
                   (float(distance_sock.default_value) if distance_sock else 1.0)
        scale = strength * distance

        h2n = self._ng.addNode('heighttonormal', self._uid('bump'), 'vector3')
        h_inp = h2n.addInput('in', 'float')
        if height_out:
            self._connect(h_inp, height_out)
        else:
            h_inp.setValue(float(height_c))
        h2n.addInput('scale', 'float').setValue(scale)
        return h2n.addOutput('out', 'vector3')

    def _tx_mapping(self, n):
        """Mapping node → explicit UV transform matching Blender's forward transform:
           output = rotate(input - location) * scale
        Blender's Mapping node multiplies by scale (forward), unlike place2d which
        divides (inverse/texture-space).  Using multiply correctly handles scale=0
        (collapses that axis to zero) and matches Cycles' behavior exactly.
        """
        import math
        loc = n.inputs.get('Location')
        rot = n.inputs.get('Rotation')
        sc  = n.inputs.get('Scale')

        lx = float(loc.default_value[0]) if loc else 0.0
        ly = float(loc.default_value[1]) if loc else 0.0
        rz = float(rot.default_value[2]) if rot else 0.0
        sx = float(sc.default_value[0])  if sc  else 1.0
        sy = float(sc.default_value[1])  if sc  else 1.0

        # Step 1: texcoord - location
        uv_out = self._texcoord()
        if abs(lx) > 1e-9 or abs(ly) > 1e-9:
            sub_n = self._ng.addNode('subtract', self._uid('msub'), 'vector2')
            self._connect(sub_n.addInput('in1', 'vector2'), uv_out)
            sub_n.addInput('in2', 'vector2').setValueString(f'{lx}, {ly}')
            uv_out = sub_n.addOutput('out', 'vector2')

        # Step 2: rotate around Z (2D rotation)
        if abs(rz) > 1e-9:
            deg = rz * 57.2957795  # rad→deg
            rot_n = self._ng.addNode('rotate2d', self._uid('mrot'), 'vector2')
            self._connect(rot_n.addInput('in', 'vector2'), uv_out)
            rot_n.addInput('amount', 'float').setValue(deg)
            uv_out = rot_n.addOutput('out', 'vector2')

        # Step 3: multiply by scale
        if abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9:
            mul_n = self._ng.addNode('multiply', self._uid('mscale'), 'vector2')
            self._connect(mul_n.addInput('in1', 'vector2'), uv_out)
            mul_n.addInput('in2', 'vector2').setValueString(f'{sx}, {sy}')
            uv_out = mul_n.addOutput('out', 'vector2')

        return uv_out

    def _tx_texcoord(self, n):
        """UV Map / Texture Coordinate → ND_texcoord_vector2."""
        return self._texcoord()

    def _tx_separate(self, n):
        """Separate XYZ/RGB/Color → extract index 0 (X or R channel).
        Uses ND_extract rather than separate3 (multi-output) to avoid
        nodedef resolution issues in the OSL generator."""
        is_color = n.type in ('SEPRGB', 'SEPARATE_COLOR')
        sock_name = 'Color' if is_color else 'Vector'
        src_sock = n.inputs.get(sock_name) or next(
            (n.inputs[s] for s in ('Color', 'Vector', 'Image') if s in n.inputs), None)
        if src_sock is None:
            return self._const_float(0.0)

        mx_type = 'color3' if is_color else 'vector3'
        if is_color:
            col_out, col_c = self._socket_color3(src_sock)
        else:
            col_out, col_c = self._socket_vector3(src_sock)

        ext = self._ng.addNode('extract', self._uid('ext'), 'float')
        inp = ext.addInput('in', mx_type)
        if col_out:
            self._connect(inp, col_out)
        else:
            inp.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        ext.addInput('index', 'integer').setValue(0)
        return ext.addOutput('out', 'float')

    def _tx_combine(self, n):
        """Combine XYZ/RGB → ND_combine3."""
        x_out, x_c = self._socket_float(n.inputs[0], 0.0) if len(n.inputs) > 0 else (None, 0.0)
        y_out, y_c = self._socket_float(n.inputs[1], 0.0) if len(n.inputs) > 1 else (None, 0.0)
        z_out, z_c = self._socket_float(n.inputs[2], 0.0) if len(n.inputs) > 2 else (None, 0.0)

        comb = self._ng.addNode('combine3', self._uid('comb'), 'vector3')
        def _si(name, out, c):
            inp = comb.addInput(name, 'float')
            if out:
                self._connect(inp, out)
            else:
                inp.setValue(float(c))
        _si('in1', x_out, x_c)
        _si('in2', y_out, y_c)
        _si('in3', z_out, z_c)
        return comb.addOutput('out', 'vector3')

    def _tx_gamma(self, n):
        """Gamma → ND_power."""
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (0.5,0.5,0.5))
        g = float(n.inputs['Gamma'].default_value) if 'Gamma' in n.inputs else 2.2

        pw = self._ng.addNode('power', self._uid('gamma'), 'color3')
        c_inp = pw.addInput('in1', 'color3')
        if col_out:
            self._connect(c_inp, col_out)
        else:
            c_inp.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        pw.addInput('in2', 'color3').setValueString(f'{g}, {g}, {g}')
        return pw.addOutput('out', 'color3')

    def _tx_curve_rgb(self, n):
        """RGB Curves — sample 8 points and build a mix chain (same as color ramp)."""
        # Approximate by treating it as a brightness/contrast adjustment
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (0.5,0.5,0.5))
        # Pass through — can't faithfully evaluate the curve without pixel data
        # Emit identity multiply
        mul = self._ng.addNode('multiply', self._uid('curve'), 'color3')
        c_inp = mul.addInput('in1', 'color3')
        if col_out:
            self._connect(c_inp, col_out)
        else:
            c_inp.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        mul.addInput('in2', 'color3').setValueString('1.0, 1.0, 1.0')
        return mul.addOutput('out', 'color3')

    def _tx_group(self, n):
        """Group node — inline by recursively translating its internal tree."""
        if not n.node_tree:
            return self._const_color3(0.5, 0.5, 0.5)
        inner_output = next(
            (nd for nd in n.node_tree.nodes if nd.type == 'GROUP_OUTPUT'), None)
        if inner_output is None:
            return self._const_color3(0.5, 0.5, 0.5)
        if inner_output.inputs and inner_output.inputs[0].links:
            return self._translate_node(inner_output.inputs[0].links[0].from_node)
        return self._const_color3(0.5, 0.5, 0.5)

    # ------------------------------------------------------------------ #
    #  Shader / BSDF nodes                                                  #
    # ------------------------------------------------------------------ #

    def _tx_bsdf_diffuse(self, n):
        """Diffuse/Toon/Sheen BSDF → open_pbr_surface (pure diffuse)."""
        node = self._ng.addNode('open_pbr_surface', self._uid('openpbr'), 'surfaceshader')
        node.addInput('base_weight', 'float').setValue(1.0)
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (0.8, 0.8, 0.8))
        c = node.addInput('base_color', 'color3')
        if col_out: self._connect(c, col_out)
        else: c.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        node.addInput('base_metalness', 'float').setValue(0.0)
        node.addInput('specular_roughness', 'float').setValue(1.0)
        return node.addOutput('out', 'surfaceshader')

    def _tx_bsdf_glossy(self, n):
        """Glossy/Metallic BSDF → open_pbr_surface (metallic)."""
        node = self._ng.addNode('open_pbr_surface', self._uid('openpbr'), 'surfaceshader')
        node.addInput('base_weight', 'float').setValue(1.0)
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (1.0, 1.0, 1.0))
        c = node.addInput('base_color', 'color3')
        if col_out: self._connect(c, col_out)
        else: c.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        node.addInput('base_metalness', 'float').setValue(1.0)
        rough_out, rough_c = self._socket_float(n.inputs.get('Roughness'), 0.5)
        r = node.addInput('specular_roughness', 'float')
        if rough_out: self._connect(r, rough_out)
        else: r.setValue(float(rough_c))
        return node.addOutput('out', 'surfaceshader')

    def _tx_bsdf_glass(self, n):
        """Glass BSDF → open_pbr_surface (full transmission)."""
        node = self._ng.addNode('open_pbr_surface', self._uid('openpbr'), 'surfaceshader')
        node.addInput('base_weight', 'float').setValue(0.0)
        node.addInput('transmission_weight', 'float').setValue(1.0)
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (1.0, 1.0, 1.0))
        c = node.addInput('base_color', 'color3')
        if col_out: self._connect(c, col_out)
        else: c.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        ior_out, ior_c = self._socket_float(n.inputs.get('IOR'), 1.5)
        i = node.addInput('specular_ior', 'float')
        if ior_out: self._connect(i, ior_out)
        else: i.setValue(float(ior_c))
        rough_out, rough_c = self._socket_float(n.inputs.get('Roughness'), 0.0)
        r = node.addInput('specular_roughness', 'float')
        if rough_out: self._connect(r, rough_out)
        else: r.setValue(float(rough_c))
        return node.addOutput('out', 'surfaceshader')

    def _tx_bsdf_refraction(self, n):
        """Refraction BSDF → open_pbr_surface (transmission, no reflection)."""
        node = self._ng.addNode('open_pbr_surface', self._uid('openpbr'), 'surfaceshader')
        node.addInput('base_weight', 'float').setValue(0.0)
        node.addInput('transmission_weight', 'float').setValue(1.0)
        node.addInput('specular_weight', 'float').setValue(0.0)
        ior_out, ior_c = self._socket_float(n.inputs.get('IOR'), 1.5)
        i = node.addInput('specular_ior', 'float')
        if ior_out: self._connect(i, ior_out)
        else: i.setValue(float(ior_c))
        return node.addOutput('out', 'surfaceshader')

    def _tx_bsdf_transparent(self, n):
        """Transparent/Translucent BSDF → open_pbr_surface (no base, no spec)."""
        node = self._ng.addNode('open_pbr_surface', self._uid('openpbr'), 'surfaceshader')
        node.addInput('base_weight', 'float').setValue(0.0)
        node.addInput('specular_weight', 'float').setValue(0.0)
        return node.addOutput('out', 'surfaceshader')

    def _tx_emission(self, n):
        """Emission / Background → open_pbr_surface (emissive only)."""
        node = self._ng.addNode('open_pbr_surface', self._uid('openpbr'), 'surfaceshader')
        node.addInput('base_weight', 'float').setValue(0.0)
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (1.0, 1.0, 1.0))
        c = node.addInput('emission_color', 'color3')
        if col_out: self._connect(c, col_out)
        else: c.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        str_out, str_c = self._socket_float(n.inputs.get('Strength') or n.inputs.get('Energy'), 1.0)
        s = node.addInput('emission_luminance', 'float')
        if str_out: self._connect(s, str_out)
        else: s.setValue(float(str_c))
        return node.addOutput('out', 'surfaceshader')

    def _tx_sss(self, n):
        """Subsurface Scattering → open_pbr_surface (subsurface)."""
        node = self._ng.addNode('open_pbr_surface', self._uid('openpbr'), 'surfaceshader')
        node.addInput('base_weight', 'float').setValue(1.0)
        col_out, col_c = self._socket_color3(n.inputs.get('Color'), (0.8, 0.8, 0.8))
        c = node.addInput('base_color', 'color3')
        if col_out: self._connect(c, col_out)
        else: c.setValueString(f'{col_c[0]}, {col_c[1]}, {col_c[2]}')
        node.addInput('subsurface_weight', 'float').setValue(1.0)
        scale_out, scale_c = self._socket_float(n.inputs.get('Scale'), 0.1)
        sc = node.addInput('subsurface_scale', 'float')
        if scale_out: self._connect(sc, scale_out)
        else: sc.setValue(float(scale_c))
        return node.addOutput('out', 'surfaceshader')

    def _tx_mix_shader(self, n):
        """Mix Shader → mix of two surfaceshaders."""
        fac_out, fac_c = self._socket_float(n.inputs.get('Fac') or n.inputs.get('Factor'), 0.5)
        sh1_out = self._get_input_output(n.inputs[1]) if len(n.inputs) > 1 else None
        sh2_out = self._get_input_output(n.inputs[2]) if len(n.inputs) > 2 else None
        mix = self._ng.addNode('mix', self._uid('mixsh'), 'surfaceshader')
        bg = mix.addInput('bg', 'surfaceshader')
        if sh1_out: self._connect(bg, sh1_out)
        fg = mix.addInput('fg', 'surfaceshader')
        if sh2_out: self._connect(fg, sh2_out)
        fc = mix.addInput('mix', 'float')
        if fac_out: self._connect(fc, fac_out)
        else: fc.setValue(float(fac_c))
        return mix.addOutput('out', 'surfaceshader')

    def _tx_add_shader(self, n):
        """Add Shader → add of two surfaceshaders."""
        sh1_out = self._get_input_output(n.inputs[0]) if len(n.inputs) > 0 else None
        sh2_out = self._get_input_output(n.inputs[1]) if len(n.inputs) > 1 else None
        add = self._ng.addNode('add', self._uid('addsh'), 'surfaceshader')
        i1 = add.addInput('in1', 'surfaceshader')
        if sh1_out: self._connect(i1, sh1_out)
        i2 = add.addInput('in2', 'surfaceshader')
        if sh2_out: self._connect(i2, sh2_out)
        return add.addOutput('out', 'surfaceshader')

    # ------------------------------------------------------------------ #
    #  Texture nodes                                                        #
    # ------------------------------------------------------------------ #

    def _tx_whitenoise(self, n):
        """White Noise Texture → cellnoise3d (closest MaterialX equivalent)."""
        tc  = self._ng.addNode('position', self._uid('pos'), 'vector3')
        tc.addInput('space', 'string').setValueString('object')
        mx_node = self._ng.addNode('cellnoise3d', self._uid('wnoise'), 'float')
        mx_node.addInput('position', 'vector3').setNodeName(tc.getName())
        return mx_node.addOutput('out', 'float')

    # ------------------------------------------------------------------ #
    #  Color nodes                                                          #
    # ------------------------------------------------------------------ #

    def _tx_blackbody(self, n):
        """Blackbody → MaterialX blackbody node."""
        temp_out, temp_c = self._socket_float(n.inputs.get('Temperature'), 6500.0)
        bb = self._ng.addNode('blackbody', self._uid('bb'), 'color3')
        t = bb.addInput('temperature', 'float')
        if temp_out: self._connect(t, temp_out)
        else: t.setValue(float(temp_c))
        return bb.addOutput('out', 'color3')

    # ------------------------------------------------------------------ #
    #  Vector nodes                                                         #
    # ------------------------------------------------------------------ #

    def _tx_displacement(self, n):
        """Displacement → MaterialX displacement node."""
        ht_out, ht_c = self._socket_float(n.inputs.get('Height'), 0.0)
        mid_out, mid_c = self._socket_float(n.inputs.get('Midlevel'), 0.5)
        sc_out, sc_c = self._socket_float(n.inputs.get('Scale'), 1.0)
        # Offset = (height - midlevel) * scale
        sub = self._ng.addNode('subtract', self._uid('dsub'), 'float')
        h = sub.addInput('in1', 'float')
        if ht_out: self._connect(h, ht_out)
        else: h.setValue(float(ht_c))
        m = sub.addInput('in2', 'float')
        if mid_out: self._connect(m, mid_out)
        else: m.setValue(float(mid_c))
        mul = self._ng.addNode('multiply', self._uid('dmul'), 'float')
        mul.addInput('in1', 'float').setNodeName(sub.getName())
        sc = mul.addInput('in2', 'float')
        if sc_out: self._connect(sc, sc_out)
        else: sc.setValue(float(sc_c))
        disp = self._ng.addNode('displacement', self._uid('disp'), 'displacementshader')
        disp.addInput('displacement', 'float').setNodeName(mul.getName())
        return disp.addOutput('out', 'displacementshader')

    def _tx_vector_displacement(self, n):
        """Vector Displacement → MaterialX displacement with vector offset."""
        vec_out, vec_c = self._socket_vector3(n.inputs.get('Vector'), (0.0, 0.0, 0.0))
        sc_out, sc_c = self._socket_float(n.inputs.get('Scale'), 1.0)
        mul = self._ng.addNode('multiply', self._uid('vdmul'), 'vector3')
        v = mul.addInput('in1', 'vector3')
        if vec_out: self._connect(v, vec_out)
        else: v.setValueString(f'{vec_c[0]}, {vec_c[1]}, {vec_c[2]}')
        sc = mul.addInput('in2', 'float')
        if sc_out: self._connect(sc, sc_out)
        else: sc.setValue(float(sc_c))
        disp = self._ng.addNode('displacement', self._uid('vdisp'), 'displacementshader')
        disp.addInput('offset', 'vector3').setNodeName(mul.getName())
        return disp.addOutput('out', 'displacementshader')

    _VECTOR_MATH_OPS = {
        'ADD':           ('add',        'vector3', 2),
        'SUBTRACT':      ('subtract',   'vector3', 2),
        'MULTIPLY':      ('multiply',   'vector3', 2),
        'DIVIDE':        ('divide',     'vector3', 2),
        'MULTIPLY_ADD':  ('multiply',   'vector3', 2),  # approximation
        'CROSS_PRODUCT': ('crossproduct', 'vector3', 2),
        'PROJECT':       ('dotproduct', 'float',   2),  # approximation
        'REFLECT':       ('reflect',    'vector3', 2),
        'REFRACT':       ('refract',    'vector3', 2),
        'DOT_PRODUCT':   ('dotproduct', 'float',   2),
        'DISTANCE':      ('distance',   'float',   2),
        'LENGTH':        ('magnitude',  'float',   1),
        'SCALE':         ('multiply',   'vector3', 2),
        'NORMALIZE':     ('normalize',  'vector3', 1),
        'ABSOLUTE':      ('absval',     'vector3', 1),
        'MINIMUM':       ('min',        'vector3', 2),
        'MAXIMUM':       ('max',        'vector3', 2),
        'FLOOR':         ('floor',      'vector3', 1),
        'CEIL':          ('ceil',       'vector3', 1),
        'FRACTION':      ('fract',      'vector3', 1),
        'MODULO':        ('modulo',     'vector3', 2),
        'WRAP':          ('add',        'vector3', 2),  # no equivalent
        'SNAP':          ('floor',      'vector3', 1),  # approximation
        'SINE':          ('sin',        'vector3', 1),
        'COSINE':        ('cos',        'vector3', 1),
        'TANGENT':       ('tan',        'vector3', 1),
    }

    def _tx_vector_math(self, n):
        """Vector Math → corresponding MaterialX node."""
        op = n.operation if hasattr(n, 'operation') else 'ADD'
        mx_op, out_type, n_inputs = self._VECTOR_MATH_OPS.get(op, ('add', 'vector3', 2))
        v1_out, v1_c = self._socket_vector3(n.inputs[0], (0.0, 0.0, 0.0)) \
                       if len(n.inputs) > 0 else (None, (0.0, 0.0, 0.0))
        v2_out, v2_c = self._socket_vector3(n.inputs[1], (0.0, 0.0, 0.0)) \
                       if len(n.inputs) > 1 else (None, (0.0, 0.0, 0.0))
        mx_node = self._ng.addNode(mx_op, self._uid('vm'), out_type)
        i1 = mx_node.addInput('in1', 'vector3')
        if v1_out: self._connect(i1, v1_out)
        else: i1.setValueString(f'{v1_c[0]}, {v1_c[1]}, {v1_c[2]}')
        if n_inputs >= 2:
            i2 = mx_node.addInput('in2', 'vector3')
            if v2_out: self._connect(i2, v2_out)
            else: i2.setValueString(f'{v2_c[0]}, {v2_c[1]}, {v2_c[2]}')
        return mx_node.addOutput('out', out_type)

    def _tx_vector_rotate(self, n):
        """Vector Rotate → rotate3d."""
        vec_out, vec_c = self._socket_vector3(n.inputs.get('Vector'), (0.0, 0.0, 0.0))
        angle_out, angle_c = self._socket_float(n.inputs.get('Angle'), 0.0)
        axis_out, axis_c = self._socket_vector3(n.inputs.get('Axis'), (0.0, 0.0, 1.0))
        rot = self._ng.addNode('rotate3d', self._uid('rot3d'), 'vector3')
        v = rot.addInput('in', 'vector3')
        if vec_out: self._connect(v, vec_out)
        else: v.setValueString(f'{vec_c[0]}, {vec_c[1]}, {vec_c[2]}')
        a = rot.addInput('amount', 'float')
        if angle_out: self._connect(a, angle_out)
        else: a.setValue(float(angle_c) * 57.2957795)  # rad→deg
        ax = rot.addInput('axis', 'vector3')
        if axis_out: self._connect(ax, axis_out)
        else: ax.setValueString(f'{axis_c[0]}, {axis_c[1]}, {axis_c[2]}')
        return rot.addOutput('out', 'vector3')

    def _tx_vector_transform(self, n):
        """Vector Transform → transformvector (world↔object↔camera)."""
        vec_out, vec_c = self._socket_vector3(n.inputs.get('Vector'), (0.0, 0.0, 0.0))
        mx_node = self._ng.addNode('transformvector', self._uid('xfvec'), 'vector3')
        v = mx_node.addInput('in', 'vector3')
        if vec_out: self._connect(v, vec_out)
        else: v.setValueString(f'{vec_c[0]}, {vec_c[1]}, {vec_c[2]}')
        mx_node.addInput('fromspace', 'string').setValueString('world')
        mx_node.addInput('tospace',   'string').setValueString('object')
        return mx_node.addOutput('out', 'vector3')

    def _tx_curve_vec(self, n):
        """Vector Curves → identity pass-through (curves need pixel-level eval)."""
        vec_out, vec_c = self._socket_vector3(n.inputs.get('Vector'), (0.0, 0.0, 0.0))
        mul = self._ng.addNode('multiply', self._uid('vcurve'), 'vector3')
        v = mul.addInput('in1', 'vector3')
        if vec_out: self._connect(v, vec_out)
        else: v.setValueString(f'{vec_c[0]}, {vec_c[1]}, {vec_c[2]}')
        mul.addInput('in2', 'vector3').setValueString('1.0, 1.0, 1.0')
        return mul.addOutput('out', 'vector3')

    def _tx_normal_const(self, n):
        """Normal node (explicit normal input) → normalmap with constant."""
        norm_out, norm_c = self._socket_vector3(n.inputs.get('Normal'), (0.0, 0.0, 1.0))
        nm = self._ng.addNode('normalmap', self._uid('nconst'), 'vector3')
        v = nm.addInput('in', 'color3')
        val = norm_c if norm_c else (0.5, 0.5, 1.0)
        v.setValueString(f'{val[0]*0.5+0.5}, {val[1]*0.5+0.5}, {val[2]*0.5+0.5}')
        return nm.addOutput('out', 'vector3')

    def _tx_tangent(self, n):
        """Tangent → geometric tangent vector."""
        tc = self._ng.addNode('tangent', self._uid('tang'), 'vector3')
        tc.addInput('space', 'string').setValueString('world')
        return tc.addOutput('out', 'vector3')

    # ------------------------------------------------------------------ #
    #  Math / Converter nodes                                               #
    # ------------------------------------------------------------------ #

    def _tx_clamp(self, n):
        """Clamp → ND_clamp."""
        val_out, val_c = self._socket_float(n.inputs.get('Value') or n.inputs[0], 0.5)
        mn  = float(n.inputs['Min'].default_value)  if 'Min'  in n.inputs else 0.0
        mx_ = float(n.inputs['Max'].default_value)  if 'Max'  in n.inputs else 1.0
        cl = self._ng.addNode('clamp', self._uid('clamp'), 'float')
        v = cl.addInput('in', 'float')
        if val_out: self._connect(v, val_out)
        else: v.setValue(float(val_c))
        cl.addInput('low',  'float').setValue(mn)
        cl.addInput('high', 'float').setValue(mx_)
        return cl.addOutput('out', 'float')

    def _tx_map_range(self, n):
        """Map Range → ND_remap."""
        val_out, val_c   = self._socket_float(n.inputs.get('Value') or n.inputs[0], 0.5)
        from_min = float(n.inputs['From Min'].default_value) if 'From Min' in n.inputs else 0.0
        from_max = float(n.inputs['From Max'].default_value) if 'From Max' in n.inputs else 1.0
        to_min   = float(n.inputs['To Min'].default_value)   if 'To Min'   in n.inputs else 0.0
        to_max   = float(n.inputs['To Max'].default_value)   if 'To Max'   in n.inputs else 1.0
        remap = self._ng.addNode('remap', self._uid('remap'), 'float')
        v = remap.addInput('in', 'float')
        if val_out: self._connect(v, val_out)
        else: v.setValue(float(val_c))
        remap.addInput('inlow',  'float').setValue(from_min)
        remap.addInput('inhigh', 'float').setValue(from_max)
        remap.addInput('outlow', 'float').setValue(to_min)
        remap.addInput('outhigh','float').setValue(to_max)
        return remap.addOutput('out', 'float')

    # ------------------------------------------------------------------ #
    #  Input nodes                                                          #
    # ------------------------------------------------------------------ #

    def _tx_rgb_input(self, n):
        """RGB node → constant color3."""
        dv = n.outputs[0].default_value if n.outputs else (0.5, 0.5, 0.5, 1.0)
        return self._const_color3(float(dv[0]), float(dv[1]), float(dv[2]))

    def _tx_value_input(self, n):
        """Value node → constant float."""
        dv = n.outputs[0].default_value if n.outputs else 0.5
        return self._const_float(float(dv))

    def _tx_vertex_color(self, n):
        """Vertex Color → geomcolor node."""
        vc = self._ng.addNode('geomcolor', self._uid('vcol'), 'color3')
        return vc.addOutput('out', 'color3')

    def _tx_attribute(self, n):
        """Attribute → geomcolor (best approximation for a named attribute)."""
        attr_name = n.attribute_name if hasattr(n, 'attribute_name') else ''
        vc = self._ng.addNode('geomcolor', self._uid('attr'), 'color3')
        if attr_name:
            vc.addInput('index', 'integer').setValue(0)
        return vc.addOutput('out', 'color3')

    def _tx_fresnel(self, n):
        """Fresnel → artistic_ior facing-ratio approximation."""
        ior_out, ior_c = self._socket_float(n.inputs.get('IOR'), 1.5)
        # Schlick approximation: F0 = ((ior-1)/(ior+1))^2
        ior_val = float(ior_c) if ior_c is not None else 1.5
        f0 = ((ior_val - 1.0) / (ior_val + 1.0)) ** 2
        fr = self._ng.addNode('fresnel', self._uid('fres'), 'float')
        fr.addInput('ior', 'float').setValue(ior_val)
        return fr.addOutput('out', 'float')

    def _tx_layer_weight(self, n):
        """Layer Weight → fresnel (facing / blend output)."""
        blend = float(n.inputs['Blend'].default_value) if 'Blend' in n.inputs else 0.5
        ior = 1.0 / max(1e-6, blend) if blend > 0 else 1.5
        fr = self._ng.addNode('fresnel', self._uid('lw'), 'float')
        fr.addInput('ior', 'float').setValue(ior)
        return fr.addOutput('out', 'float')

    def _tx_geometry(self, n):
        """New Geometry → world-space normal vector."""
        geo = self._ng.addNode('normal', self._uid('geo'), 'vector3')
        geo.addInput('space', 'string').setValueString('world')
        return geo.addOutput('out', 'vector3')


def _export_materialx_graphs(out_dir: str):
    """
    Walk all materials in the scene.  For each Principled BSDF material,
    build a MaterialX document from the full Blender node tree and write
    <out_dir>/<material_name>.mtlx.
    """
    try:
        import MaterialX as mx
    except ImportError:
        print("  [mtlx export] MaterialX not available — skipping.")
        return

    import os

    # Path to MaterialX standard libraries shipped with Blender
    mx_lib_path = os.path.dirname(mx.__file__)

    os.makedirs(out_dir, exist_ok=True)

    # Remove stale outputs from previous runs so old node representations
    # (e.g. separate3 before it was replaced with extract) don't persist.
    for fname in os.listdir(out_dir):
        if fname.endswith(('.mtlx', '.osl', '.h')):
            try:
                os.remove(os.path.join(out_dir, fname))
            except OSError:
                pass

    exported = 0
    skipped  = 0

    print()
    print(f"  Exporting MaterialX graphs → {out_dir}")

    for mat in bpy.data.materials:
        if not mat.use_nodes or not mat.node_tree:
            skipped += 1
            continue
        bsdf = next((n for n in mat.node_tree.nodes
                     if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is None:
            skipped += 1
            continue
        try:
            builder = _MtlxBuilder(mat, out_dir, mx_lib_path)
            out_path = builder.build()
            if out_path:
                exported += 1
            else:
                skipped += 1
        except Exception as e:
            import traceback
            print(f"  [mtlx export] ERROR on material '{mat.name}': {e}")
            traceback.print_exc()
            skipped += 1

    print(f"  [mtlx export] Exported {exported} material(s), skipped {skipped}.")

    # Generate OSL source from each .mtlx file
    if exported > 0:
        _generate_osl_from_mtlx_dir(out_dir, mx_lib_path)

    print("=" * 60)


def _generate_osl_from_mtlx_dir(mtlx_dir: str, mx_lib_path: str):
    """
    For each .mtlx file in mtlx_dir, run the MaterialX OSL shader generator
    to produce a .osl source file alongside it.  The generated .osl calls into
    the MaterialX standard OSL implementations (mx_*.osl) that ship with Blender.
    """
    import os
    try:
        import MaterialX as mx
        # GenShader and GenOsl are separate .so modules inside the MaterialX package
        from MaterialX import PyMaterialXGenShader as mxgen  # noqa: F401
        from MaterialX import PyMaterialXGenOsl as mxosl
    except ImportError as e:
        print(f"  [osl gen] PyMaterialXGenOsl not available — skipping ({e}).")
        return

    search_path = mx.FileSearchPath(mx_lib_path)

    # Load the stdlib once; all per-material docs import it for validation
    stdlib = mx.createDocument()
    try:
        mx.loadLibraries(mx.getDefaultDataLibraryFolders(), search_path, stdlib)
    except Exception as e:
        print(f"  [osl gen] Could not load MaterialX stdlib: {e}")
        return

    gen = mxosl.OslShaderGenerator.create()
    ctx = mxgen.GenContext(gen)
    ctx.registerSourceCodeSearchPath(search_path)

    osl_exported = 0
    osl_errors   = 0
    # Maps mtlx filename → raw generated OSL source, collected before writing
    generated: list = []  # [(osl_path, osl_source), ...]

    for fname in sorted(os.listdir(mtlx_dir)):
        if not fname.endswith('.mtlx'):
            continue
        mtlx_path = os.path.join(mtlx_dir, fname)
        osl_path  = mtlx_path[:-5] + '.osl'

        try:
            doc = mx.createDocument()
            doc.importLibrary(stdlib)
            mx.readFromXmlFile(doc, mtlx_path, search_path)

            valid, msg = doc.validate()
            if not valid:
                print(f"  [osl gen] WARNING: {fname} validation errors: {msg}")

            mat_nodes = [n for n in doc.getNodes()
                         if n.getCategory() == 'surfacematerial']
            if not mat_nodes:
                print(f"  [osl gen] No surfacematerial in {fname} — skipping.")
                osl_errors += 1
                continue

            shader = gen.generate(mat_nodes[0].getName(), mat_nodes[0], ctx)
            osl_source = shader.getStage(mxgen.PIXEL_STAGE).getSourceCode()
            generated.append((osl_path, osl_source))

        except Exception as e:
            import traceback
            print(f"  [osl gen] ERROR generating OSL for {fname}: {e}")
            traceback.print_exc()
            osl_errors += 1

    if not generated:
        print(f"  [osl gen] Generated 0 OSL shader(s)"
              + (f", {osl_errors} error(s)." if osl_errors else "."))
        return

    # ------------------------------------------------------------------
    # Extract shared boilerplate into _mx_stdlib.h.
    #
    # The MaterialX OSL generator inlines stdosl.h + all helper functions
    # before the final `shader MaterialName(...)` block.  That block is
    # always introduced by `shader ` at the start of a line — and it is
    # always the LAST such occurrence in the file (helpers are plain
    # functions, not shaders).
    #
    # We split each file at that boundary, write the first file's
    # boilerplate as the shared header, then rewrite every file as just:
    #   #include "_mx_stdlib.h"
    #   <shader block>
    # ------------------------------------------------------------------
    import re as _re

    def _split_osl(src):
        """Return (boilerplate, shader_body) by splitting before the last
        line-starting `shader ` token."""
        matches = list(_re.finditer(r'(?m)^shader ', src))
        if not matches:
            return '', src
        last = matches[-1]
        cut = last.start()
        return src[:cut], src[cut:]

    # Write the first file's boilerplate as the shared header
    first_boilerplate, _ = _split_osl(generated[0][1])
    shared_header = None
    if first_boilerplate.strip():
        header_path = os.path.join(mtlx_dir, '_mx_stdlib.h')

        # Collect mx_*.osl function bodies from the MaterialX genosl directory.
        # These files contain function definitions (e.g. mx_fractal3d_float,
        # mx_heighttonormal_vector3) that the generator calls but does not
        # inline.  We need to append them to the shared header so oslc can
        # find them when compiling each per-material shader.
        genosl_extras = []
        genosl_dir = os.path.join(mx_lib_path, 'libraries', 'stdlib', 'genosl')
        genosl_file_count = 0

        if os.path.isdir(genosl_dir):
            import re as _re2

            # Helper: strip all #include lines from OSL source.
            def _strip_includes(src):
                return ''.join(
                    ln for ln in src.splitlines(keepends=True)
                    if not ln.lstrip().startswith('#include')
                )

            # Helper: extract function/struct names defined at top level.
            # Covers all OSL primitives + Imath/MaterialX extended types.
            _DEF_RE = _re2.compile(
                r'^(?:void|float|int|string'
                r'|color[234]?|vector[234]?|point|normal|matrix(?:33|44)?'
                r'|closure\s+color|struct'
                r'|MATERIAL|surfaceshader|displacementshader)\s+(\w+)\s*[({]',
                _re2.MULTILINE)

            def _defined_names(src):
                return set(_DEF_RE.findall(src))

            # Names already defined in the boilerplate — skip these to avoid
            # redefinition warnings/errors.
            already_defined = _defined_names(first_boilerplate)

            # Read all mx_*.osl files from the genosl root.
            genosl_contents = {}  # name -> raw text
            for gname in os.listdir(genosl_dir):
                if gname.startswith('mx_') and gname.endswith('.osl'):
                    gpath = os.path.join(genosl_dir, gname)
                    with open(gpath, 'r', errors='replace') as gf:
                        genosl_contents[gname] = gf.read()

            # Also read lib/ helpers (mx_transform_uv etc.).
            genosl_lib_contents = {}
            genosl_lib_dir = os.path.join(genosl_dir, 'lib')
            if os.path.isdir(genosl_lib_dir):
                for lname in os.listdir(genosl_lib_dir):
                    if lname.endswith('.osl'):
                        lpath = os.path.join(genosl_lib_dir, lname)
                        with open(lpath, 'r', errors='replace') as lf:
                            genosl_lib_contents[lname] = lf.read()

            # Build dependency graph from #include "mx_*.osl" directives
            # (e.g. mx_burn_color3.osl includes mx_burn_float.osl).
            def _cross_deps(name, contents_dict):
                src = contents_dict.get(name, '')
                return [
                    d for d in _re2.findall(
                        r'^#include\s+"(mx_[^"]+\.osl)"', src, _re2.MULTILINE)
                    if d in contents_dict
                ]

            # Topological sort via DFS so dependencies come before dependents.
            def _topo_sort(names, contents_dict):
                result, visited, in_stack = [], set(), set()
                def visit(n):
                    if n in in_stack or n in visited:
                        return
                    in_stack.add(n)
                    for dep in _cross_deps(n, contents_dict):
                        visit(dep)
                    in_stack.discard(n)
                    visited.add(n)
                    result.append(n)
                for n in sorted(names):
                    visit(n)
                return result

            # Append a file's stripped body if it introduces new function names.
            def _append_if_new(name, src, label):
                nonlocal genosl_file_count
                detected = _defined_names(src)
                new_names = detected - already_defined
                if detected and not new_names:
                    return  # all functions already covered by the boilerplate
                # If detected is empty (regex didn't match the type), include
                # the file anyway to be safe — don't silently drop it.
                already_defined.update(new_names)
                stripped = _strip_includes(src)
                genosl_extras.append(f'// --- {label} ---\n')
                genosl_extras.append(stripped)
                if not stripped.endswith('\n'):
                    genosl_extras.append('\n')
                genosl_extras.append('\n')
                genosl_file_count += 1

            # 1. lib/ helpers first (mx_transform_uv etc.)
            for lname in sorted(genosl_lib_contents):
                _append_if_new(lname, genosl_lib_contents[lname], f'lib/{lname}')

            # 2. mx_*.osl in topological order (deps before dependents)
            for gname in _topo_sort(genosl_contents, genosl_contents):
                _append_if_new(gname, genosl_contents[gname], gname)

            print(f"  [osl gen] Appended {genosl_file_count} genosl function file(s) "
                  f"from {genosl_dir}")

        # NG_place2d_vector2 is defined as a nodegraph in stdlib_ng.mtlx but
        # has no native .osl file in genosl/.  The MaterialX OSL code generator
        # emits calls to this function but never writes the body.  We provide
        # a hand-written OSL implementation here so it is always available.
        place2d_impl = r"""
// ---- NG_place2d_vector2 (from stdlib_ng.mtlx, no genosl counterpart) ----
vector2 mx_rotate2d(vector2 v, float degrees) {
    float r = radians(degrees);
    float c = cos(r), s = sin(r);
    return vector2(v.x*c - v.y*s, v.x*s + v.y*c);
}
void NG_place2d_vector2(
    vector2 texcoord, vector2 pivot, vector2 scale,
    float rotate, vector2 offset, int operationorder,
    output vector2 result)
{
    vector2 p = texcoord - pivot;
    if (operationorder == 0) {
        // SRT: scale -> rotate -> translate
        float sx = (scale.x != 0.0) ? p.x / scale.x : p.x;
        float sy = (scale.y != 0.0) ? p.y / scale.y : p.y;
        vector2 ro = mx_rotate2d(vector2(sx, sy), rotate);
        result = ro - offset + pivot;
    } else {
        // TRS: translate -> rotate -> scale
        vector2 tr = p - offset;
        vector2 ro = mx_rotate2d(tr, rotate);
        float rx = (scale.x != 0.0) ? ro.x / scale.x : ro.x;
        float ry = (scale.y != 0.0) ? ro.y / scale.y : ro.y;
        result = vector2(rx, ry) + pivot;
    }
}
"""

        with open(header_path, 'w') as f:
            f.write(first_boilerplate)
            if genosl_extras:
                f.write('\n// ---- mx_*.osl function definitions ----\n')
                f.writelines(genosl_extras)
            f.write(place2d_impl)
        shared_header = '_mx_stdlib.h'
        print(f"  [osl gen] Boilerplate extracted → {shared_header} "
              f"({len(first_boilerplate.splitlines())} lines)")

    # Rewrite each .osl: strip its own boilerplate, prepend the #include,
    # and inject "Ci = out;" before the final closing brace so the renderer
    # can read the closure via sg.Ci without needing find_symbol().
    def _inject_ci_assignment(shader_body: str) -> str:
        idx = shader_body.rfind('}')
        if idx < 0:
            return shader_body
        return shader_body[:idx] + '    Ci = out;\n' + shader_body[idx:]

    for osl_path, osl_source in generated:
        if shared_header:
            _, shader_body = _split_osl(osl_source)
            shader_body = _inject_ci_assignment(shader_body)
            content = f'#include "{shared_header}"\n\n{shader_body}'
        else:
            content = osl_source
        with open(osl_path, 'w') as f:
            f.write(content)
        osl_exported += 1

    print(f"  [osl gen] Generated {osl_exported} OSL shader(s)"
          + (f", {osl_errors} error(s)." if osl_errors else "."))


# ---------------------------------------------------------------------------
# Sun DistantLight injector
# ---------------------------------------------------------------------------
def _add_sun_light(usd_path, intensity=3.0, angle=0.53,
                   color_rgb=(1.0, 0.95, 0.8),
                   elevation_deg=45.0, azimuth_deg=135.0):
    """
    Append a DistantLight prim named '/World/sun' to the exported USD.

    Direction convention
    ────────────────────
    A USD DistantLight emits along -Z in its local frame.  We rotate it so
    that the light comes *from* the direction described by elevation/azimuth:

        rotateX = -elevation_deg   (tilts up from the horizon; negative because
                                    we're rotating the -Z emission axis upward)
        rotateY = azimuth_deg      (compass heading, 0 = +Z, 90 = +X)
        rotateZ = 0

    This matches the manual patch we applied earlier:
        elevation=44.71, azimuth=135 → rotateXYZ = (-44.71, 135, 0)
    """
    try:
        from pxr import Usd, UsdLux, Sdf, Gf, Vt
    except ImportError:
        print("  [sun] pxr not available — cannot add DistantLight.")
        return

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"  [sun] Could not open stage '{usd_path}'.")
        return

    # Find or create a /World scope to park the light in
    world_path = Sdf.Path("/World")
    world_prim = stage.GetPrimAtPath(world_path)
    if not world_prim or not world_prim.IsValid():
        world_prim = stage.DefinePrim(world_path, "Xform")

    sun_path = world_path.AppendChild("sun")
    # Remove any existing sun so we can recreate cleanly
    existing = stage.GetPrimAtPath(sun_path)
    if existing and existing.IsValid():
        stage.RemovePrim(sun_path)

    sun = UsdLux.DistantLight.Define(stage, sun_path)

    # Intensity / angle / color
    sun.CreateIntensityAttr().Set(float(intensity))
    sun.CreateAngleAttr().Set(float(angle))
    r, g, b = color_rgb
    sun.CreateColorAttr().Set(Gf.Vec3f(float(r), float(g), float(b)))

    # Orientation: rotateXYZ applied as a single xformOp
    rot_x = -float(elevation_deg)
    rot_y = float(azimuth_deg)
    rot_z = 0.0

    prim = sun.GetPrim()
    rot_attr = prim.CreateAttribute(
        "xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3, False)
    rot_attr.Set(Gf.Vec3f(rot_x, rot_y, rot_z))

    order_attr = prim.CreateAttribute(
        "xformOpOrder", Sdf.ValueTypeNames.TokenArray, False)
    order_attr.Set(Vt.TokenArray(["xformOp:rotateXYZ"]))

    stage.Save()
    print(f"  [sun] Added DistantLight '/World/sun' "
          f"(intensity={intensity}, angle={angle}, "
          f"elevation={elevation_deg}°, azimuth={azimuth_deg}°)")


# ---------------------------------------------------------------------------
# DomeLight texture patcher
# ---------------------------------------------------------------------------
def _patch_dome_light_texture(usd_path, out_dir, explicit_sky=""):
    """
    Blender's USD exporter sometimes writes a near-black solid-color EXR as
    the DomeLight texture when the World shader has no proper HDRI (e.g. the
    sky is a mesh sphere, not a World background image node).

    This function:
      1. Inspects the exported USD for a DomeLight with a suspiciously dark
         (average luminance < 0.05) solid-color texture.
      2. If explicit_sky is provided, uses that image directly.
         Otherwise, searches the World shader node tree for a TEX_ENVIRONMENT
         or TEX_IMAGE node that looks like a sky/environment image.
      3. Copies the image to the textures/ dir and rewrites the DomeLight
         texture path in the USD.
    """
    try:
        import bpy
        import shutil
        from pxr import Usd, UsdLux, Sdf
    except ImportError:
        return

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return

    tex_dir = os.path.join(out_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    patched = False
    for prim in stage.Traverse():
        if not prim.IsA(UsdLux.DomeLight):
            continue
        dome = UsdLux.DomeLight(prim)
        ap_attr = dome.GetTextureFileAttr()
        if not ap_attr:
            continue
        ap = ap_attr.Get()
        if not ap:
            continue
        tex_path = ap.resolvedPath or ap.path

        # Check if the texture is a near-black solid-color swatch
        # (Blender names these color_RRGGBB.exr)
        is_black_swatch = False
        fname = os.path.basename(tex_path)
        if fname.startswith('color_') and fname.endswith('.exr'):
            # Parse hex color from filename: color_RRGGBB.exr
            try:
                hex_col = fname[6:12]
                r = int(hex_col[0:2], 16) / 255.0
                g = int(hex_col[2:4], 16) / 255.0
                b = int(hex_col[4:6], 16) / 255.0
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                if lum < 0.05:
                    is_black_swatch = True
                    print(f"  [dome patch] DomeLight '{prim.GetPath()}' has near-black "
                          f"texture '{fname}' (lum={lum:.3f}) — searching World shader...")
            except Exception:
                pass

        if not is_black_swatch:
            continue

        # Use explicit --sky-texture if provided, otherwise search World shaders
        sky_image_path = None
        if explicit_sky and os.path.exists(explicit_sky):
            sky_image_path = os.path.abspath(explicit_sky)
            print(f"  [dome patch] Using explicit sky texture: '{sky_image_path}'")
        else:
            # Search World shader node trees for a sky/environment image
            for world in bpy.data.worlds:
                if not world.use_nodes:
                    continue
                for node in world.node_tree.nodes:
                    if node.type in ('TEX_ENVIRONMENT', 'TEX_SKY', 'TEX_IMAGE'):
                        img = getattr(node, 'image', None)
                        if img and img.filepath:
                            fpath = bpy.path.abspath(img.filepath)
                            if os.path.exists(fpath):
                                sky_image_path = fpath
                                break
                if sky_image_path:
                    break

        if not sky_image_path:
            print(f"  [dome patch] No sky image found — DomeLight stays dark. "
                  f"Re-export with --sky-texture <path> to fix.")
            continue

        # Copy sky image to textures/
        sky_fname = os.path.basename(sky_image_path)
        dst = os.path.join(tex_dir, sky_fname)
        if not os.path.exists(dst):
            shutil.copy2(sky_image_path, dst)

        # Rewrite the DomeLight texture path (relative to USD)
        rel_path = f"./textures/{sky_fname}"
        ap_attr.Set(Sdf.AssetPath(rel_path))
        print(f"  [dome patch] Rewrote DomeLight texture → '{rel_path}'")
        patched = True

    if patched:
        stage.Save()


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


# ---------------------------------------------------------------------------
# Public API — callable from the Blender addon without command-line parsing
# ---------------------------------------------------------------------------

def prepare_scene(bake_dir="."):
    """
    Run all scene preparation steps in-place on the current Blender scene.
    Call this before exporting USD.  The caller is responsible for pushing/
    restoring an undo state if the scene modifications should not be permanent.

    bake_dir — directory where baked texture images will be saved.
    """
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    realize_instances()
    realize_particle_instances()
    convert_to_mesh()
    apply_modifiers()
    apply_transforms()
    bake_unsupported_nodes(bake_dir)
    convert_glass_materials()
    strip_unsupported_custom_props()
    if REMOVE_RENDER_HIDDEN:
        remove_render_hidden()


def post_process_usd(usd_path, sky_texture=""):
    """
    Run USD post-processing steps after Blender's USD exporter has written
    the file.  Patches the DomeLight texture, injects MaterialX textures,
    exports per-material .mtlx files, and writes the JSON sidecar.

    usd_path    — absolute path to the exported .usdc/.usda file.
    sky_texture — optional path to an equirectangular HDRI to use as the
                  DomeLight texture instead of whatever Blender exported.
    """
    if not os.path.exists(usd_path):
        return

    out_dir = os.path.dirname(usd_path)
    _patch_dome_light_texture(usd_path, out_dir, explicit_sky=sky_texture)
    _inject_materialx_textures(usd_path)
    mtlx_dir = os.path.join(out_dir, "materials")
    _export_materialx_graphs(mtlx_dir)
    _extract_materialx_sidecar(usd_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
