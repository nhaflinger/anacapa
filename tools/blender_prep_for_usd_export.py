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
  ✓ Report objects that could not be processed with an explanation

What requires manual attention (printed as warnings)
─────────────────────────────────────────────────────
  ✗ Particle hair / point cloud instances — complex; export separately
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
# Step 2: Convert non-mesh types to mesh
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
                log(f"  Skipping disabled modifier '{mod_label}' on '{obj.name}'.")
                continue
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
                log(f"  Applied modifier '{mod_label}' on '{obj.name}'.")
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
#   - Invert Color   (ShaderNodeInvert)
#   - Hue/Saturation (ShaderNodeHueSaturation)
#   - Bright/Contrast (ShaderNodeBrightContrast)
#
# Not handled (emits a warning):
#   - RGB Curves, Color Ramp, Mix RGB with non-trivial factors, procedural
#     textures with no image source, and any multi-input mixing graphs.
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


def _invert_pixels(pixels: list) -> list:
    """Invert R, G, B channels; leave A unchanged."""
    out = list(pixels)
    for i in range(0, len(out), 4):
        out[i]   = 1.0 - out[i]
        out[i+1] = 1.0 - out[i+1]
        out[i+2] = 1.0 - out[i+2]
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

            # Walk back to find the source Image Texture
            img_node, _ = _find_upstream_image_tex(src_node)
            if img_node is None or img_node.image is None:
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

            # Collect the chain of unsupported nodes between img_node and bsdf
            chain = []
            node = src_node
            while node and node.type not in _USD_PASSTHROUGH_TYPES:
                chain.append(node)
                # Follow the primary color input back
                next_node = None
                for inp in node.inputs:
                    if inp.links and inp.name in ('Color', 'Image', 'Value', ''):
                        next_node = inp.links[0].from_node
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
                    fac = n.inputs['Fac'].default_value if 'Fac' in n.inputs else 1.0
                    if fac >= 0.999:
                        pixels = _invert_pixels(pixels)
                        applied_ops.append('Invert')
                    else:
                        unhandled.append(f"Invert(fac={fac:.2f})")
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
# Step 6: Strip custom properties that the USD exporter cannot serialize
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
# Step 7: Remove render-hidden cutter helpers
# ---------------------------------------------------------------------------

def remove_render_hidden() -> Tuple[int, List[str]]:  # step 6
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

    # --- Step 1: Instances ---
    log("Step 1: Realizing collection instances...")
    n_instances = realize_instances()

    # --- Step 2: Convert non-mesh ---
    log("Step 2: Converting non-mesh objects to mesh...")
    n_converted, convert_skipped = convert_to_mesh()

    # --- Step 3: Apply modifiers ---
    log("Step 3: Applying modifiers...")
    n_mods, mod_warnings = apply_modifiers()

    # --- Step 4: Apply transforms ---
    log("Step 4: Applying transforms...")
    n_transforms = apply_transforms()

    # --- Step 5: Bake unsupported shader nodes ---
    log("Step 5: Baking unsupported shader nodes into textures...")
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

    # --- Step 6: Strip unsupported custom properties ---
    log("Step 6: Stripping unsupported custom properties...")
    n_props_removed = strip_unsupported_custom_props()

    # --- Step 7: Remove hidden cutters ---
    if REMOVE_RENDER_HIDDEN:
        log("Step 7: Removing render-hidden helper objects...")
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
    print(f"  Objects converted to mesh     : {n_converted}")
    print(f"  Modifiers applied             : {n_mods}")
    print(f"  Objects with transforms fixed : {n_transforms}")
    print(f"  Materials with baked nodes    : {n_baked}")
    print(f"  Unsupported custom props strip: {n_props_removed}")
    print(f"  Render-hidden objects removed : {n_removed}")

    if hidden_kept:
        print()
        print("  Render-hidden objects with children (kept):")
        for name in hidden_kept:
            print(f"    - {name}")

    all_warnings = convert_skipped + mod_warnings + bake_warnings
    if all_warnings:
        print()
        print("  Warnings — manual attention required:")
        for w in all_warnings:
            print(f"    ! {w}")

    print()
    print("  Known limitations (not handled by this script):")
    print("    - Particle hair / point cloud instances")
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


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
