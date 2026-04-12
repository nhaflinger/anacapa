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
"""

import bpy
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
# Step 5: Remove render-hidden cutter helpers
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

    # --- Step 5: Remove hidden cutters ---
    if REMOVE_RENDER_HIDDEN:
        log("Step 5: Removing render-hidden helper objects...")
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
    print(f"  Render-hidden objects removed : {n_removed}")

    if hidden_kept:
        print()
        print("  Render-hidden objects with children (kept):")
        for name in hidden_kept:
            print(f"    - {name}")

    if convert_skipped or mod_warnings:
        print()
        print("  Warnings — manual attention required:")
        for w in convert_skipped + mod_warnings:
            print(f"    ! {w}")

    print()
    print("  Known limitations (not handled by this script):")
    print("    - Particle hair / point cloud instances")
    print("    - Volume / VDB objects")
    print("    - Grease Pencil objects")
    print("    - Library-linked objects that cannot be made local")

    # --- Export USD ---
    import os, sys, argparse
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
