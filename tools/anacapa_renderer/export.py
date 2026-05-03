"""
Shared export helpers: USD scene export and anacapa command assembly.

A single cached USD file is maintained in bpy.app.tempdir.
The dirty tracker determines whether a full re-export is needed:

  geometry/material change  → run prep + full export
  transform-only change     → skip prep, re-export USD (fast)
  nothing changed           → reuse cached USD entirely

This avoids re-running the expensive prep script (modifier baking,
Glass BSDF conversion, etc.) when only transforms or camera changed.
"""

import bpy
import os
import importlib.util
import shutil

# ---------------------------------------------------------------------------
# Persistent state — stored in bpy.app.driver_namespace so it survives
# module reloads within a Blender session.
# ---------------------------------------------------------------------------
_NS = "anacapa_export_state"

def _state():
    if _NS not in bpy.app.driver_namespace:
        bpy.app.driver_namespace[_NS] = {
            "dirty_scene":     True,
            "dirty_transform": True,
            "suppress_dirty":  False,
            "cached_usd_path": None,
        }
    return bpy.app.driver_namespace[_NS]


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    s = _state()
    if s["suppress_dirty"]:
        return
    for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.Material):
            s["dirty_scene"] = True
        elif isinstance(update.id, bpy.types.Object):
            obj = update.id
            if update.is_updated_geometry:
                s["dirty_scene"] = True
            if update.is_updated_shading and obj.type == 'MESH':
                s["dirty_scene"] = True
            if update.is_updated_transform:
                s["dirty_transform"] = True
        elif isinstance(update.id, bpy.types.World):
            s["dirty_scene"] = True


def mark_all_dirty():
    s = _state()
    s["dirty_scene"]     = True
    s["dirty_transform"] = True


def register_dirty_handler():
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister_dirty_handler():
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)


# ---------------------------------------------------------------------------
# Prep module loader
# ---------------------------------------------------------------------------

def _load_prep_module():
    if _load_prep_module._mod is not None:
        return _load_prep_module._mod
    script_path = os.path.join(os.path.dirname(__file__),
                               "blender_prep_for_usd_export.py")
    if not os.path.exists(script_path):
        return None
    spec = importlib.util.spec_from_file_location("anacapa_prep", script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _load_prep_module._mod = mod
    return mod

_load_prep_module._mod = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_executable(context):
    prefs = context.preferences.addons[__package__].preferences
    return prefs.executable_path


def _usd_export(filepath, context, **kwargs):
    window = context.window_manager.windows[0]
    with context.temp_override(window=window):
        bpy.ops.wm.usd_export(filepath=filepath, **kwargs)


def export_usd(usd_path, context, run_prep=True):
    """
    Export the scene to a single USD file.
    Skips prep when only transforms changed.
    Skips export entirely when nothing changed.
    """
    s = _state()

    # Nothing changed — reuse cached USD
    cached = s["cached_usd_path"]
    if not s["dirty_scene"] and not s["dirty_transform"] \
            and cached and os.path.exists(cached):
        if usd_path != cached:
            shutil.copy2(cached, usd_path)
        print("[Anacapa] Scene unchanged — reusing cached USD")
        return

    prep = _load_prep_module() if run_prep else None

    # suppress_dirty should already be set by the caller (operator).
    # We set it here too as a safety net in case export_usd is called directly.
    was_suppressed = s["suppress_dirty"]
    s["suppress_dirty"] = True
    try:
        if s["dirty_scene"] and prep is not None:
            print("[Anacapa] Running scene prep…")
            bake_dir = os.path.dirname(usd_path)
            try:
                prep.prepare_scene(bake_dir=bake_dir)
            except Exception as e:
                print(f"[Anacapa] Scene prep warning: {e}")
        elif not s["dirty_scene"] and s["dirty_transform"]:
            print("[Anacapa] Transform-only change — skipping prep")

        _usd_export(usd_path, context,
            export_animation=True,
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

        if prep is not None and s["dirty_scene"]:
            try:
                prep.post_process_usd(usd_path)
            except Exception as e:
                print(f"[Anacapa] USD post-process warning: {e}")

        s["dirty_scene"]     = False
        s["dirty_transform"] = False
        s["cached_usd_path"] = usd_path

    finally:
        # Only clear suppress if we set it (don't clear if caller set it)
        if not was_suppressed:
            s["suppress_dirty"] = False


def build_command(executable, usd_path, settings, width, height, output_path):
    cmd = [
        executable,
        "--scene",  usd_path,
        "--output", output_path,
        "-W", str(width),
        "-H", str(height),
        "-s", str(settings.samples),
        "-d", str(settings.max_depth),
        "--integrator",    settings.integrator,
        "--tile-size",     str(settings.tile_size),
        "--firefly-clamp", str(settings.firefly_clamp),
    ]

    if settings.interactive:
        cmd.append("--interactive")

    if settings.num_threads > 0:
        cmd += ["-t", str(settings.num_threads)]

    if not settings.adaptive:
        cmd.append("--no-adaptive")
    elif settings.adaptive_base_spp > 0:
        cmd += ["--adaptive-base-spp", str(settings.adaptive_base_spp)]

    env = bpy.path.abspath(settings.env_path) if settings.env_path else ""
    if env:
        cmd += ["--env", env, "--env-intensity", str(settings.env_intensity)]

    if settings.light_angle > 0:
        cmd += ["--light-angle", str(settings.light_angle)]

    if settings.override_lights:
        cmd.append("--override-lights")
    if settings.override_materials:
        cmd.append("--override-materials")

    if settings.fstop > 0 and settings.focus_distance > 0:
        cmd += ["--fstop", str(settings.fstop),
                "--focus-distance", str(settings.focus_distance)]

    if settings.shutter_close > settings.shutter_open:
        cmd += ["--shutter-open",  str(settings.shutter_open),
                "--shutter-close", str(settings.shutter_close)]

    if settings.denoise:
        cmd.append("--denoise")
    if settings.write_aovs:
        cmd.append("--write-aovs")

    if settings.camera_path:
        cmd += ["--camera", settings.camera_path]

    png_path = bpy.path.abspath(settings.png_path) if settings.png_path else None
    if png_path:
        cmd += ["--png", png_path, "--exposure", str(settings.exposure)]

    return cmd, png_path
