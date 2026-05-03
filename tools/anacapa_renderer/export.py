"""
Shared export helpers: USD scene export and anacapa command assembly.
"""

import bpy
import os
import importlib.util

# ---------------------------------------------------------------------------
# Scene dirty tracking
#
# _scene_dirty starts True so the first render always exports.
# depsgraph_update_post sets it True whenever anything changes.
# _suppress_dirty temporarily blocks the handler during prep (which modifies
# the scene itself and would otherwise immediately re-dirty the flag).
# ---------------------------------------------------------------------------
_scene_dirty     = True
_suppress_dirty  = False
_cached_usd_path = None   # path to last successfully exported USD


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    global _scene_dirty
    if not _suppress_dirty:
        _scene_dirty = True


def mark_dirty():
    global _scene_dirty
    _scene_dirty = True


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
    """Load blender_prep_for_usd_export.py as a module (cached after first call)."""
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


def export_usd(filepath, context, run_prep=True):
    """
    Optionally run scene prep, export USD, then run USD post-processing.
    Skips prep + export entirely if the scene has not changed since the last
    export and a cached USD already exists.
    """
    global _scene_dirty, _suppress_dirty, _cached_usd_path

    # If nothing changed and we have a cached USD, reuse it
    if not _scene_dirty and _cached_usd_path and os.path.exists(_cached_usd_path):
        if filepath != _cached_usd_path:
            import shutil
            shutil.copy2(_cached_usd_path, filepath)
        print("[Anacapa] Scene unchanged — reusing cached USD")
        return

    prep = _load_prep_module() if run_prep else None
    bake_dir = os.path.dirname(filepath)

    if prep is not None:
        _suppress_dirty = True
        try:
            prep.prepare_scene(bake_dir=bake_dir)
        except Exception as e:
            print(f"[Anacapa] Scene prep warning: {e}")
        finally:
            _suppress_dirty = False

    window = context.window_manager.windows[0]
    with context.temp_override(window=window):
        bpy.ops.wm.usd_export(
            filepath=filepath,
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

    if prep is not None:
        try:
            prep.post_process_usd(filepath)
        except Exception as e:
            print(f"[Anacapa] USD post-process warning: {e}")

    _scene_dirty     = False
    _cached_usd_path = filepath


def build_command(executable, usd_path, settings, width, height, output_path):
    """
    Assemble the anacapa CLI command as a list of strings.
    Returns (cmd_list, png_path_or_None).
    """
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
