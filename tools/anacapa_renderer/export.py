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
import re
import importlib.util
import shutil


def _set_render_display(value):
    """Set Blender's render display type (Blender 4.x+ API)."""
    try:
        bpy.context.preferences.view.render_display_type = value
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Frame-token substitution for output paths
#
# Replaces $F, $F2, $F3, ... in a path with the current frame number.
# $F alone uses no padding ("42"); $Fn pads with leading zeros to width n.
# Mirrors the convention used by Houdini, RV, and Nuke.
# ---------------------------------------------------------------------------
_FRAME_TOKEN_RE = re.compile(r'\$F(\d*)')

def substitute_frame_tokens(path, frame):
    def _sub(m):
        width = int(m.group(1)) if m.group(1) else 0
        return f"{int(frame):0{width}d}" if width else str(int(frame))
    return _FRAME_TOKEN_RE.sub(_sub, path)

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
            "dirty_hair":      True,
            "suppress_dirty":  False,
            "cached_usd_path": None,
            "cached_abc_path": None,
        }
    s = bpy.app.driver_namespace[_NS]
    # Migrate state dicts created before dirty_hair was added
    s.setdefault("dirty_hair",      True)
    s.setdefault("cached_abc_path", None)
    s.setdefault("cached_frame",    None)
    return s


def _obj_has_hair(obj):
    """True if obj is a Curves object or a mesh with a HAIR particle system."""
    if obj.type == 'CURVES':
        return True
    if obj.type == 'MESH':
        return any(ps.settings.type == 'HAIR' for ps in obj.particle_systems)
    return False


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
                if _obj_has_hair(obj):
                    s["dirty_hair"] = True
            if update.is_updated_shading and obj.type == 'MESH':
                s["dirty_scene"] = True
            if update.is_updated_transform:
                s["dirty_transform"] = True
                if _obj_has_hair(obj):
                    s["dirty_hair"] = True
        elif isinstance(update.id, bpy.types.World):
            s["dirty_scene"] = True


def mark_all_dirty():
    s = _state()
    s["dirty_scene"]     = True
    s["dirty_transform"] = True
    s["dirty_hair"]      = True


def register_dirty_handler():
    for handler, lst in [
        (_on_depsgraph_update,  bpy.app.handlers.depsgraph_update_post),
        (_anacapa_render_pre,   bpy.app.handlers.render_pre),
        (_anacapa_render_post,  bpy.app.handlers.render_post),
        (_anacapa_render_cancel, bpy.app.handlers.render_cancel),
        (_anacapa_load_post,    bpy.app.handlers.load_post),
    ]:
        if handler not in lst:
            lst.append(handler)


def unregister_dirty_handler():
    for handler, lst in [
        (_on_depsgraph_update,  bpy.app.handlers.depsgraph_update_post),
        (_anacapa_render_pre,   bpy.app.handlers.render_pre),
        (_anacapa_render_post,  bpy.app.handlers.render_post),
        (_anacapa_render_cancel, bpy.app.handlers.render_cancel),
        (_anacapa_load_post,    bpy.app.handlers.load_post),
    ]:
        if handler in lst:
            lst.remove(handler)


# ---------------------------------------------------------------------------
# Viewer management — shared between operators and the registered engine
# ---------------------------------------------------------------------------
VIEWER_SOCK  = "/tmp/anacapa_viewer.sock"
_viewer_proc = None  # outlives operator instances


def is_viewer_running():
    import socket as _sock
    try:
        s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        s.settimeout(0.3)
        s.connect(VIEWER_SOCK)
        s.close()
        return True
    except Exception:
        return False


def launch_viewer(viewer_path):
    """Kill any stale viewer process, start a fresh one, wait up to 3 s."""
    global _viewer_proc
    import subprocess, time
    if _viewer_proc and _viewer_proc.poll() is None:
        _viewer_proc.terminate()
    _viewer_proc = subprocess.Popen([viewer_path, "--listen"])
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if is_viewer_running():
            return
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Render-handler shared state
# ---------------------------------------------------------------------------
_NS_RENDER = "anacapa_render_pre_data"


def _render_pre_data():
    if _NS_RENDER not in bpy.app.driver_namespace:
        bpy.app.driver_namespace[_NS_RENDER] = {
            'abc_path':        None,
            'matassign_paths': None,
            'hidden_curves':   [],
            'motion_xf':       None,  # pre-collected open/close transforms
        }
    d = bpy.app.driver_namespace[_NS_RENDER]
    d.setdefault('motion_xf', None)
    return d


def _shutter_interval(settings):
    s_open, s_close = 0.0, 0.0
    if getattr(settings, 'use_motion_blur', False):
        shutter  = getattr(settings, 'motion_blur_shutter', 0.5)
        position = getattr(settings, 'motion_blur_position', 'CENTER')
        if shutter > 0:
            if position == 'START':
                s_open, s_close = 0.0, shutter
            elif position == 'END':
                s_open, s_close = -shutter, 0.0
            else:
                s_open, s_close = -shutter / 2.0, shutter / 2.0
    return s_open, s_close


def _collect_motion_xf(scene, s_open, s_close):
    """Collect object transforms at shutter open and close sub-frames.

    When CURVES are present, delegates to a background subprocess (same pattern
    as hair export) so that scene.frame_set() never runs in the main process —
    avoiding the Blender 5.1 SIGSEGV from in-process CURVES evaluation.
    Falls back to direct frame-seeking when no CURVES are in the scene.

    Returns a dict with keys xf_open, xf_close, xf_inst_open, xf_inst_close,
    motion_times, or None on failure.
    """
    import math, json, subprocess, tempfile
    from mathutils import Matrix

    current_frame = scene.frame_current
    t_open  = float(current_frame) + s_open
    t_close = float(current_frame) + s_close

    # Check for VISIBLE CURVES only — by the time render_pre calls this,
    # all CURVES are already hidden (hide_render=True), so the direct
    # scene.frame_set() path is safe and uses the same depsgraph state as
    # export_scene_binary (matching persistent_id keys for collection instances).
    has_curves = any(obj.type == 'CURVES' and not obj.hide_render
                     for obj in scene.objects)

    def _parse_result(data):
        """Reconstruct xf dicts from JSON-serialised matrices."""
        def _to_mat(flat):
            return Matrix([flat[r*4:(r+1)*4] for r in range(4)])

        xf_open      = {k: _to_mat(v) for k, v in data['xf_open'].items()}
        xf_close     = {k: _to_mat(v) for k, v in data['xf_close'].items()}
        # Instance keys were encoded as "parent_name|||p0,p1,..."
        xf_inst_open = {}
        for k, v in data['xf_inst_open'].items():
            pname, pids = k.split('|||')
            pid = tuple(int(x) for x in pids.split(',')) if pids else ()
            xf_inst_open[(pname, pid)] = _to_mat(v)
        xf_inst_close = {}
        for k, v in data['xf_inst_close'].items():
            pname, pids = k.split('|||')
            pid = tuple(int(x) for x in pids.split(',')) if pids else ()
            xf_inst_close[(pname, pid)] = _to_mat(v)
        return xf_open, xf_close, xf_inst_open, xf_inst_close

    if has_curves:
        # Subprocess path: avoids frame_set() in main process.
        blend_path = bpy.data.filepath
        if not blend_path:
            print("[Anacapa] motion xf subprocess: blend file not saved — skipping")
            return None

        blender_exe = bpy.app.binary_path
        out_path    = blend_path + ".motion_xf.json"

        script = f"""\
import bpy, json, math
from mathutils import Matrix

scene = bpy.context.scene

_ZUP_TO_YUP = Matrix((
    (1,  0,  0,  0),
    (0,  0,  1,  0),
    (0, -1,  0,  0),
    (0,  0,  0,  1),
))

def to_yup(m):
    return _ZUP_TO_YUP @ m

def mat_flat(m):
    return [m[r][c] for r in range(4) for c in range(4)]

def collect(t):
    fi = int(math.floor(t))
    scene.frame_set(fi, subframe=t - fi)
    dg = bpy.context.evaluated_depsgraph_get()
    xf = {{}}
    xi = {{}}
    for inst in dg.object_instances:
        obj  = inst.object
        orig = obj.original
        if orig.hide_render:
            continue
        mat = to_yup(inst.matrix_world)
        if not inst.is_instance:
            if obj.type in ('MESH', 'LIGHT', 'CAMERA'):
                xf[orig.name] = mat_flat(mat)
        elif obj.type == 'MESH':
            par = inst.parent
            if par is None:
                continue
            op = par.original if hasattr(par, 'original') else par
            if getattr(op, 'instance_type', '') != 'COLLECTION':
                continue
            pid_str = ','.join(str(x) for x in inst.persistent_id)
            xi[par.name + '|||' + pid_str] = mat_flat(mat)
    return xf, xi

xo, xio = collect({t_open!r})
xc, xic = collect({t_close!r})

import json
with open({out_path!r}, 'w') as fh:
    json.dump({{'xf_open': xo, 'xf_close': xc,
               'xf_inst_open': xio, 'xf_inst_close': xic}}, fh)
"""

        script_path = blend_path + ".motion_xf_eval.py"
        try:
            with open(script_path, 'w') as fh:
                fh.write(script)
            proc = subprocess.run(
                [blender_exe, "--background", blend_path,
                 "--python", script_path],
                capture_output=True, text=True, timeout=60)
            if proc.returncode != 0 or not os.path.exists(out_path):
                print(f"[Anacapa] motion xf subprocess failed "
                      f"(code {proc.returncode}) — mesh motion blur skipped")
                return None
            with open(out_path) as fh:
                data = json.load(fh)
            xf_open, xf_close, xf_inst_open, xf_inst_close = _parse_result(data)
        except Exception as e:
            print(f"[Anacapa] motion xf subprocess error: {e} — skipping")
            return None
        finally:
            for p in (script_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
    else:
        # Direct path: no CURVES, scene.frame_set() is safe.
        scene_mod = _load_scene_export_module()
        if scene_mod is None:
            return None

        def _fs(t):
            fi = int(math.floor(t))
            scene.frame_set(fi, subframe=t - fi)

        try:
            _fs(t_open)
            dg_open = bpy.context.evaluated_depsgraph_get()
            xf_open      = scene_mod._collect_transforms(dg_open)
            xf_inst_open = scene_mod._collect_instance_transforms(dg_open)

            _fs(t_close)
            dg_close = bpy.context.evaluated_depsgraph_get()
            xf_close      = scene_mod._collect_transforms(dg_close)
            xf_inst_close = scene_mod._collect_instance_transforms(dg_close)
        except Exception as e:
            print(f"[Anacapa] motion transform collection warning: {e}")
            return None
        finally:
            scene.frame_set(current_frame)

    return {
        'xf_open':       xf_open,
        'xf_close':      xf_close,
        'xf_inst_open':  xf_inst_open,
        'xf_inst_close': xf_inst_close,
        'motion_times':  [t_open, t_close],
    }


@bpy.app.handlers.persistent
def _anacapa_render_pre(scene):
    if scene.render.engine != 'ANACAPA':
        return
    d = _render_pre_data()
    d['abc_path']        = None
    d['matassign_paths'] = None
    d['hidden_curves']   = []
    d['motion_xf']       = None

    _state()["suppress_dirty"] = True

    settings = scene.anacapa
    s_open, s_close = _shutter_interval(settings)

    ctx = bpy.context

    # Detect hair objects BEFORE hiding CURVES (get_hair_objects filters by hide_render).
    hair_objs = get_hair_objects(ctx)

    # Hide CURVES immediately — before any subprocess or frame_set call.
    # The close-frame subprocess in export_hair_abc releases the GIL; Blender's
    # background evaluation thread can then touch CURVES data → SIGSEGV.
    # Hiding with both flags suppresses viewport and render evaluation.
    # Hair export receives the pre-found hair_objs list so it doesn't need to
    # re-detect (which would return empty now that hide_render is True).
    hidden = []
    for obj in scene.objects:
        if obj.type == 'CURVES':
            was_vp     = obj.hide_viewport
            was_render = obj.hide_render
            obj.hide_viewport = True
            obj.hide_render   = True
            obj["anacapa_hidden_for_render"] = True
            hidden.append((obj.name, was_vp, was_render))
    d['hidden_curves'] = hidden
    if hidden:
        print(f"[Anacapa] render_pre: hiding {len(hidden)} CURVES objects "
              f"(viewport+render) to prevent Blender 5.1 SIGSEGV")

    # Hair export — subprocesses are now safe because CURVES are hidden.
    if hair_objs:
        blend_path = bpy.data.filepath
        if blend_path:
            frame     = scene.frame_current
            cache_dir = get_cache_dir(blend_path)
            abc_path  = os.path.join(cache_dir, f"hair.{frame:04d}.abc")
            try:
                matassign_paths = export_hair_abc(abc_path, ctx,
                                                   shutter_open=s_open,
                                                   shutter_close=s_close,
                                                   hair_objs=hair_objs)
                d['abc_path']        = abc_path
                d['matassign_paths'] = matassign_paths
                print(f"[Anacapa] render_pre: hair → {abc_path}")
            except Exception as e:
                print(f"[Anacapa] render_pre hair export warning: {e}")

    # Pre-collect mesh/light/camera transforms at shutter open and close.
    # When CURVES are present, _collect_motion_xf uses a background subprocess
    # so scene.frame_set() never runs in the main process (avoids SIGSEGV).
    if s_open < s_close - 1e-5:
        d['motion_xf'] = _collect_motion_xf(scene, s_open, s_close)
        if d['motion_xf']:
            print(f"[Anacapa] render_pre: motion transforms collected "
                  f"(open={s_open:+.3f}, close={s_close:+.3f})")


def _restore_hidden_curves(scene):
    d = _render_pre_data()
    for entry in d.get('hidden_curves', []):
        name = entry[0]
        was_vp     = entry[1]
        was_render = entry[2] if len(entry) > 2 else False
        obj = scene.objects.get(name)
        if obj:
            obj.hide_viewport = was_vp
            obj.hide_render   = was_render
            obj.pop("anacapa_hidden_for_render", None)
    d['hidden_curves'] = []


@bpy.app.handlers.persistent
def _anacapa_render_post(scene):
    if scene.render.engine != 'ANACAPA':
        return
    _restore_hidden_curves(scene)
    def _reenable():
        _state()["suppress_dirty"] = False
    bpy.app.timers.register(_reenable, first_interval=2.0)


@bpy.app.handlers.persistent
def _anacapa_render_cancel(scene):
    if scene.render.engine != 'ANACAPA':
        return
    _restore_hidden_curves(scene)
    _state()["suppress_dirty"] = False


@bpy.app.handlers.persistent
def _anacapa_load_post(*args, **kwargs):
    """Restore CURVES left hidden by a crash; sync display_mode with use_viewer."""
    for scene in bpy.data.scenes:
        # Restore any CURVES left hidden by a crashed render.
        for obj in scene.objects:
            if obj.get("anacapa_hidden_for_render"):
                obj.hide_viewport = False
                obj.hide_render   = False
                obj.pop("anacapa_hidden_for_render", None)
        # Sync render display type so F12 respects use_viewer after file load.
        if scene.render.engine == 'ANACAPA':
            settings = getattr(scene, 'anacapa', None)
            if settings:
                try:
                    bpy.context.preferences.view.render_display_type = (
                        'NONE' if getattr(settings, 'use_viewer', False) else 'WINDOW')
                except Exception:
                    pass


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


def get_viewer_executable(context):
    """Viewer binary lives next to the anacapa binary."""
    import os
    anacapa = get_executable(context)
    return os.path.join(os.path.dirname(os.path.abspath(anacapa)), "viewer")


def get_scenes_dir(blend_path):
    """The scenes directory — the directory that contains the blend file.

    USD scene files, materials, and textures all live here:
        scenes/
          char.blend
          char_scene.usdc
          materials/
          textures/
          caches/
            char/
              hair.0001.abc
              char.matassign.json
    """
    scenes_dir = os.path.dirname(os.path.abspath(blend_path))
    os.makedirs(os.path.join(scenes_dir, "materials"), exist_ok=True)
    os.makedirs(os.path.join(scenes_dir, "textures"),  exist_ok=True)
    return scenes_dir


def get_cache_dir(blend_path):
    """Return (and create) the per-blend Alembic / matassign cache directory.

    Resolution order:
      1. $ANACAPA_CACHE_DIR/<blend_stem>/  — pipeline override (farm, NAS, etc.)
      2. <scenes_dir>/caches/<blend_stem>/ — default, alongside the scene files
    """
    blend_stem   = os.path.splitext(os.path.basename(blend_path))[0]
    env_override = os.environ.get("ANACAPA_CACHE_DIR", "").strip()
    if env_override:
        cache_dir = os.path.join(env_override, blend_stem)
    else:
        cache_dir = os.path.join(get_scenes_dir(blend_path), "caches", blend_stem)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


# Keep the old name as an alias so any external code referencing it still works.
get_project_dir = get_cache_dir


def get_hair_objects(context):
    """Return all renderable hair/curve objects in the scene.

    Uses hide_render (the camera icon) rather than viewport visibility so that
    objects temporarily hidden via hide_viewport — as this addon does to prevent
    the Blender 5.1 hair-nodes crash during rendering — are still exported.
    """
    vl_objects = set(context.view_layer.objects)
    objs = []
    for obj in context.scene.objects:
        if obj not in vl_objects:
            continue
        if obj.hide_render:
            continue
        if _obj_has_hair(obj):
            objs.append(obj)
    return objs


def _frame_subframe(total):
    """Split a fractional frame number into (int_frame, subframe) for frame_set."""
    import math
    int_frame = int(math.floor(total))
    sub = total - int_frame
    return int_frame, sub


def export_hair_abc(abc_path, context, shutter_open=0.0, shutter_close=0.0,
                    hair_objs=None):
    """
    Export all visible hair/curve objects to an Alembic file.

    When shutter_close > shutter_open (motion blur enabled), exports a second
    hair snapshot at frame + shutter_close and passes it to hair_export as
    --close, producing a 2-sample Alembic file for velocity motion blur.

    Blender 5.1 crashes (SIGSEGV) on any in-process access to hair-nodes
    Curves data — including obj.data.attributes, alembic_export, and
    save_as_mainfile.  The only crash-safe strategy is to delegate ALL
    Curves data access to a --background subprocess.  If that subprocess
    also crashes, it only kills the child process and main Blender is
    unaffected; hair is simply omitted from the render.

    Requires the blend file to be saved (bpy.data.filepath must be set)
    so the subprocess can open it.  The subprocess writes a compact binary
    strand file; the standalone hair_export tool then converts it to .abc.

    Skips the export entirely when dirty_hair is False (nothing changed).
    """
    import subprocess

    s = _state()
    s.setdefault("cached_shutter_close", None)

    # Invalidate caches when the frame changes — particles, animation, and hair
    # are all frame-dependent.
    frame = context.scene.frame_current
    if s["cached_frame"] != frame:
        s["dirty_scene"]  = True
        s["dirty_hair"]   = True
        s["cached_frame"] = frame
    if s["cached_shutter_close"] != shutter_close:
        s["dirty_hair"]           = True
        s["cached_shutter_close"] = shutter_close

    cached = s["cached_abc_path"]
    if not s["dirty_hair"] and cached and os.path.exists(cached):
        if abc_path != cached:
            shutil.copy2(cached, abc_path)
        print("[Anacapa] Hair unchanged — reusing cached Alembic")
        # Return both matassign layers so the caller can pass --matassign for each.
        blend_path = bpy.data.filepath
        if blend_path:
            blend_stem    = os.path.splitext(os.path.basename(blend_path))[0]
            proj          = get_project_dir(blend_path)
            base_p     = os.path.join(proj, blend_stem + ".matassign.json")
            override_p = os.path.join(proj, blend_stem + ".override.matassign.json")
            paths = [p for p in (base_p, override_p) if os.path.exists(p)]
            return paths if paths else None
        return None

    if hair_objs is None:
        hair_objs = get_hair_objects(context)
    if not hair_objs:
        return None

    # Locate hair_export tool — expected next to the anacapa executable
    executable  = get_executable(context)
    hair_export = os.path.join(os.path.dirname(executable), "hair_export")
    if not os.path.exists(hair_export):
        print(f"[Anacapa] hair_export not found at {hair_export} "
              f"— rebuild with ANACAPA_ENABLE_ALEMBIC=ON")
        return None

    # A saved blend file is required so the subprocess can open it.
    blend_path = bpy.data.filepath
    if not blend_path:
        print("[Anacapa] Hair export requires the blend file to be saved first. "
              "Save the file (Ctrl+S) and render again.")
        return None

    hair_names  = [obj.name for obj in hair_objs]
    bin_path    = abc_path.replace(".abc", ".hairbin")
    counts_path = abc_path.replace(".abc", "_counts.json")
    eval_py  = abc_path.replace(".abc", "_eval.py")
    guide_py = abc_path.replace(".abc", "_guide.py")

    do_motion_blur = shutter_close > shutter_open
    close_bin_path = abc_path.replace(".abc", ".close.hairbin")
    close_eval_py  = abc_path.replace(".abc", "_close_eval.py")
    close_guide_py = abc_path.replace(".abc", "_close_guide.py")

    # Shared strand-writing body used by both helpers (template, no depsgraph line)
    # Format: AHAIR003 — per strand: num_points u32, r g b f32×3, u v f32×2, then n*(x y z radius f32×4)
    _STRAND_BODY = """\
        pos_attr = curves_data.attributes.get("position")
        if pos_attr is None:
            continue
        rad_attr = curves_data.attributes.get("radius")

        positions = [0.0] * (num_points_total * 3)
        pos_attr.data.foreach_get("vector", positions)

        radii = [0.005] * num_points_total
        if rad_attr is not None:
            try:
                rad_attr.data.foreach_get("value", radii)
            except Exception:
                pass

        sc = mw.to_scale()
        radius_scale = (abs(sc.x) + abs(sc.y) + abs(sc.z)) / 3.0

        offsets = [0] * (num_curves + 1)
        try:
            curves_data.curve_offset_data.foreach_get("value", offsets)
        except Exception:
            off = 0
            for ci, curve in enumerate(curves_data.curves):
                offsets[ci] = off
                off += curve.points_length
            offsets[num_curves] = off

        # Read per-strand color from CURVE-domain attribute (try common names)
        # Falls back to per-point domain using the first point's color, or white.
        curve_colors = None  # list of (r, g, b) per curve index
        for col_name in ("Col", "col", "color", "Color", "hair_color"):
            col_attr = curves_data.attributes.get(col_name)
            if col_attr is None:
                continue
            n_curves  = num_curves
            n_points  = num_points_total
            is_curve  = col_attr.domain == "CURVE"
            is_point  = col_attr.domain == "POINT"
            if not is_curve and not is_point:
                continue
            count = n_curves if is_curve else n_points
            # BYTE_COLOR and FLOAT_COLOR both expose 4 floats (RGBA) via foreach_get("color")
            raw = [1.0] * (count * 4)
            try:
                col_attr.data.foreach_get("color", raw)
                if is_curve:
                    curve_colors = [(raw[ci*4], raw[ci*4+1], raw[ci*4+2]) for ci in range(n_curves)]
                else:
                    curve_colors = [(raw[offsets[ci]*4], raw[offsets[ci]*4+1], raw[offsets[ci]*4+2])
                                    for ci in range(n_curves)]
            except Exception:
                # Fallback: try vector (RGB without alpha)
                try:
                    raw3 = [1.0] * (count * 3)
                    col_attr.data.foreach_get("vector", raw3)
                    if is_curve:
                        curve_colors = [(raw3[ci*3], raw3[ci*3+1], raw3[ci*3+2]) for ci in range(n_curves)]
                    else:
                        curve_colors = [(raw3[offsets[ci]*3], raw3[offsets[ci]*3+1], raw3[offsets[ci]*3+2])
                                        for ci in range(n_curves)]
                except Exception:
                    pass
            if curve_colors is not None:
                break

        # Read per-strand root UV from surface_uv_coordinate (CURVE domain, float2)
        curve_root_uvs = None
        uv_attr = curves_data.attributes.get("surface_uv_coordinate")
        if uv_attr is not None and uv_attr.domain == "CURVE":
            raw_uv = [0.0] * (num_curves * 2)
            try:
                uv_attr.data.foreach_get("vector", raw_uv)
                curve_root_uvs = [(raw_uv[ci*2], raw_uv[ci*2+1]) for ci in range(num_curves)]
            except Exception:
                pass

        obj_strand_count = 0
        for ci in range(num_curves):
            start = offsets[ci]
            end   = offsets[ci + 1]
            n     = end - start
            if n < 2:
                continue
            cr, cg, cb = curve_colors[ci] if curve_colors else (1.0, 1.0, 1.0)
            ru, rv     = curve_root_uvs[ci] if curve_root_uvs else (0.0, 0.0)
            fh.write(struct.pack("<I", n))
            fh.write(struct.pack("<fff", cr, cg, cb))
            fh.write(struct.pack("<ff",  ru, rv))
            for pi in range(start, end):
                lx = positions[pi * 3]
                ly = positions[pi * 3 + 1]
                lz = positions[pi * 3 + 2]
                wp = mw @ Vector((lx, ly, lz))
                r  = radii[pi] * radius_scale
                fh.write(struct.pack("<ffff", wp.x, wp.z, -wp.y, r))
            total_strands += 1
            obj_strand_count += 1
        strand_counts[name] = obj_strand_count
"""

    was_suppressed = s["suppress_dirty"]
    s["suppress_dirty"] = True
    try:
        # ------------------------------------------------------------------
        # 1a. Evaluated helper — attempts full hair density via
        #     evaluated_get(depsgraph).  May crash (SIGSEGV) on Blender 5.1.
        # ------------------------------------------------------------------
        helper_eval = f"""\
import bpy, struct, sys
from mathutils import Vector

hair_names = {hair_names!r}
bin_path   = {bin_path!r}
frame      = {frame!r}

bpy.context.scene.frame_set(frame)
depsgraph = bpy.context.evaluated_depsgraph_get()

total_strands = 0
strand_counts = {{}}
with open(bin_path, "wb") as fh:
    fh.write(b"AHAIR003")
    count_offset = fh.tell()
    fh.write(b"\\x00\\x00\\x00\\x00")

    for name in hair_names:
        obj_orig = bpy.data.objects.get(name)
        if obj_orig is None or obj_orig.type != "CURVES":
            continue
        obj              = obj_orig.evaluated_get(depsgraph)
        curves_data      = obj.data
        mw               = obj.matrix_world
        num_curves       = len(curves_data.curves)
        num_points_total = len(curves_data.points)
        if num_curves == 0 or num_points_total == 0:
            continue
{_STRAND_BODY}
    fh.seek(count_offset)
    fh.write(struct.pack("<I", total_strands))

import json as _json
counts_path = bin_path.replace(".hairbin", "_counts.json")
with open(counts_path, "w") as _cfh:
    _json.dump(strand_counts, _cfh)

print(f"[hair_eval] {{total_strands}} strand(s) written to {{bin_path}}")
sys.exit(0 if total_strands > 0 else 2)
"""

        # ------------------------------------------------------------------
        # 1b. Guide helper — reads obj.data directly; no depsgraph eval.
        #     Guide strands only (~3 k), but crash-safe on Blender 5.1.
        # ------------------------------------------------------------------
        helper_guide = f"""\
import bpy, struct, sys
from mathutils import Vector

hair_names = {hair_names!r}
bin_path   = {bin_path!r}
frame      = {frame!r}

bpy.context.scene.frame_set(frame)
depsgraph = bpy.context.evaluated_depsgraph_get()

total_strands = 0
strand_counts = {{}}
with open(bin_path, "wb") as fh:
    fh.write(b"AHAIR003")
    count_offset = fh.tell()
    fh.write(b"\\x00\\x00\\x00\\x00")

    for name in hair_names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "CURVES":
            continue
        curves_data = obj.data
        # Get matrix_world from the evaluated object so animated parent
        # transforms are included. Only .matrix_world is accessed on the
        # evaluated object — never .data — so this is crash-safe.
        try:
            mw = obj.evaluated_get(depsgraph).matrix_world
        except Exception:
            mw = obj.matrix_world
        num_curves       = len(curves_data.curves)
        num_points_total = len(curves_data.points)
        if num_curves == 0 or num_points_total == 0:
            continue
{_STRAND_BODY}
    fh.seek(count_offset)
    fh.write(struct.pack("<I", total_strands))

import json as _json
counts_path = bin_path.replace(".hairbin", "_counts.json")
with open(counts_path, "w") as _cfh:
    _json.dump(strand_counts, _cfh)

print(f"[hair_guide] {{total_strands}} strand(s) written to {{bin_path}}")
sys.exit(0 if total_strands > 0 else 2)
"""

        with open(eval_py,  "w") as fh:
            fh.write(helper_eval)
        with open(guide_py, "w") as fh:
            fh.write(helper_guide)

        # ------------------------------------------------------------------
        # 1c. Collect material parameters from each hair object's active
        #     material.  Written to a persistent .matassign.json after the
        #     subprocess runs (strand counts needed for --objects).
        # ------------------------------------------------------------------
        import json
        import math
        import datetime

        def _resolve_color_socket(sock):
            """Walk a node tree to find the linear RGB value on a color socket."""
            if sock is None:
                return None
            if not sock.links:
                try:
                    v = sock.default_value
                    return (float(v[0]), float(v[1]), float(v[2]))
                except Exception:
                    return None
            from_node = sock.links[0].from_node
            if from_node.type == 'REROUTE':
                return _resolve_color_socket(from_node.inputs[0])
            if from_node.type == 'HUE_SAT':
                return _resolve_color_socket(from_node.inputs.get('Color'))
            if from_node.type == 'RGB':
                try:
                    v = from_node.outputs[0].default_value
                    return (float(v[0]), float(v[1]), float(v[2]))
                except Exception:
                    return None
            return None

        def _get_marschner_params(mat):
            """Extract Marschner hair params from the active material's node tree.
            Returns a dict with sigma_a, beta_m, beta_n, alpha."""
            # Defaults (medium brown human hair)
            params = {"sigma_a": [0.06, 0.10, 0.20],
                      "beta_m": 0.30, "beta_n": 0.45, "alpha": -2.0}
            if not mat or not mat.use_nodes or not mat.node_tree:
                return params
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_HAIR_PRINCIPLED':
                    # Base color → sigma_a via Beer-Lambert inversion
                    color = _resolve_color_socket(node.inputs.get('Color'))
                    if color:
                        params["sigma_a"] = [
                            -math.log(max(c, 0.001)) for c in color]
                    # Roughness → beta_m (longitudinal) and beta_n (azimuthal)
                    rough_sock = node.inputs.get('Roughness')
                    if rough_sock and not rough_sock.links:
                        r = float(rough_sock.default_value)
                        params["beta_m"] = max(0.05, min(r, 1.0))
                        params["beta_n"] = max(0.05, min(r * 1.5, 1.0))
                    # Tilt (radians in Blender) → alpha (degrees in anacapa)
                    tilt_sock = node.inputs.get('Tilt')
                    if tilt_sock and not tilt_sock.links:
                        params["alpha"] = math.degrees(float(tilt_sock.default_value))
                    return params
                elif node.type == 'BSDF_PRINCIPLED':
                    sock = node.inputs.get('Base Color')
                    color = _resolve_color_socket(sock)
                    if color:
                        params["sigma_a"] = [
                            -math.log(max(c, 0.001)) for c in color]
                    return params
            return params

        # Per-object params list — indices match hair_objs order
        obj_params = []
        for obj in hair_objs:
            mat = obj.active_material
            p = _get_marschner_params(mat)
            comment = f"from Blender material '{mat.name}'" if mat else "no material assigned"
            obj_params.append({"obj": obj, "params": p, "comment": comment})
            print(f"[Anacapa] '{obj.name}' sigma_a=({p['sigma_a'][0]:.3f},"
                  f"{p['sigma_a'][1]:.3f},{p['sigma_a'][2]:.3f}) "
                  f"beta_m={p['beta_m']:.2f} beta_n={p['beta_n']:.2f}")
        # matassign.json is written after subprocess (needs strand counts)

        # ------------------------------------------------------------------
        # 2. Try evaluated subprocess first (full density).
        #    A SIGSEGV produces returncode != 0; fall through to guide.
        # ------------------------------------------------------------------
        print(f"[Anacapa] Reading {len(hair_objs)} hair object(s) "
              f"via background subprocess (evaluated)…")
        result = subprocess.run(
            [bpy.app.binary_path, "--background", blend_path,
             "--python", eval_py],
            capture_output=True,
            text=True,
            timeout=20,
        )

        for line in result.stdout.splitlines():
            if "[hair_eval]" in line and line.strip():
                print(f"  {line.strip()}")

        if result.returncode != 0 or not os.path.exists(bin_path):
            if result.returncode != 0:
                print(f"[Anacapa] Evaluated hair subprocess crashed (exit {result.returncode})"
                      f" — falling back to guide hairs")
            else:
                print("[Anacapa] Evaluated hair produced no binary — falling back to guide hairs")

            # ------------------------------------------------------------------
            # 2b. Guide fallback — reads obj.data directly (crash-safe).
            # ------------------------------------------------------------------
            print(f"[Anacapa] Reading guide hairs via background subprocess…")
            result = subprocess.run(
                [bpy.app.binary_path, "--background", blend_path,
                 "--python", guide_py],
                capture_output=True,
                text=True,
                timeout=180,
            )

            for line in result.stdout.splitlines():
                if "[hair_guide]" in line and line.strip():
                    print(f"  {line.strip()}")
            if result.returncode != 0:
                print(f"[Anacapa] Guide hair subprocess exited {result.returncode} "
                      f"— rendering without hair")
                return None
            if not os.path.exists(bin_path):
                print("[Anacapa] Guide hair produced no binary — rendering without hair")
                return None

        # ------------------------------------------------------------------
        # 2c. Export close-frame hair for motion blur (shutter_close sample).
        #     Reuses the same helper approach as the open sample.
        # ------------------------------------------------------------------
        if do_motion_blur:
            close_total = frame + shutter_close
            cf_int, cf_sub = _frame_subframe(close_total)

            close_helper_eval = f"""\
import bpy, struct, sys
from mathutils import Vector

hair_names = {hair_names!r}
bin_path   = {close_bin_path!r}
frame      = {cf_int!r}
subframe   = {cf_sub!r}

bpy.context.scene.frame_set(frame, subframe=subframe)
depsgraph = bpy.context.evaluated_depsgraph_get()

total_strands = 0
strand_counts = {{}}
with open(bin_path, "wb") as fh:
    fh.write(b"AHAIR003")
    count_offset = fh.tell()
    fh.write(b"\\x00\\x00\\x00\\x00")

    for name in hair_names:
        obj_orig = bpy.data.objects.get(name)
        if obj_orig is None or obj_orig.type != "CURVES":
            continue
        obj              = obj_orig.evaluated_get(depsgraph)
        curves_data      = obj.data
        mw               = obj.matrix_world
        num_curves       = len(curves_data.curves)
        num_points_total = len(curves_data.points)
        if num_curves == 0 or num_points_total == 0:
            continue
{_STRAND_BODY}
    fh.seek(count_offset)
    fh.write(struct.pack("<I", total_strands))

print(f"[hair_close_eval] {{total_strands}} strand(s) written to {{bin_path}}")
sys.exit(0 if total_strands > 0 else 2)
"""

            close_helper_guide = f"""\
import bpy, struct, sys
from mathutils import Vector

hair_names = {hair_names!r}
bin_path   = {close_bin_path!r}
frame      = {cf_int!r}
subframe   = {cf_sub!r}

bpy.context.scene.frame_set(frame, subframe=subframe)
depsgraph = bpy.context.evaluated_depsgraph_get()

total_strands = 0
strand_counts = {{}}
with open(bin_path, "wb") as fh:
    fh.write(b"AHAIR003")
    count_offset = fh.tell()
    fh.write(b"\\x00\\x00\\x00\\x00")

    for name in hair_names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "CURVES":
            continue
        try:
            mw = obj.evaluated_get(depsgraph).matrix_world
        except Exception:
            mw = obj.matrix_world
        curves_data      = obj.data
        num_curves       = len(curves_data.curves)
        num_points_total = len(curves_data.points)
        if num_curves == 0 or num_points_total == 0:
            continue
{_STRAND_BODY}
    fh.seek(count_offset)
    fh.write(struct.pack("<I", total_strands))

print(f"[hair_close_guide] {{total_strands}} strand(s) written to {{bin_path}}")
sys.exit(0 if total_strands > 0 else 2)
"""

            with open(close_eval_py,  "w") as fh:
                fh.write(close_helper_eval)
            with open(close_guide_py, "w") as fh:
                fh.write(close_helper_guide)

            print(f"[Anacapa] Exporting close-frame hair (frame {close_total:.3f}) for motion blur…")
            close_result = subprocess.run(
                [bpy.app.binary_path, "--background", blend_path,
                 "--python", close_eval_py],
                capture_output=True, text=True, timeout=180,
            )
            for line in close_result.stdout.splitlines():
                if "[hair_close_eval]" in line and line.strip():
                    print(f"  {line.strip()}")

            if close_result.returncode != 0 or not os.path.exists(close_bin_path):
                print("[Anacapa] Close-frame evaluated hair failed — trying guide fallback")
                close_result = subprocess.run(
                    [bpy.app.binary_path, "--background", blend_path,
                     "--python", close_guide_py],
                    capture_output=True, text=True, timeout=180,
                )
                for line in close_result.stdout.splitlines():
                    if "[hair_close_guide]" in line and line.strip():
                        print(f"  {line.strip()}")
                if close_result.returncode != 0 or not os.path.exists(close_bin_path):
                    print("[Anacapa] Close-frame hair export failed — hair will render without motion blur")
                    do_motion_blur = False

        # ------------------------------------------------------------------
        # 2d. Write persistent .matassign.json alongside the ABC.
        #     Strand counts from subprocess are logged for diagnostics.
        # ------------------------------------------------------------------
        strand_counts = {}
        if os.path.exists(counts_path):
            try:
                with open(counts_path) as _cf:
                    strand_counts = json.load(_cf)
            except Exception as e:
                print(f"[Anacapa] Failed to read strand counts: {e}")

        assignments = []
        for entry in obj_params:
            obj      = entry["obj"]
            p        = entry["params"]
            count    = strand_counts.get(obj.name, 0)
            print(f"[Anacapa] '{obj.name}' → {count} strand(s)")
            assignments.append({
                "object":  obj.name,
                "comment": entry["comment"],
                "material": {
                    "type":    "marschner",
                    "sigma_a": [round(v, 5) for v in p["sigma_a"]],
                    "beta_m":  round(p["beta_m"], 4),
                    "beta_n":  round(p["beta_n"], 4),
                    "alpha":   round(p["alpha"],  4),
                },
            })

        matassign_data = {
            "version":       1,
            "exported_at":   datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_app":    f"Blender {bpy.app.version_string}",
            "source_file":   blend_path,
            "source_frame":  frame,
            "geometry_cache": os.path.basename(abc_path),
            "notes":         "",
            "assignments":   assignments,
        }
        # Write the Blender-generated matassign into the project cache directory.
        # Two-layer system:
        #   <stem>.base.matassign.json — Blender-owned; always overwritten on export.
        #   <stem>.matassign.json      — User-owned; created from base on first export
        #                                only, never overwritten.  TDs edit this file
        #                                to override per-object materials without
        #                                touching Blender.  The renderer receives both
        #                                via --matassign (base first, user second) so
        #                                user entries always win.
        blend_stem     = os.path.splitext(os.path.basename(blend_path))[0]
        project_dir    = get_project_dir(blend_path)
        base_path     = os.path.join(project_dir, blend_stem + ".matassign.json")
        override_path = os.path.join(project_dir, blend_stem + ".override.matassign.json")
        with open(base_path, "w") as fh:
            json.dump(matassign_data, fh, indent=2)
        print(f"[Anacapa] Material assignments → {base_path}")
        if not os.path.exists(override_path):
            shutil.copy2(base_path, override_path)
            print(f"[Anacapa] Created override layer → {override_path}")
        matassign_paths = [base_path, override_path]

        # ------------------------------------------------------------------
        # 3. Convert binary → Alembic via hair_export.
        #    This is our own standalone tool, completely safe.
        # ------------------------------------------------------------------
        cmd2 = [hair_export, bin_path, abc_path]
        if os.path.exists(counts_path):
            cmd2 += ["--objects", counts_path]
        if do_motion_blur and os.path.exists(close_bin_path):
            cmd2 += ["--close", close_bin_path]
        result2 = subprocess.run(
            cmd2,
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in result2.stdout.splitlines():
            if line.strip():
                print(f"  [hair_export] {line}")
        if result2.returncode != 0 or not os.path.exists(abc_path):
            err = result2.stderr.strip()
            print(f"[Anacapa] hair_export failed"
                  + (f": {err}" if err else ""))
            # If an old ABC cache is still on disk, the renderer will use it.
            # Still return matassign_paths so current material settings are applied
            # (matassign was already written above regardless of ABC outcome).
            if os.path.exists(abc_path):
                print(f"[Anacapa] Using stale ABC cache — rebuild to refresh geometry")
                return matassign_paths
            print(f"[Anacapa] No usable hair cache — rendering without hair")
            return None

    except subprocess.TimeoutExpired:
        print("[Anacapa] Hair subprocess timed out — rendering without hair")
        return None
    except Exception as e:
        print(f"[Anacapa] Hair export error: {e} — rendering without hair")
        return None
    finally:
        for path in (eval_py, guide_py, bin_path, counts_path,
                     close_eval_py, close_guide_py, close_bin_path):
            try:
                os.remove(path)
            except OSError:
                pass
        if not was_suppressed:
            s["suppress_dirty"] = False

    s["dirty_hair"]           = False
    s["cached_abc_path"]      = abc_path
    s["cached_frame"]         = frame
    s["cached_shutter_close"] = shutter_close
    print(f"[Anacapa] Hair exported → {abc_path}")
    return matassign_paths


def _load_scene_export_module():
    if _load_scene_export_module._mod is not None:
        return _load_scene_export_module._mod
    path = os.path.join(os.path.dirname(__file__), "anacapa_scene_export.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("anacapa_scene_export", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _load_scene_export_module._mod = mod
    return mod

_load_scene_export_module._mod = None


def _anacapa_export_scene(usd_path, context, executable,
                          shutter_open=0.0, shutter_close=0.0,
                          depsgraph_override=None):
    """
    New scene export pipeline — reads from the depsgraph without touching the
    live scene, writes a binary blob, invokes the C++ exporter to produce USD,
    then writes .mtlx/.osl material files alongside it.
    """
    import subprocess

    scene_mod = _load_scene_export_module()
    if scene_mod is None:
        raise RuntimeError("anacapa_scene_export.py not found")

    blob_path = usd_path + ".blob.bin"
    prep = _load_prep_module()

    # 1. Collect HALO / GN-based particles before any frame seeking so that GN
    #    simulation zones are captured at the current evaluated state.
    # Skip when depsgraph_override is set (render engine path): bpy.ops calls
    # inside RenderEngine.render() deadlock waiting for the event loop.
    # In the render engine path GN instances come from the render-mode depsgraph
    # directly, so halo collection is not needed for mesh-based GN scatter.
    if prep is not None and depsgraph_override is None:
        try:
            n = prep.collect_halo_particles()
            if n:
                print(f"[Anacapa] Collected {n} halo particles")
        except Exception as e:
            print(f"[Anacapa] Halo particle collection warning: {e}")

    # 2. Write binary blob from depsgraph (motion blur keys sampled when enabled).
    # In the render-engine path, pass pre-collected transforms from render_pre so
    # export_scene_binary doesn't need to call scene.frame_set() internally.
    rdata = _render_pre_data()
    motion_xf = rdata.get('motion_xf') if depsgraph_override is not None else None
    scene_mod.export_scene_binary(blob_path, context,
                                  shutter_open=shutter_open,
                                  shutter_close=shutter_close,
                                  depsgraph_override=depsgraph_override,
                                  motion_xf=motion_xf)

    # 3. C++ converts blob → USD
    result = subprocess.run(
        [executable, "export-scene", blob_path, usd_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"anacapa export-scene failed (code {result.returncode})")
    if result.stdout:
        print(result.stdout, end="")

    # 4. Inject halo particles as UsdGeomPoints into the written USD.
    # Skip in render engine path (depsgraph_override set): collect_halo_particles
    # was not called (bpy.ops calls in that function crash/corrupt state inside
    # RenderEngine.render), so the stash is empty or stale.
    if prep is not None and depsgraph_override is None:
        try:
            prep._inject_halo_particles(usd_path, shutter_close=shutter_close)
        except Exception as e:
            print(f"[Anacapa] Halo particle injection warning: {e}")

    # 5. Write MaterialX / OSL material files
    if prep is not None:
        mtlx_dir = os.path.join(os.path.dirname(usd_path), "materials")
        try:
            prep._export_materialx_graphs(mtlx_dir)
        except Exception as e:
            print(f"[Anacapa] MaterialX export warning: {e}")

    # Blob is a temp artefact
    try:
        os.remove(blob_path)
    except OSError:
        pass


def export_usd(usd_path, context, run_prep=True,
               shutter_open=0.0, shutter_close=0.0,
               depsgraph_override=None):
    """
    Export the scene to a single USD file.
    Skips export entirely when nothing changed.
    """
    s = _state()

    # Invalidate when shutter_close changes so particle close positions are rewritten.
    s.setdefault("cached_particle_shutter_close", None)
    if s["cached_particle_shutter_close"] != shutter_close:
        s["dirty_scene"] = True
        s["cached_particle_shutter_close"] = shutter_close

    # Nothing changed — reuse cached USD
    cached = s["cached_usd_path"]
    if not s["dirty_scene"] and not s["dirty_transform"] \
            and cached and os.path.exists(cached):
        if usd_path != cached:
            shutil.copy2(cached, usd_path)
        print("[Anacapa] Scene unchanged — reusing cached USD")
        return

    executable = get_executable(context)

    was_suppressed = s["suppress_dirty"]
    s["suppress_dirty"] = True
    try:
        _anacapa_export_scene(usd_path, context, executable,
                              shutter_open=shutter_open,
                              shutter_close=shutter_close,
                              depsgraph_override=depsgraph_override)

        # Sky injection and DOF settings written into the USD post-export
        prep = _load_prep_module() if run_prep else None
        if prep is not None:
            try:
                ana      = getattr(bpy.context.scene, 'anacapa', None)
                use_dof  = getattr(ana, 'use_dof', False)
                sky      = ana if (ana and getattr(ana, 'use_sky', False)) else None
                prep.post_process_usd(usd_path, shutter_close=shutter_close,
                                      disable_dof=not use_dof,
                                      sky_settings=sky)
            except Exception as e:
                print(f"[Anacapa] USD post-process warning: {e}")

        s["dirty_scene"]     = False
        s["dirty_transform"] = False
        s["cached_usd_path"] = usd_path

    finally:
        if not was_suppressed:
            s["suppress_dirty"] = False


def build_command(executable, usd_path, settings, width, height, output_path,
                  curves_path=None, matassign_paths=None, frame=None):
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

    if settings.integrator == "photon":
        # Trace photons when caustics or SSS map is enabled.
        effective_photons = settings.num_photons if (settings.pm_caustics or settings.pm_subsurface) else 0
        cmd += [
            "--num-photons",   str(effective_photons),
            "--photon-radius", str(settings.photon_radius),
        ]

    if frame is not None:
        cmd += ["--frame", str(int(frame))]

    if settings.gpu_assist:
        cmd.append("--gpu-assist")

    if settings.num_threads > 0:
        cmd += ["-t", str(settings.num_threads)]

    if not settings.adaptive:
        cmd.append("--no-adaptive")
    elif settings.adaptive_base_spp > 0:
        cmd += ["--adaptive-base-spp", str(settings.adaptive_base_spp)]

    # Sky overrides the HDRI env map — attributes are already in the USD.
    use_sky = getattr(settings, 'use_sky', False)
    if not use_sky:
        env = bpy.path.abspath(settings.env_path) if settings.env_path else ""
        if env:
            cmd += ["--env", env, "--env-intensity", str(settings.env_intensity)]

    if settings.light_angle > 0:
        cmd += ["--light-angle", str(settings.light_angle)]

    if settings.override_lights:
        cmd.append("--override-lights")
    if settings.override_materials:
        cmd.append("--override-materials")
    if not getattr(settings, 'use_osl', True):
        cmd.append("--skip-osl")

    if not getattr(settings, 'use_dof', False):
        cmd.append("--no-dof")
    else:
        fstop      = settings.fstop
        focus_dist = getattr(settings, 'focus_distance', 0.0)
        if fstop == 0:
            try:
                import bpy as _bpy
                cam_obj = _bpy.context.scene.camera
                if cam_obj and cam_obj.data and cam_obj.data.dof:
                    dof       = cam_obj.data.dof
                    fstop     = dof.aperture_fstop
                    focus_obj = dof.focus_object
                    if focus_obj:
                        cam_loc    = cam_obj.matrix_world.translation
                        obj_loc    = focus_obj.matrix_world.translation
                        focus_dist = (cam_loc - obj_loc).length
                    else:
                        focus_dist = dof.focus_distance
            except Exception as _e:
                print(f"[Anacapa] DOF: failed to read Blender camera: {_e}")
        if fstop > 0 and focus_dist > 0:
            # Blender's USD exporter writes metersPerUnit=0.01 (cm convention) but
            # exports coordinates in Blender units, so the renderer's focalLength is
            # 100× too large relative to scene scale. Scale fstop by the same factor
            # so the computed aperture radius matches the actual coordinate scale.
            try:
                import bpy as _bpy
                scale_length = _bpy.context.scene.unit_settings.scale_length
            except Exception:
                scale_length = 1.0
            fstop = fstop * (scale_length / 0.01)
            cmd += ["--fstop", str(fstop), "--focus-distance", str(focus_dist)]

    if getattr(settings, 'use_motion_blur', False):
        shutter = getattr(settings, 'motion_blur_shutter', 0.5)
        position = getattr(settings, 'motion_blur_position', 'CENTER')
        if shutter > 0:
            if position == 'START':
                s_open, s_close = 0.0, shutter
            elif position == 'END':
                s_open, s_close = -shutter, 0.0
            else:  # CENTER
                s_open, s_close = -shutter / 2.0, shutter / 2.0
            cmd += ["--shutter-open",  str(s_open),
                    "--shutter-close", str(s_close)]

    if settings.denoise:
        cmd.append("--denoise")
    if settings.write_aovs:
        cmd.append("--write-aovs")

    # Film / pixel reconstruction filter
    if getattr(settings, 'pixel_filter', None):
        cmd += ["--filter", settings.pixel_filter]
    if getattr(settings, 'filter_width', 0.0) > 0.0:
        cmd += ["--filter-width", str(settings.filter_width)]

    # Hair tessellation
    hair_tess = getattr(settings, 'hair_tess_steps', 4)
    if hair_tess != 4:
        cmd += ["--hair-tess-steps", str(hair_tess)]

    if settings.camera_path:
        cmd += ["--camera", settings.camera_path]

    if curves_path:
        cmd += ["--curves", curves_path]


    if matassign_paths:
        paths = [matassign_paths] if isinstance(matassign_paths, str) else matassign_paths
        for p in paths:
            cmd += ["--matassign", p]

    return cmd
