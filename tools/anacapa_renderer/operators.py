import bpy
import os
import shlex
import subprocess
import tempfile
import shutil
import threading
import queue

from .export import (get_executable, get_viewer_executable, export_usd,
                      build_command, get_scenes_dir, get_cache_dir,
                      substitute_frame_tokens)

VIEWER_SOCK = "/tmp/anacapa_viewer.sock"


def _is_viewer_running():
    """Try connecting to the viewer socket; return True if it answers."""
    import socket as _sock
    try:
        s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        s.settimeout(0.3)
        s.connect(VIEWER_SOCK)
        s.close()
        return True
    except Exception:
        return False


def _launch_viewer(viewer_path):
    """Start the viewer process and wait up to 3 s for its socket to appear."""
    import time
    proc = subprocess.Popen([viewer_path, "--listen"])
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if _is_viewer_running():
            return proc
        time.sleep(0.1)
    return proc  # return even if we timed out — render will still try --display


_viewer_proc = None  # module-level handle so it outlives operator instances


class ANACAPA_OT_launch_viewer(bpy.types.Operator):
    """Launch the Anacapa viewer and start listening for a renderer connection"""
    bl_idname = "anacapa.launch_viewer"
    bl_label  = "Launch Viewer"

    def execute(self, context):
        global _viewer_proc
        viewer_path = get_viewer_executable(context)
        import os
        if not os.path.exists(viewer_path):
            self.report({'ERROR'}, f"Viewer not found: {viewer_path}")
            return {'CANCELLED'}

        # If viewer is already running, just report and return
        if _is_viewer_running():
            self.report({'INFO'}, "Viewer is already running")
            return {'FINISHED'}

        # Kill stale process if any
        if _viewer_proc and _viewer_proc.poll() is None:
            _viewer_proc.terminate()

        _viewer_proc = _launch_viewer(viewer_path)
        if _is_viewer_running():
            self.report({'INFO'}, "Viewer launched")
        else:
            self.report({'WARNING'}, "Viewer launched but socket not ready yet")
        return {'FINISHED'}


class ANACAPA_OT_render(bpy.types.Operator):
    """Export scene to USD then launch an Anacapa render"""
    bl_idname = "anacapa.render"
    bl_label  = "Render"

    _proc              = None
    _timer             = None
    _tmp_dir           = None
    _output_path       = None
    _preview_path      = None
    _preview_img       = None   # bpy.data.images entry for the live preview
    _log_queue         = None
    _reader            = None
    _hidden_for_render = []     # list of (name, original_hide_viewport) for CURVES objects

    def execute(self, context):
        # Restore any CURVES objects left hidden by a previous render that
        # crashed or was cancelled before _finish() could run.
        for entry in ANACAPA_OT_render._hidden_for_render:
            name, was_hidden = entry if isinstance(entry, tuple) else (entry, False)
            obj = bpy.data.objects.get(name)
            if obj:
                obj.hide_viewport = was_hidden
                obj.pop("anacapa_hidden_for_render", None)
        ANACAPA_OT_render._hidden_for_render = []

        scene    = context.scene
        settings = scene.anacapa
        scale    = scene.render.resolution_percentage / 100.0
        width    = int(scene.render.resolution_x * scale)
        height   = int(scene.render.resolution_y * scale)

        # Scene files, caches, and material assignments are project products and
        # must live in a stable project directory.  Require the blend to be saved.
        blend_path = bpy.data.filepath
        if not blend_path:
            self.report({'ERROR'},
                        "Save the blend file before rendering — "
                        "scene files and caches are written alongside it.")
            return {'CANCELLED'}

        blend_stem  = os.path.splitext(os.path.basename(blend_path))[0]
        scenes_dir  = get_scenes_dir(blend_path)   # also creates materials/ textures/
        cache_dir   = get_cache_dir(blend_path)

        tmp_dir  = os.path.join(bpy.app.tempdir, "anacapa_render")
        os.makedirs(tmp_dir, exist_ok=True)

        usd_path = os.path.join(scenes_dir, blend_stem + "_scene.usdc")

        from . import export as export_mod

        # --- Export hair to Alembic FIRST (before prep_scene converts/removes
        #     Curves objects for the USD export) ---
        abc_path        = None
        matassign_paths = None
        hair_objs = export_mod.get_hair_objects(context)
        if hair_objs:
            frame    = context.scene.frame_current
            abc_path = os.path.join(cache_dir, f"hair.{frame:04d}.abc")
            self.report({'INFO'}, f"Exporting {len(hair_objs)} hair object(s) to Alembic…")
            # Compute shutter interval for hair motion blur (same as renderer cmd).
            _s_open, _s_close = 0.0, 0.0
            if getattr(settings, 'use_motion_blur', False):
                _shutter  = getattr(settings, 'motion_blur_shutter', 0.5)
                _position = getattr(settings, 'motion_blur_position', 'CENTER')
                if _shutter > 0:
                    if _position == 'START':
                        _s_open, _s_close = 0.0, _shutter
                    elif _position == 'END':
                        _s_open, _s_close = -_shutter, 0.0
                    else:
                        _s_open, _s_close = -_shutter / 2.0, _shutter / 2.0
            try:
                matassign_paths = export_mod.export_hair_abc(
                    abc_path, context,
                    shutter_open=_s_open, shutter_close=_s_close)
            except Exception as e:
                self.report({'WARNING'}, f"Hair export failed: {e}")
                abc_path = None
            if abc_path and not os.path.exists(abc_path):
                self.report({'WARNING'}, "Hair export produced no file — rendering without hair")
                abc_path        = None
                matassign_paths = None

        # --- Hide CURVES objects from the viewport before prep/USD export ---
        # Blender 5.1 crashes (SIGSEGV in hair_nodes evaluation) when the
        # viewport depsgraph evaluates visible CURVES objects while anacapa
        # renders in the background.  Hiding them prevents that evaluation.
        # They are restored in _finish() after the render completes.
        # Hide ALL CURVES objects before USD export/render, recording their
        # original hide_viewport state so _finish() can restore exactly.
        hidden_for_render = []
        for obj in context.scene.objects:
            if obj.type == 'CURVES':
                was_hidden = obj.hide_viewport
                obj.hide_viewport = True
                # Custom property survives Blender restart — load_post handler
                # uses it to unhide objects if _finish() never ran.
                obj["anacapa_hidden_for_render"] = True
                hidden_for_render.append((obj.name, was_hidden))
        ANACAPA_OT_render._hidden_for_render = hidden_for_render
        if hidden_for_render:
            self.report({'INFO'}, f"Hidden {len(hidden_for_render)} hair object(s) for render")

        # --- Export USD (runs prep_scene, which may convert/remove Curves) ---
        _shutter_close = 0.0
        if getattr(settings, 'use_motion_blur', False):
            _shutter  = getattr(settings, 'motion_blur_shutter', 0.5)
            _position = getattr(settings, 'motion_blur_position', 'CENTER')
            if _shutter > 0:
                _shutter_close = _shutter if _position in ('START', 'END') \
                                 else _shutter / 2.0

        self.report({'INFO'}, "Exporting USD…")
        s = export_mod._state()
        # Always mark transforms dirty before export so camera/object moves are
        # captured even if the depsgraph event fired after suppress_dirty was set
        # on a previous frame or during a previous render's grace period.
        s["dirty_transform"] = True
        s["suppress_dirty"]  = True
        try:
            export_usd(usd_path, context, shutter_close=_shutter_close)
        except Exception as e:
            self.report({'ERROR'}, f"USD export failed: {e}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {'CANCELLED'}

        if not os.path.exists(usd_path):
            self.report({'ERROR'}, "USD export produced no file")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {'CANCELLED'}

        # --- Build command ---
        output_path  = os.path.join(tmp_dir, "render.exr")
        preview_path = os.path.join(tmp_dir, "preview.exr")
        executable   = get_executable(context)
        cmd          = build_command(executable, usd_path, settings,
                                     width, height, output_path,
                                     curves_path=abc_path,
                                     matassign_paths=matassign_paths,
                                     frame=context.scene.frame_current)

        # --- Viewer / display driver ---
        use_viewer = settings.use_viewer
        if use_viewer:
            global _viewer_proc
            viewer_path = get_viewer_executable(context)
            if not os.path.exists(viewer_path):
                self.report({'WARNING'},
                            f"Viewer binary not found at {viewer_path} — rendering without viewer")
                use_viewer = False
            elif not _is_viewer_running():
                if _viewer_proc and _viewer_proc.poll() is None:
                    _viewer_proc.terminate()
                _viewer_proc = _launch_viewer(viewer_path)

            if use_viewer:
                if _is_viewer_running():
                    cmd += ["--display"]
                else:
                    self.report({'WARNING'},
                                "Viewer not reachable — rendering without live display")
                    use_viewer = False

        # Progressive EXR preview — linear float with alpha, tone-mapped by the viewer.
        # Written when not using the socket viewer so Blender's Image Editor can show
        # progressive updates with correct colour and alpha channel.
        if not use_viewer:
            cmd += ["--preview-exr", preview_path]

        # --- Launch Anacapa ---
        import shlex
        print(f"[Anacapa] Command: {shlex.join(cmd)}")
        self.report({'INFO'}, "Launching Anacapa…")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        except FileNotFoundError:
            self.report({'ERROR'}, f"Anacapa executable not found: {executable}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {'CANCELLED'}

        # Background thread drains stdout so the pipe never blocks
        log_queue = queue.Queue()
        def _reader_fn():
            for line in proc.stdout:
                log_queue.put(line.rstrip())
            log_queue.put(None)  # sentinel

        reader = threading.Thread(target=_reader_fn, daemon=True)
        reader.start()

        ANACAPA_OT_render._proc         = proc
        ANACAPA_OT_render._tmp_dir      = tmp_dir
        ANACAPA_OT_render._output_path  = output_path
        ANACAPA_OT_render._preview_path = None if use_viewer else preview_path
        ANACAPA_OT_render._preview_img  = None
        ANACAPA_OT_render._log_queue    = log_queue
        ANACAPA_OT_render._reader       = reader

        # Suppress dirty tracking while anacapa is running so Blender's
        # post-render internal updates don't re-dirty the scene flags.
        export_mod._state()["suppress_dirty"] = True

        wm = context.window_manager
        ANACAPA_OT_render._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            if event.type == 'ESC':
                proc = ANACAPA_OT_render._proc
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                self._finish(context)
                self.report({'WARNING'}, "Render cancelled")
                return {'CANCELLED'}
            return {'PASS_THROUGH'}

        # Drain log queue — echo to terminal and update status bar
        q = ANACAPA_OT_render._log_queue
        last_line = None
        while True:
            try:
                line = q.get_nowait()
                if line is None:
                    break
                last_line = line
                print(f"  [anacapa] {line}")
            except queue.Empty:
                break
        if last_line:
            context.workspace.status_text_set(f"Anacapa: {last_line}")

        # Update progressive preview if PNG has been written (not in viewer mode)
        if ANACAPA_OT_render._preview_path:
            self._refresh_preview(context)

        # Check if process finished
        ret = ANACAPA_OT_render._proc.poll()
        if ret is None:
            return {'PASS_THROUGH'}

        ANACAPA_OT_render._reader.join(timeout=1.0)
        self._finish(context)

        if ret != 0:
            self.report({'WARNING'}, f"Anacapa exited with code {ret}")
            return {'CANCELLED'}

        return self._load_result(context)

    def _refresh_preview(self, context):
        preview_path = ANACAPA_OT_render._preview_path
        if not preview_path or not os.path.exists(preview_path):
            return

        img = ANACAPA_OT_render._preview_img

        if img is None:
            # First appearance — load and show in Image Editor
            existing = bpy.data.images.get("Anacapa Preview")
            if existing:
                bpy.data.images.remove(existing)
            img = bpy.data.images.load(preview_path, check_existing=False)
            img.name = "Anacapa Preview"
            img.alpha_mode = 'STRAIGHT'
            ANACAPA_OT_render._preview_img = img

            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = img
                    break
        else:
            # Reload in place so the Image Editor updates automatically
            img.reload()

    def _finish(self, context):
        wm = context.window_manager
        if ANACAPA_OT_render._timer:
            wm.event_timer_remove(ANACAPA_OT_render._timer)
            ANACAPA_OT_render._timer = None
        context.workspace.status_text_set(None)

        # Restore CURVES objects to their original viewport visibility
        for name, was_hidden in ANACAPA_OT_render._hidden_for_render:
            obj = bpy.data.objects.get(name)
            if obj:
                obj.hide_viewport = was_hidden
                obj.pop("anacapa_hidden_for_render", None)
        ANACAPA_OT_render._hidden_for_render = []

        # Clear the preview image and its backing file so a subsequent render
        # always starts with a blank slate.  Without this, a cancelled render
        # leaves a stale preview.png on disk; the next render (especially GPU
        # mode, which writes no progressive PNG) reloads the old partial image
        # making it look like the cancelled render was paused and resumed.
        preview_img = ANACAPA_OT_render._preview_img
        if preview_img:
            try:
                bpy.data.images.remove(preview_img)
            except Exception:
                pass
            ANACAPA_OT_render._preview_img = None
        preview_path = ANACAPA_OT_render._preview_path
        if preview_path and os.path.exists(preview_path):
            try:
                os.remove(preview_path)
            except Exception:
                pass
        ANACAPA_OT_render._preview_path = None

        # Re-enable dirty tracking after a short delay so Blender's own
        # post-render depsgraph updates flush through without dirtying our flags.
        def _reenable():
            from . import export as export_mod
            export_mod._state()["suppress_dirty"] = False
        bpy.app.timers.register(_reenable, first_interval=2.0)

    def _load_result(self, context):
        output_path  = ANACAPA_OT_render._output_path
        tmp_dir      = ANACAPA_OT_render._tmp_dir
        settings     = context.scene.anacapa

        if not output_path or not os.path.exists(output_path):
            self.report({'ERROR'}, f"No EXR at {output_path}")
            return {'CANCELLED'}

        # Keep the EXR accessible after tmp_dir is cleaned up
        persist_path = os.path.join(bpy.app.tempdir, "anacapa_last_render.exr")
        shutil.copy2(output_path, persist_path)

        # If the user set an EXR output path, copy the rendered EXR there
        # with $F frame tokens substituted.  Create parent dirs as needed.
        if settings.output_path:
            user_path = substitute_frame_tokens(
                bpy.path.abspath(settings.output_path),
                context.scene.frame_current)
            os.makedirs(os.path.dirname(user_path) or ".", exist_ok=True)
            shutil.copy2(output_path, user_path)
            print(f"[Anacapa] Saved EXR -> {user_path}")

        # Remove preview image; replace with final EXR
        preview_img = ANACAPA_OT_render._preview_img
        if preview_img:
            bpy.data.images.remove(preview_img)
            ANACAPA_OT_render._preview_img = None

        existing = bpy.data.images.get("Anacapa Render")
        if existing:
            bpy.data.images.remove(existing)

        try:
            img = bpy.data.images.load(persist_path, check_existing=False)
            img.name = "Anacapa Render"
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load EXR: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Render done: {img.size[0]}x{img.size[1]}")

        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.spaces.active.image = img
                break

        return {'FINISHED'}


class ANACAPA_OT_export_scene(bpy.types.Operator):
    """Export the scene as USD and print the anacapa render command to the console"""
    bl_idname = "anacapa.export_scene"
    bl_label  = "Export Scene for Anacapa"

    filepath:    bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.usdc;*.usda", options={'HIDDEN'})

    def invoke(self, context, event):
        if bpy.data.filepath:
            self.filepath = os.path.splitext(bpy.data.filepath)[0] + ".usdc"
        else:
            self.filepath = os.path.join(os.getcwd(), "scene.usdc")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene    = context.scene
        settings = scene.anacapa
        scale    = scene.render.resolution_percentage / 100.0
        width    = int(scene.render.resolution_x * scale)
        height   = int(scene.render.resolution_y * scale)

        usd_path    = bpy.path.abspath(self.filepath)
        output_path = os.path.splitext(usd_path)[0] + ".exr"
        executable  = get_executable(context)

        # Export hair FIRST (before prep_scene converts/removes Curves objects)
        from . import export as export_mod
        abc_path        = None
        matassign_paths = None
        hair_objs = export_mod.get_hair_objects(context)
        if hair_objs:
            frame      = context.scene.frame_current
            blend_path = bpy.data.filepath
            if blend_path:
                abc_path = os.path.join(get_cache_dir(blend_path), f"hair.{frame:04d}.abc")
            else:
                abc_path = os.path.splitext(usd_path)[0] + f"_hair.{frame:04d}.abc"
            self.report({'INFO'}, f"Exporting {len(hair_objs)} hair object(s)…")
            _s_open, _s_close = 0.0, 0.0
            if getattr(settings, 'use_motion_blur', False):
                _shutter  = getattr(settings, 'motion_blur_shutter', 0.5)
                _position = getattr(settings, 'motion_blur_position', 'CENTER')
                if _shutter > 0:
                    if _position == 'START':
                        _s_open, _s_close = 0.0, _shutter
                    elif _position == 'END':
                        _s_open, _s_close = -_shutter, 0.0
                    else:
                        _s_open, _s_close = -_shutter / 2.0, _shutter / 2.0
            try:
                matassign_paths = export_mod.export_hair_abc(
                    abc_path, context,
                    shutter_open=_s_open, shutter_close=_s_close)
            except Exception as e:
                self.report({'WARNING'}, f"Hair export failed: {e}")
                abc_path = None
            if abc_path and not os.path.exists(abc_path):
                self.report({'WARNING'}, "Hair export produced no file — skipping curves")
                abc_path        = None
                matassign_paths = None

        self.report({'INFO'}, f"Exporting USD to {usd_path}…")
        try:
            export_usd(usd_path, context)
        except Exception as e:
            self.report({'ERROR'}, f"USD export failed: {e}")
            return {'CANCELLED'}

        if not os.path.exists(usd_path):
            self.report({'ERROR'}, "USD export produced no file")
            return {'CANCELLED'}

        cmd = build_command(executable, usd_path, settings,
                            width, height, output_path,
                            curves_path=abc_path,
                            matassign_paths=matassign_paths,
                            frame=context.scene.frame_current)
        cmd_str = shlex.join(cmd)

        print("\n" + "=" * 72)
        print(f"[Anacapa] Scene exported to:\n  {usd_path}")
        print(f"\n[Anacapa] Render command:\n  {cmd_str}")
        print("=" * 72 + "\n")

        self.report({'INFO'}, "Exported. Render command printed to System Console.")
        return {'FINISHED'}


class ANACAPA_OT_bake_particles(bpy.types.Operator):
    """Bake all GN simulation particle caches in the scene to disk."""
    bl_idname = "anacapa.bake_particles"
    bl_label  = "Bake Particles"

    def execute(self, context):
        baked = 0
        for obj in context.scene.objects:
            if obj.hide_render:
                continue
            if not any(m.type == 'NODES' for m in obj.modifiers):
                continue
            context.view_layer.objects.active = obj
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            result = bpy.ops.object.simulation_nodes_cache_bake(selected=False)
            if 'FINISHED' in result:
                baked += 1
                self.report({'INFO'}, f"Baked '{obj.name}'")
            else:
                self.report({'WARNING'}, f"'{obj.name}' bake returned {result} — check modifier settings")
        if baked == 0:
            self.report({'WARNING'}, "No GN simulation objects found to bake.")
        return {'FINISHED'}


_classes = [ANACAPA_OT_launch_viewer, ANACAPA_OT_render, ANACAPA_OT_export_scene,
            ANACAPA_OT_bake_particles]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
