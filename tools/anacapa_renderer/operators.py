import bpy
import os
import shlex
import subprocess
import tempfile
import shutil
import threading
import queue

from .export import get_executable, export_usd, build_command


class ANACAPA_OT_render(bpy.types.Operator):
    """Export scene to USD then launch an Anacapa render"""
    bl_idname = "anacapa.render"
    bl_label  = "Render"

    _proc         = None
    _timer        = None
    _tmp_dir      = None
    _output_path  = None
    _preview_path = None
    _preview_img  = None   # bpy.data.images entry for the live preview
    _log_queue    = None
    _reader       = None

    def execute(self, context):
        scene    = context.scene
        settings = scene.anacapa
        scale    = scene.render.resolution_percentage / 100.0
        width    = int(scene.render.resolution_x * scale)
        height   = int(scene.render.resolution_y * scale)

        tmp_dir  = tempfile.mkdtemp(prefix="anacapa_")
        usd_path = os.path.join(bpy.app.tempdir, "anacapa_scene_cache.usdc")

        # --- Export USD ---
        self.report({'INFO'}, "Exporting USD…")
        try:
            export_usd(usd_path, context)
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
        preview_path = os.path.join(tmp_dir, "preview.png")
        executable   = get_executable(context)
        cmd, _       = build_command(executable, usd_path, settings,
                                     width, height, output_path)

        # Always add progressive PNG preview (overrides settings.png_path for temp use)
        if "--png" not in cmd:
            cmd += ["--png", preview_path]
        else:
            # Replace the user-specified png with our temp preview;
            # the final PNG will be written by _load_result if settings.png_path is set.
            idx = cmd.index("--png")
            cmd[idx + 1] = preview_path

        # --- Launch Anacapa ---
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
        ANACAPA_OT_render._preview_path = preview_path
        ANACAPA_OT_render._preview_img  = None
        ANACAPA_OT_render._log_queue    = log_queue
        ANACAPA_OT_render._reader       = reader

        wm = context.window_manager
        ANACAPA_OT_render._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            if event.type == 'ESC':
                ANACAPA_OT_render._proc.terminate()
                ANACAPA_OT_render._proc.wait()
                self._finish(context)
                self.report({'WARNING'}, "Render cancelled")
                return {'CANCELLED'}
            return {'PASS_THROUGH'}

        # Drain log queue, update status bar with latest line
        q = ANACAPA_OT_render._log_queue
        last_line = None
        while True:
            try:
                line = q.get_nowait()
                if line is None:
                    break
                last_line = line
            except queue.Empty:
                break
        if last_line:
            context.workspace.status_text_set(f"Anacapa: {last_line}")

        # Update progressive preview if PNG has been written
        self._refresh_preview(context)

        # Check if process finished
        ret = ANACAPA_OT_render._proc.poll()
        if ret is None:
            return {'PASS_THROUGH'}

        ANACAPA_OT_render._reader.join(timeout=1.0)
        self._finish(context)

        if ret != 0:
            self.report({'WARNING'}, f"Anacapa exited with code {ret}")
            shutil.rmtree(ANACAPA_OT_render._tmp_dir or "", ignore_errors=True)
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

    def _load_result(self, context):
        output_path  = ANACAPA_OT_render._output_path
        tmp_dir      = ANACAPA_OT_render._tmp_dir
        settings     = context.scene.anacapa

        if not output_path or not os.path.exists(output_path):
            self.report({'ERROR'}, f"No EXR at {output_path}")
            shutil.rmtree(tmp_dir or "", ignore_errors=True)
            return {'CANCELLED'}

        # Persist the EXR
        persist_path = os.path.join(bpy.app.tempdir, "anacapa_last_render.exr")
        shutil.copy2(output_path, persist_path)

        # If user specified a PNG output path, copy the preview PNG there too
        if settings.png_path:
            preview_path = ANACAPA_OT_render._preview_path
            if preview_path and os.path.exists(preview_path):
                shutil.copy2(preview_path, bpy.path.abspath(settings.png_path))

        shutil.rmtree(tmp_dir, ignore_errors=True)
        ANACAPA_OT_render._tmp_dir = None

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

        self.report({'INFO'}, f"Exporting USD to {usd_path}…")
        try:
            export_usd(usd_path, context)
        except Exception as e:
            self.report({'ERROR'}, f"USD export failed: {e}")
            return {'CANCELLED'}

        if not os.path.exists(usd_path):
            self.report({'ERROR'}, "USD export produced no file")
            return {'CANCELLED'}

        cmd, _ = build_command(executable, usd_path, settings,
                               width, height, output_path)
        cmd_str = shlex.join(cmd)

        print("\n" + "=" * 72)
        print(f"[Anacapa] Scene exported to:\n  {usd_path}")
        print(f"\n[Anacapa] Render command:\n  {cmd_str}")
        print("=" * 72 + "\n")

        self.report({'INFO'}, "Exported. Render command printed to System Console.")
        return {'FINISHED'}


_classes = [ANACAPA_OT_render, ANACAPA_OT_export_scene]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
