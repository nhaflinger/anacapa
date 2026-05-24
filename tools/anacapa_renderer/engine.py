import bpy
import os
import subprocess
import shutil
import time


class AnacapaRenderEngine(bpy.types.RenderEngine):
    bl_idname = "ANACAPA"
    bl_label  = "Anacapa"
    bl_use_preview        = False
    bl_use_eevee_viewport = True

    def render(self, depsgraph):
        scene = depsgraph.scene
        # Blender calls render() once per view layer; only run for the first.
        if depsgraph.view_layer != scene.view_layers[0]:
            result = self.begin_result(0, 0,
                int(scene.render.resolution_x * scene.render.resolution_percentage / 100),
                int(scene.render.resolution_y * scene.render.resolution_percentage / 100))
            self.end_result(result)
            return

        print("[Anacapa RenderEngine] render() called")
        result = None
        try:
            settings = scene.anacapa
            print(f"[Anacapa RenderEngine] scene={scene.name} "
                  f"engine={scene.render.engine}")

            scale  = scene.render.resolution_percentage / 100.0
            width  = int(scene.render.resolution_x * scale)
            height = int(scene.render.resolution_y * scale)

            blend_path = bpy.data.filepath
            if not blend_path:
                self.report({'ERROR'},
                            "Save the blend file before rendering.")
                return

            from . import export as export_mod

            blend_stem  = os.path.splitext(os.path.basename(blend_path))[0]
            scenes_dir  = export_mod.get_scenes_dir(blend_path)
            usd_path    = os.path.join(scenes_dir, blend_stem + "_scene.usdc")

            # Write render output directly to Blender's temp dir — no mkdtemp,
            # no cleanup needed, no leftover directories.
            output_path = os.path.join(bpy.app.tempdir, "anacapa_render_output.exr")

            print(f"[Anacapa RenderEngine] usd={usd_path}")
            print(f"[Anacapa RenderEngine] output={output_path}")

            # Shutter interval
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

            # Minimal context substitute — export_usd uses context.scene for
            # motion-blur frame seeks and get_executable() for settings lookup.
            class _Ctx:
                pass
            ctx = _Ctx()
            ctx.scene       = scene
            ctx.view_layer  = depsgraph.view_layer
            ctx.preferences = bpy.context.preferences

            # Force re-export so the render-mode depsgraph is always used.
            state = export_mod._state()
            state["dirty_scene"]     = True
            state["dirty_transform"] = True
            state["suppress_dirty"]  = True

            self.update_stats("Anacapa", "Exporting scene...")
            print("[Anacapa RenderEngine] Starting USD export...")

            export_mod.export_usd(usd_path, ctx,
                                  shutter_open=s_open,
                                  shutter_close=s_close,
                                  depsgraph_override=depsgraph)

            if not os.path.exists(usd_path):
                self.report({'ERROR'}, "USD export produced no file")
                return

            print("[Anacapa RenderEngine] Export complete, building command...")

            # Pick up hair paths written by the render_pre handler.
            rdata          = export_mod._render_pre_data()
            abc_path       = rdata.get('abc_path')
            matassign_paths = rdata.get('matassign_paths')

            executable = export_mod.get_executable(ctx)
            cmd = export_mod.build_command(executable, usd_path, settings,
                                           width, height, output_path,
                                           curves_path=abc_path,
                                           matassign_paths=matassign_paths,
                                           frame=scene.frame_current)

            # Viewer support — launch if needed and wire up --display.
            if getattr(settings, 'use_viewer', False):
                viewer_path = export_mod.get_viewer_executable(ctx)
                if os.path.exists(viewer_path):
                    if not export_mod.is_viewer_running():
                        export_mod.launch_viewer(viewer_path)
                    if export_mod.is_viewer_running():
                        cmd.append("--display")

            # Determine whether the custom viewer is receiving updates.
            using_viewer = '--display' in cmd

            # Progressive preview EXR — written by anacapa after each tile batch.
            # Always enable so Blender's F12 viewer updates live, unless the custom
            # viewer is active (in that case anacapa sends via socket instead).
            preview_path = None
            if not using_viewer:
                preview_path = os.path.join(bpy.app.tempdir, "anacapa_preview.exr")
                cmd += ["--preview-exr", preview_path]

            print(f"[Anacapa RenderEngine] cmd: {' '.join(cmd)}")
            self.update_stats("Anacapa", "Rendering...")

            # Open the render result slot now so update_result() works during render.
            result = self.begin_result(0, 0, width, height)
            last_preview_mtime = 0.0

            proc = subprocess.Popen(cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)

            while proc.poll() is None:
                if self.test_break():
                    proc.kill()
                    print("[Anacapa RenderEngine] Cancelled.")
                    self.end_result(result, cancel=True)
                    result = None
                    return
                line = proc.stdout.readline()
                if line:
                    print(line, end="")
                else:
                    time.sleep(0.05)

                # Progressive Blender viewer update.
                if preview_path:
                    try:
                        mtime = os.path.getmtime(preview_path)
                        if mtime > last_preview_mtime:
                            last_preview_mtime = mtime
                            result.layers[0].load_from_file(preview_path)
                            self.update_result(result)
                    except Exception:
                        pass

            for line in proc.stdout:
                print(line, end="")

            if proc.returncode != 0:
                self.report({'ERROR'},
                            f"Anacapa exited with code {proc.returncode}")
                self.end_result(result, cancel=True)
                result = None
                return

            if not os.path.exists(output_path):
                self.report({'ERROR'}, "Anacapa produced no output file")
                self.end_result(result, cancel=True)
                result = None
                return

            print("[Anacapa RenderEngine] Loading result into Blender...")

            # If the user set an output path, copy there with frame tokens.
            if settings.output_path:
                user_path = export_mod.substitute_frame_tokens(
                    bpy.path.abspath(settings.output_path),
                    scene.frame_current)
                os.makedirs(os.path.dirname(user_path) or ".", exist_ok=True)
                shutil.copy2(output_path, user_path)
                print(f"[Anacapa RenderEngine] Saved EXR -> {user_path}")

            # Load final EXR into the render result slot.
            try:
                result.layers[0].load_from_file(output_path)
            except Exception as load_err:
                print(f"[Anacapa RenderEngine] load_from_file error: {load_err}")
                self.report({'WARNING'}, f"Could not load result: {load_err}")
            self.end_result(result)
            result = None

            # When not using the custom viewer, also load into bpy.data.images
            # so the render persists if Blender resets the render-result slot.
            if not using_viewer:
                _output = output_path

                def _show_anacapa_render():
                    try:
                        name = "Anacapa Render"
                        img = bpy.data.images.get(name)
                        if img:
                            img.filepath = _output
                            img.reload()
                        else:
                            img = bpy.data.images.load(_output)
                            img.name = name
                        img.use_fake_user = True
                        for window in bpy.context.window_manager.windows:
                            for area in window.screen.areas:
                                if area.type == 'IMAGE_EDITOR':
                                    area.spaces.active.image = img
                    except Exception as _e:
                        print(f"[Anacapa RenderEngine] show error: {_e}")
                    return None  # do not repeat

                bpy.app.timers.register(_show_anacapa_render, first_interval=0.1)

            print("[Anacapa RenderEngine] Done.")

        except Exception as e:
            import traceback
            print(f"[Anacapa RenderEngine] EXCEPTION: {e}")
            traceback.print_exc()
            self.report({'ERROR'}, f"Anacapa render engine error: {e}")
            try:
                from . import export as _exp
                _exp._state()["suppress_dirty"] = False
            except Exception:
                pass
        finally:
            # Guarantee end_result is called if an exception escaped after
            # begin_result was already invoked.
            if result is not None:
                try:
                    self.end_result(result, cancel=True)
                except Exception:
                    pass


def register():
    bpy.utils.register_class(AnacapaRenderEngine)
    from . import export
    export.register_dirty_handler()


def unregister():
    from . import export
    export.unregister_dirty_handler()
    bpy.utils.unregister_class(AnacapaRenderEngine)
