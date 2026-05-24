import bpy
import os
import shlex

from .export import (get_executable, get_viewer_executable, export_usd,
                      build_command, get_scenes_dir, get_cache_dir,
                      substitute_frame_tokens,
                      is_viewer_running, launch_viewer)


class ANACAPA_OT_launch_viewer(bpy.types.Operator):
    """Launch the Anacapa viewer and start listening for a renderer connection"""
    bl_idname = "anacapa.launch_viewer"
    bl_label  = "Launch Viewer"

    def execute(self, context):
        viewer_path = get_viewer_executable(context)
        if not os.path.exists(viewer_path):
            self.report({'ERROR'}, f"Viewer not found: {viewer_path}")
            return {'CANCELLED'}
        if is_viewer_running():
            self.report({'INFO'}, "Viewer is already running")
            return {'FINISHED'}
        launch_viewer(viewer_path)
        if is_viewer_running():
            self.report({'INFO'}, "Viewer launched")
        else:
            self.report({'WARNING'}, "Viewer launched but socket not ready yet")
        return {'FINISHED'}


class ANACAPA_OT_render(bpy.types.Operator):
    """Render using Anacapa — equivalent to F12, uses the render-mode depsgraph
    for correct GN instance counts.  Pre-render steps (hair export, CURVES
    hiding) run via the render_pre handler registered in export.py."""
    bl_idname = "anacapa.render"
    bl_label  = "Render"

    def execute(self, context):
        if context.scene.render.engine != 'ANACAPA':
            self.report({'ERROR'}, "Scene render engine is not set to Anacapa")
            return {'CANCELLED'}
        return bpy.ops.render.render('INVOKE_DEFAULT')


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

        _s_open2, _s_close2 = 0.0, 0.0
        if getattr(settings, 'use_motion_blur', False):
            _sh2  = getattr(settings, 'motion_blur_shutter', 0.5)
            _pos2 = getattr(settings, 'motion_blur_position', 'CENTER')
            if _sh2 > 0:
                if _pos2 == 'START':
                    _s_open2, _s_close2 = 0.0, _sh2
                elif _pos2 == 'END':
                    _s_open2, _s_close2 = -_sh2, 0.0
                else:
                    _s_open2, _s_close2 = -_sh2 / 2.0, _sh2 / 2.0

        self.report({'INFO'}, f"Exporting USD to {usd_path}…")
        try:
            export_usd(usd_path, context,
                       shutter_open=_s_open2, shutter_close=_s_close2)
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
