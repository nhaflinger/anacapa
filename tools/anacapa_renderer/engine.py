import bpy


class AnacapaRenderEngine(bpy.types.RenderEngine):
    bl_idname = "ANACAPA"
    bl_label  = "Anacapa"
    bl_use_preview        = False
    bl_use_eevee_viewport = True


def register():
    bpy.utils.register_class(AnacapaRenderEngine)
    from . import export
    export.register_dirty_handler()


def unregister():
    from . import export
    export.unregister_dirty_handler()
    bpy.utils.unregister_class(AnacapaRenderEngine)
