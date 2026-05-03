import bpy


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class ANACAPA_PT_sampling(bpy.types.Panel):
    bl_label       = "Sampling"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "render"
    COMPAT_ENGINES = {'ANACAPA'}

    @classmethod
    def poll(cls, context):
        return context.engine == 'ANACAPA'

    def draw(self, context):
        layout = self.layout
        s = context.scene.anacapa

        layout.operator("anacapa.render", text="Render", icon='RENDER_STILL')
        layout.separator()
        layout.use_property_split = True

        layout.prop(s, "interactive")
        layout.separator()
        layout.prop(s, "samples")
        layout.prop(s, "max_depth")
        layout.prop(s, "integrator")
        layout.prop(s, "tile_size")
        layout.prop(s, "num_threads")


class ANACAPA_PT_adaptive(bpy.types.Panel):
    bl_label       = "Adaptive Sampling"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "render"
    bl_parent_id   = 'ANACAPA_PT_sampling'
    bl_options     = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'ANACAPA'}

    @classmethod
    def poll(cls, context):
        return context.engine == 'ANACAPA'

    def draw_header(self, context):
        self.layout.prop(context.scene.anacapa, "adaptive", text="")

    def draw(self, context):
        layout = self.layout
        s = context.scene.anacapa
        layout.use_property_split = True
        layout.active = s.adaptive
        layout.prop(s, "adaptive_base_spp")


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

class ANACAPA_PT_lighting(bpy.types.Panel):
    bl_label       = "Lighting"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "render"
    COMPAT_ENGINES = {'ANACAPA'}

    @classmethod
    def poll(cls, context):
        return context.engine == 'ANACAPA'

    def draw(self, context):
        layout = self.layout
        s = context.scene.anacapa
        layout.use_property_split = True

        layout.prop(s, "env_path")
        row = layout.row()
        row.active = bool(s.env_path)
        row.prop(s, "env_intensity")

        layout.separator()
        layout.prop(s, "light_angle")
        layout.prop(s, "firefly_clamp")

        layout.separator()
        layout.prop(s, "override_lights")
        layout.prop(s, "override_materials")


# ---------------------------------------------------------------------------
# Depth of Field & Motion Blur
# ---------------------------------------------------------------------------

class ANACAPA_PT_camera(bpy.types.Panel):
    bl_label       = "Camera"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "render"
    bl_options     = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'ANACAPA'}

    @classmethod
    def poll(cls, context):
        return context.engine == 'ANACAPA'

    def draw(self, context):
        layout = self.layout
        s = context.scene.anacapa
        layout.use_property_split = True

        layout.label(text="Depth of Field")
        layout.prop(s, "fstop")
        col = layout.column()
        col.active = s.fstop > 0
        col.prop(s, "focus_distance")

        layout.separator()
        layout.label(text="Motion Blur")
        layout.prop(s, "shutter_open")
        layout.prop(s, "shutter_close")

        layout.separator()
        layout.prop(s, "camera_path")


# ---------------------------------------------------------------------------
# Output & Denoising
# ---------------------------------------------------------------------------

class ANACAPA_PT_output(bpy.types.Panel):
    bl_label       = "Output"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "render"
    COMPAT_ENGINES = {'ANACAPA'}

    @classmethod
    def poll(cls, context):
        return context.engine == 'ANACAPA'

    def draw(self, context):
        layout = self.layout
        s = context.scene.anacapa
        layout.use_property_split = True

        layout.prop(s, "denoise")
        layout.prop(s, "write_aovs")

        layout.separator()
        layout.prop(s, "png_path")
        row = layout.row()
        row.active = bool(s.png_path)
        row.prop(s, "exposure")

        layout.separator()
        layout.operator("anacapa.export_scene", icon='EXPORT')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = [
    ANACAPA_PT_sampling,
    ANACAPA_PT_adaptive,
    ANACAPA_PT_lighting,
    ANACAPA_PT_camera,
    ANACAPA_PT_output,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
