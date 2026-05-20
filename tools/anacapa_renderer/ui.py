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

        row = layout.row(align=True)
        row.prop(s, "use_viewer", text="Use Viewer",
                 icon='RESTRICT_VIEW_OFF' if s.use_viewer else 'RESTRICT_VIEW_ON')
        row.operator("anacapa.launch_viewer", text="", icon='WINDOW')

        layout.separator()
        layout.use_property_split = True

        layout.prop(s, "gpu_assist")
        layout.separator()
        layout.prop(s, "samples")
        layout.prop(s, "max_depth")
        layout.prop(s, "integrator")
        if s.integrator == "photon":
            box = layout.box()
            box.prop(s, "pm_caustics")
            box.prop(s, "pm_subsurface")
            col = box.column(align=True)
            col.prop(s, "num_photons")
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
# Film — pixel reconstruction filter
# ---------------------------------------------------------------------------

class ANACAPA_PT_film(bpy.types.Panel):
    bl_label       = "Film"
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

        layout.prop(s, "pixel_filter")
        layout.prop(s, "filter_width")


# ---------------------------------------------------------------------------
# Hair
# ---------------------------------------------------------------------------

class ANACAPA_PT_hair(bpy.types.Panel):
    bl_label       = "Hair"
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
        layout.prop(s, "hair_tess_steps")


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------

class ANACAPA_PT_particles(bpy.types.Panel):
    bl_label       = "Particles"
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

        layout.operator("anacapa.bake_particles", icon='PHYSICS')
        layout.label(text="Bake GN simulations before rendering.", icon='INFO')



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

        row = layout.row()
        row.prop(s, "use_dof", text="Depth of Field")
        col = layout.column()
        col.active = s.use_dof
        col.prop(s, "fstop")
        sub = col.column()
        sub.active = s.use_dof and s.fstop > 0
        sub.prop(s, "focus_distance")

        layout.separator()
        row = layout.row()
        row.prop(s, "use_motion_blur")
        col = layout.column()
        col.active = s.use_motion_blur
        col.prop(s, "motion_blur_shutter")
        col.prop(s, "motion_blur_position")

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
        layout.prop(s, "output_path")

        layout.separator()
        layout.operator("anacapa.export_scene", icon='EXPORT')


# ---------------------------------------------------------------------------
# Material — per-material caustic-generator toggle
# ---------------------------------------------------------------------------

class ANACAPA_PT_material(bpy.types.Panel):
    bl_label       = "Anacapa"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "material"
    COMPAT_ENGINES = {'ANACAPA'}

    @classmethod
    def poll(cls, context):
        return (context.engine == 'ANACAPA'
                and context.material is not None)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        mat = context.material

        # -- Caustics ----------------------------------------------------------
        box = layout.box()
        box.label(text="Caustics", icon='LIGHT_SUN')
        box.prop(mat, "anacapa_caustic", text="Enable Caustic Generator")
        col = box.column(align=True)
        col.prop(mat, "anacapa_caustic_radius", text="Caustic Radius")
        box.label(text="Photon integrator only.  Radius 0 = use scene default.",
                  icon='INFO')

        layout.separator()

        # -- Subsurface Scattering ---------------------------------------------
        box = layout.box()
        box.label(text="Subsurface Scattering", icon='MATFLUID')
        col = box.column(align=True)
        col.prop(mat, "anacapa_subsurface",       text="Weight")
        if mat.anacapa_subsurface > 0.0:
            col.prop(mat, "anacapa_subsurface_color",    text="Color")
            col.prop(mat, "anacapa_subsurface_radius",   text="Radius")
            col.prop(mat, "anacapa_subsurface_scale",    text="Scale")
            col.prop(mat, "anacapa_subsurface_strength", text="Strength")
            box.label(text="Photon integrator only.  Effective distance = Radius × Scale.",
                      icon='INFO')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = [
    ANACAPA_PT_sampling,
    ANACAPA_PT_adaptive,
    ANACAPA_PT_film,
    ANACAPA_PT_hair,
    ANACAPA_PT_particles,
    ANACAPA_PT_lighting,
    ANACAPA_PT_camera,
    ANACAPA_PT_output,
    ANACAPA_PT_material,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
