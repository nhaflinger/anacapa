import bpy


class AnacapaAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    executable_path: bpy.props.StringProperty(
        name="Anacapa Executable",
        description="Full path to the anacapa binary",
        subtype='FILE_PATH',
        default="anacapa",
    )

    def draw(self, context):
        self.layout.prop(self, "executable_path")


class AnacapaRenderSettings(bpy.types.PropertyGroup):
    # GPU
    interactive: bpy.props.BoolProperty(
        name="GPU (Interactive)",
        description="Use Metal or CUDA GPU backend for fast preview renders "
                    "(requires ANACAPA_ENABLE_METAL or ANACAPA_ENABLE_CUDA)",
        default=False,
    )

    # Core
    samples: bpy.props.IntProperty(
        name="Samples", default=64, min=1, max=16384)
    max_depth: bpy.props.IntProperty(
        name="Max Depth", default=8, min=1, max=64)
    integrator: bpy.props.EnumProperty(
        name="Integrator",
        items=[
            ("path", "Path",  "Unidirectional path tracer"),
            ("bdpt", "BDPT",  "Bidirectional path tracer"),
        ],
        default="path",
    )
    tile_size: bpy.props.IntProperty(
        name="Tile Size", default=64, min=8, max=512)
    num_threads: bpy.props.IntProperty(
        name="Threads", description="0 = auto", default=0, min=0)

    # Adaptive sampling
    adaptive: bpy.props.BoolProperty(
        name="Adaptive Sampling", default=True)
    adaptive_base_spp: bpy.props.IntProperty(
        name="Base SPP", description="0 = auto (spp/4, min 16)", default=0, min=0)

    # Environment
    env_path: bpy.props.StringProperty(
        name="Environment Map",
        description="Equirectangular HDRI (EXR or HDR)",
        subtype='FILE_PATH',
    )
    env_intensity: bpy.props.FloatProperty(
        name="Intensity", default=1.0, min=0.0)

    # Lighting tweaks
    light_angle: bpy.props.FloatProperty(
        name="Light Angle",
        description="Angular radius for directional lights in degrees (0=hard, 0.27=sun)",
        default=0.0, min=0.0, max=45.0,
    )
    firefly_clamp: bpy.props.FloatProperty(
        name="Firefly Clamp",
        description="Max luminance per path contribution (0 = off)",
        default=10.0, min=0.0,
    )
    override_lights: bpy.props.BoolProperty(
        name="Override Lights",
        description="Replace all scene lights with a single white directional light",
        default=False,
    )
    override_materials: bpy.props.BoolProperty(
        name="Override Materials",
        description="Replace all materials with white Lambertian",
        default=False,
    )

    # Depth of field (overrides USD camera values when both are non-zero)
    fstop: bpy.props.FloatProperty(
        name="F-Stop", description="0 = use USD camera value", default=0.0, min=0.0)
    focus_distance: bpy.props.FloatProperty(
        name="Focus Distance", description="0 = use USD camera value", default=0.0, min=0.0)

    # Motion blur
    shutter_open: bpy.props.FloatProperty(name="Shutter Open", default=0.0)
    shutter_close: bpy.props.FloatProperty(
        name="Shutter Close",
        description="Set > Shutter Open to enable motion blur",
        default=0.0,
    )

    # Denoising / AOVs
    denoise: bpy.props.BoolProperty(name="Denoise (OIDN)", default=False)
    write_aovs: bpy.props.BoolProperty(
        name="Write AOVs",
        description="Include albedo and normals layers in the EXR",
        default=False,
    )

    # PNG preview output
    png_path: bpy.props.StringProperty(
        name="PNG Output",
        description="Write an ACES-tonemapped sRGB PNG alongside the EXR",
        subtype='FILE_PATH',
    )
    exposure: bpy.props.FloatProperty(
        name="Exposure", description="EV adjustment for PNG output", default=0.0)

    # Camera
    camera_path: bpy.props.StringProperty(
        name="Camera USD Path",
        description="USD prim path of camera to use (e.g. /World/RenderCam). "
                    "Leave empty to use the first camera found.",
    )


def register():
    bpy.utils.register_class(AnacapaAddonPreferences)
    bpy.utils.register_class(AnacapaRenderSettings)
    bpy.types.Scene.anacapa = bpy.props.PointerProperty(type=AnacapaRenderSettings)


def unregister():
    del bpy.types.Scene.anacapa
    bpy.utils.unregister_class(AnacapaRenderSettings)
    bpy.utils.unregister_class(AnacapaAddonPreferences)
