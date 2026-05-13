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
        name="GPU compute",
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
            ("path",   "Path",       "Unidirectional path tracer"),
            ("bdpt",   "BDPT",       "Bidirectional path tracer"),
            ("photon", "Path + Photon Map", "Path tracer + photon map passes (caustics, SSS, volumes)"),
        ],
        default="path",
    )

    # Photon map settings (visible when integrator == photon)
    pm_caustics: bpy.props.BoolProperty(
        name="Caustic Map",
        description="Trace photons through specular surfaces to produce sharp caustics "
                    "and correct glass shadows",
        default=True,
    )
    pm_subsurface: bpy.props.BoolProperty(
        name="Subsurface Map",
        description="Trace photons into SSS volumes for correct color bleeding and "
                    "scattering (not yet implemented)",
        default=False,
    )
    num_photons: bpy.props.IntProperty(
        name="Photon Count",
        description="Total photons emitted from lights per frame",
        default=500000, min=1000, max=10000000,
    )
    photon_radius: bpy.props.FloatProperty(
        name="Search Radius",
        description="Photon density estimate search radius in world units. "
                    "Smaller = sharper but noisier; larger = blurrier but cleaner",
        default=0.1, min=0.001, max=10.0,
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

    # Film — pixel reconstruction filter
    pixel_filter: bpy.props.EnumProperty(
        name="Pixel Filter",
        description="Reconstruction filter applied when projecting samples to "
                    "pixels. Mitchell-Netravali is a balanced default; "
                    "Blackman-Harris matches Cycles' look",
        items=[
            ("box",             "Box",             "Single-pixel box (no AA — fastest, blockiest)"),
            ("triangle",        "Triangle",        "Tent filter (linear)"),
            ("gaussian",        "Gaussian",        "Truncated Gaussian"),
            ("mitchell",        "Mitchell-Netravali", "Cubic with mild ringing — production default"),
            ("blackman-harris", "Blackman-Harris", "Smooth window — Cycles-style"),
            ("catmull-rom",     "Catmull-Rom",     "Sharper cubic with stronger ringing"),
            ("lanczos",         "Lanczos",         "Windowed sinc — sharpest, can ring"),
        ],
        default="mitchell",
    )
    filter_width: bpy.props.FloatProperty(
        name="Filter Width",
        description="Filter radius in pixels. 0 = use the filter's default "
                    "(Box=0.5, Triangle=1.0, Gaussian=1.5, Mitchell=2.0, "
                    "Blackman-Harris=1.5, Catmull-Rom=2.0, Lanczos=4.0)",
        default=0.0, min=0.0, soft_max=8.0,
    )

    # Hair
    hair_tess_steps: bpy.props.IntProperty(
        name="Hair Tess Steps",
        description="Quads per cubic Bézier hair segment. Higher = smoother "
                    "ribbons; 1–2 for fast previews, 4 default, 8+ for hero close-ups",
        default=4, min=1, max=32,
    )

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
    use_motion_blur: bpy.props.BoolProperty(
        name="Motion Blur",
        description="Enable transformation motion blur",
        default=False,
    )
    motion_blur_shutter: bpy.props.FloatProperty(
        name="Shutter",
        description="Shutter duration in frames (e.g. 0.5 = half-frame, 1.0 = full frame)",
        default=0.5,
        min=0.0,
        max=2.0,
    )
    motion_blur_position: bpy.props.EnumProperty(
        name="Position",
        description="How the shutter window is positioned relative to the render frame",
        items=[
            ("START",  "Start on Frame",  "Shutter opens at the render frame"),
            ("CENTER", "Center on Frame", "Shutter centered on the render frame"),
            ("END",    "End on Frame",    "Shutter closes at the render frame"),
        ],
        default="CENTER",
    )
    # Legacy — kept so old .blend files don't lose data; not shown in UI
    shutter_open: bpy.props.FloatProperty(name="Shutter Open", default=0.0)
    shutter_close: bpy.props.FloatProperty(name="Shutter Close", default=0.0)

    # Denoising / AOVs
    denoise: bpy.props.BoolProperty(name="Denoise (OIDN)", default=False)
    write_aovs: bpy.props.BoolProperty(
        name="Write AOVs",
        description="Include albedo and normals layers in the EXR",
        default=False,
    )

    # EXR output path.  $F tokens substitute the current frame number:
    #   $F  -> "42"        (no padding)
    #   $F2 -> "42"         $F3 -> "042"     $F4 -> "0042"   etc.
    # See substitute_frame_tokens() in export.py.  Empty = don't persist
    # the EXR (it still goes to a temp dir for in-Blender display).
    output_path: bpy.props.StringProperty(
        name="EXR Output",
        description='EXR output path. Use "$F4" for 4-padded frame numbers '
                    '(e.g. "render.$F4.exr" -> "render.0042.exr"). '
                    'Tokens: $F (no padding), $F2..$F8 (zero-padded).',
        subtype='FILE_PATH',
    )

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
