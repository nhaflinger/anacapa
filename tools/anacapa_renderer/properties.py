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
    gpu_assist: bpy.props.BoolProperty(
        name="GPU Assist",
        description="Use Metal or CUDA GPU backend for rendering "
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
        description="Trace photons into SSS surfaces for correct color bleeding. "
                    "Subsurface Radius is set per-material in the Anacapa material panel.",
        default=False,
    )
    num_photons: bpy.props.IntProperty(
        name="Photon Count",
        description="Total photons emitted from lights per frame",
        default=500000, min=1000, max=10000000,
    )
    photon_radius: bpy.props.FloatProperty(
        name="Caustic Search Radius",
        description="Search radius for caustic photon density estimation (world units). "
                    "Does NOT affect SSS — set SSS radius per-material. "
                    "Smaller = sharper but noisier caustics; larger = blurrier but cleaner.",
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

    # Nishita sky
    use_sky: bpy.props.BoolProperty(
        name="Nishita Sky",
        description="Replace the HDRI environment with a physically-based Nishita sky. "
                    "Provides sun + atmosphere illumination without an HDR file.",
        default=False,
    )
    sky_sun_elevation: bpy.props.FloatProperty(
        name="Sun Elevation",
        description="Angle of the sun above the horizon in degrees (0 = horizon, 90 = overhead)",
        default=45.0, min=-10.0, max=90.0, soft_min=0.0,
    )
    sky_sun_azimuth: bpy.props.FloatProperty(
        name="Sun Azimuth",
        description="Compass direction of the sun in degrees (0 = north/+Z, 90 = east/+X, 180 = south/-Z)",
        default=180.0, min=0.0, max=360.0,
    )
    sky_sun_intensity: bpy.props.FloatProperty(
        name="Sun Intensity",
        description="Radiance multiplier for the sun and sky",
        default=1.0, min=0.0, soft_max=10.0,
    )
    sky_sun_disc_size: bpy.props.FloatProperty(
        name="Sun Disc Size",
        description="Angular radius of the visible sun disc in degrees (~0.27 = real sun, "
                    "2.0 = Blender default). Larger values produce more scene illumination "
                    "and softer shadows as well as a bigger visible disc.",
        default=2.0, min=0.01, max=5.0,
    )
    sky_altitude: bpy.props.FloatProperty(
        name="Altitude",
        description="Viewer altitude above sea level in metres. Higher altitudes produce a "
                    "darker, bluer zenith and less horizon haze.",
        default=0.0, min=0.0, max=60000.0, soft_max=5000.0,
    )
    sky_air_density: bpy.props.FloatProperty(
        name="Air Density",
        description="Rayleigh scattering scale (1.0 = Earth standard atmosphere). "
                    "Lower = thinner air, blacker sky; higher = denser scattering.",
        default=1.0, min=0.0, soft_max=5.0,
    )
    sky_dust_density: bpy.props.FloatProperty(
        name="Dust Density",
        description="Mie (aerosol) scattering scale (1.0 = standard haze). "
                    "Higher = more haze, whiter horizon, hazier sun disc.",
        default=1.0, min=0.0, soft_max=10.0,
    )
    sky_ozone_density: bpy.props.FloatProperty(
        name="Ozone Density",
        description="Ozone absorption layer scale (1.0 = standard). "
                    "Controls how blue/cyan the zenith appears; higher = more ozone absorption.",
        default=1.0, min=0.0, soft_max=5.0,
    )
    sky_transparent_bg: bpy.props.BoolProperty(
        name="Transparent Background",
        description="Render sky as transparent (alpha=0) for compositing. "
                    "The sky still illuminates the scene — only the visible "
                    "sky dome background is made transparent. EXR output will "
                    "include an alpha channel.",
        default=False,
    )

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

    # Depth of field
    use_dof: bpy.props.BoolProperty(
        name="Enable DOF",
        description="Export depth-of-field settings to Anacapa. When off, pinhole camera is used regardless of Blender camera settings",
        default=False,
    )
    fstop: bpy.props.FloatProperty(
        name="F-Stop", description="0 = use Blender camera value", default=0.0, min=0.0)
    focus_distance: bpy.props.FloatProperty(
        name="Focus Distance", description="0 = use Blender camera value", default=0.0, min=0.0)

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

    # Viewer / display driver
    use_viewer: bpy.props.BoolProperty(
        name="Use Viewer",
        description="Send live tile updates to the Anacapa viewer over a socket. "
                    "When enabled, rendering auto-launches the viewer if it is not "
                    "already running, then passes --display so tiles stream directly "
                    "into the viewer as they finish.",
        default=False,
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

    # Per-material: caustic-generator opt-in (drives inputs:anacapa_caustic on
    # UsdPreviewSurface).  Only meaningful in --integrator photon mode.
    bpy.types.Material.anacapa_caustic = bpy.props.BoolProperty(
        name="Caustic Generator",
        description="Flag this material as a caustic-generating surface. "
                    "In --integrator photon, the photon map will deposit "
                    "caustics through it and NEE will skip it. Other "
                    "transparent materials are unaffected. Off by default.",
        default=False,
    )
    bpy.types.Material.anacapa_caustic_radius = bpy.props.FloatProperty(
        name="Caustic Radius",
        description="Search radius for caustic photon density estimation (world units). "
                    "Smaller = sharper but noisier caustics; larger = blurrier but cleaner. "
                    "0 = use the global scene default.",
        default=0.0, min=0.0, soft_max=2.0, precision=4,
    )

    # Per-material: subsurface scattering (drives inputs:subsurface_weight,
    # subsurface_color, subsurface_radius on UsdPreviewSurface via post-process).
    # Only meaningful in --integrator photon mode.
    bpy.types.Material.anacapa_subsurface = bpy.props.FloatProperty(
        name="Subsurface Weight",
        description="Subsurface scattering weight [0–1]. Replaces the diffuse "
                    "layer in proportion; the photon map fills in the scattered "
                    "contribution. Only active in --integrator photon mode.",
        default=0.0, min=0.0, max=1.0, precision=3,
    )
    bpy.types.Material.anacapa_subsurface_color = bpy.props.FloatVectorProperty(
        name="Subsurface Color",
        description="Scattering albedo — the color of light that diffuses "
                    "through the material (e.g. warm skin tones, waxy yellow).",
        subtype='COLOR', default=(1.0, 1.0, 1.0), min=0.0, max=1.0,
    )
    bpy.types.Material.anacapa_subsurface_radius = bpy.props.FloatProperty(
        name="Subsurface Radius",
        description="Mean free path (scattering length). Effective scattering "
                    "distance = Radius × Scale. Match the value from your "
                    "Principled BSDF Subsurface Radius.",
        default=0.1, min=0.0001, soft_max=1.0, precision=4,
    )
    bpy.types.Material.anacapa_subsurface_scale = bpy.props.FloatProperty(
        name="Subsurface Scale",
        description="World-space scale multiplier for Subsurface Radius. "
                    "Effective scattering distance = Radius × Scale. "
                    "Match the Scale value from your Principled BSDF.",
        default=1.0, min=0.001, soft_max=10.0, precision=3,
    )
    bpy.types.Material.anacapa_subsurface_strength = bpy.props.FloatProperty(
        name="Subsurface Strength",
        description="Independent amplifier for the SSS radiance contribution. "
                    "1.0 = physically based; increase to boost the SSS effect "
                    "beyond what the weight and radius alone produce.",
        default=1.0, min=0.0, soft_max=10.0, precision=3,
    )


def unregister():
    del bpy.types.Material.anacapa_subsurface_strength
    del bpy.types.Material.anacapa_subsurface_scale
    del bpy.types.Material.anacapa_subsurface_radius
    del bpy.types.Material.anacapa_subsurface_color
    del bpy.types.Material.anacapa_subsurface
    del bpy.types.Material.anacapa_caustic_radius
    del bpy.types.Material.anacapa_caustic
    del bpy.types.Scene.anacapa
    bpy.utils.unregister_class(AnacapaRenderSettings)
    bpy.utils.unregister_class(AnacapaAddonPreferences)
