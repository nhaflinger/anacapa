"""
Anacapa Renderer — Blender addon
Registers the Anacapa path tracer as a Blender render engine.

Installation:
    Zip the anacapa_renderer/ folder and install via
    Edit > Preferences > Add-ons > Install from Disk.

Requirements:
    Set the Anacapa executable path in Preferences > Add-ons > Anacapa Renderer.
"""

bl_info = {
    "name":        "Anacapa Renderer",
    "author":      "Anacapa",
    "version":     (0, 1, 0),
    "blender":     (3, 5, 0),
    "location":    "Render Properties",
    "description": "Anacapa path tracer render engine",
    "category":    "Render",
}

if "bpy" in dir():
    # Reload submodules when the addon is reloaded in a running Blender session.
    # Reset the prep module cache so changes to the prep script are picked up.
    import importlib
    from . import properties, export, engine, operators, ui
    importlib.reload(properties)
    importlib.reload(export)
    export._load_prep_module._mod = None   # invalidate prep module cache
    export.mark_all_dirty()                # force full re-export after reload
    importlib.reload(engine)
    importlib.reload(operators)
    importlib.reload(ui)
else:
    from . import properties, export, engine, operators, ui

import bpy


def register():
    properties.register()
    engine.register()
    operators.register()
    ui.register()


def unregister():
    ui.unregister()
    operators.unregister()
    engine.unregister()
    properties.unregister()
