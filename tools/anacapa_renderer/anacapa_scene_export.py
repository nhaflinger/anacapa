"""
anacapa_scene_export.py — Blender depsgraph → anacapa binary scene format.

Reads the fully-evaluated scene from Blender's depsgraph without touching
the live scene (no modifiers baked, no objects hidden, no state mutation).
Writes a binary blob that the C++ SceneExporter converts to USD + MaterialX.

Binary format is defined in src/export/SceneFormat.h.
All values are little-endian.  Matrices are written row-major in anacapa's
column-vector convention (Y-up, right-handed) after converting from Blender's
Z-up coordinate system.
"""

import bpy
import struct
import math
import os
from mathutils import Matrix


# ---------------------------------------------------------------------------
# Coordinate system conversion: Blender Z-up → anacapa Y-up
# ---------------------------------------------------------------------------
# Rotation that maps Z-up to Y-up:  X→X, Y→-Z, Z→Y
_ZUP_TO_YUP = Matrix((
    (1,  0,  0,  0),
    (0,  0,  1,  0),
    (0, -1,  0,  0),
    (0,  0,  0,  1),
))


def _to_yup(matrix_world):
    """Convert a Blender Z-up world matrix to Y-up."""
    return _ZUP_TO_YUP @ matrix_world


def _is_pure_translucent(mat):
    """Return (True, [r,g,b]) if the material's Output node is directly wired
    to a Translucent BSDF, matching the old bpy.ops.wm.usd_export behaviour
    where ND_translucent_bsdf is the sole BSDF input to ND_surface."""
    if mat is None or mat.node_tree is None:
        return False, [0.8, 0.8, 0.8]
    for node in mat.node_tree.nodes:
        if node.type != 'OUTPUT_MATERIAL':
            continue
        for link in mat.node_tree.links:
            if link.to_node == node and link.to_socket.name == 'Surface':
                src = link.from_node
                if src.type == 'BSDF_TRANSLUCENT':
                    color = list(src.inputs['Color'].default_value[:3])
                    return True, color
    return False, [0.8, 0.8, 0.8]


def _get_sss_params(mat):
    """Return (weight, color[3], radius[3], scale, anisotropy) from the first
    Principled BSDF node that has subsurface enabled, or None if not present."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    for node in mat.node_tree.nodes:
        if node.type != 'BSDF_PRINCIPLED':
            continue
        # Blender 4+ uses "Subsurface Weight"; 3.x used "Subsurface"
        w_sock = (node.inputs.get('Subsurface Weight') or
                  node.inputs.get('Subsurface'))
        if w_sock is None:
            continue
        weight = float(w_sock.default_value) if not w_sock.links else 0.0
        if weight <= 0.0:
            return None

        # Blender 4+ dropped "Subsurface Color" — SSS uses Base Color instead
        c_sock = (node.inputs.get('Subsurface Color') or
                  node.inputs.get('Base Color'))
        color = list(c_sock.default_value[:3]) if (c_sock and not c_sock.links) else [0.8, 0.8, 0.8]

        r_sock = node.inputs.get('Subsurface Radius')
        radius = list(r_sock.default_value[:3]) if (r_sock and not r_sock.links) else [0.1, 0.1, 0.1]

        sc_sock = node.inputs.get('Subsurface Scale')
        scale = float(sc_sock.default_value) if (sc_sock and not sc_sock.links) else 1.0

        a_sock = node.inputs.get('Subsurface Anisotropy')
        aniso = float(a_sock.default_value) if (a_sock and not a_sock.links) else 0.0

        return (weight, color, radius, scale, aniso)
    return None


def _pack_str(s: str) -> bytes:
    b = s.encode('utf-8')
    return struct.pack('<I', len(b)) + b


def _pack_matrix(m) -> bytes:
    """Write a mathutils.Matrix as 16 row-major little-endian floats."""
    flat = [m[row][col] for row in range(4) for col in range(4)]
    return struct.pack('<16f', *flat)


def _pack_floats(seq) -> bytes:
    data = list(seq)
    return struct.pack(f'<{len(data)}f', *data)


def _pack_uints(seq) -> bytes:
    data = list(seq)
    return struct.pack(f'<{len(data)}I', *data)


def _collect_transforms(depsgraph, types=('MESH', 'LIGHT', 'CAMERA')):
    """Return {orig_name: yup_matrix} for all non-instanced objects of the
    given types in the current depsgraph state.  Instanced objects are skipped."""
    result = {}
    for inst in depsgraph.object_instances:
        if inst.is_instance:
            continue
        obj = inst.object
        if obj.original.hide_render:
            continue
        if obj.type in types:
            result[obj.original.name] = _to_yup(inst.matrix_world)
    return result


def _collect_instance_transforms(depsgraph):
    """Return {(parent_name, pid_tuple): yup_matrix} for instanced mesh objects
    whose parent has instance_type == 'COLLECTION' (i.e. regular collection instances).

    The composite key (parent_name, pid) is unique per instance: different collection
    instance objects have distinct parent names, and objects within one collection are
    distinguished by their persistent_id index (0 for the first prototype, 1 for the
    second, etc.).  GN/particle instances and plain object-instances are excluded."""
    result = {}
    for inst in depsgraph.object_instances:
        if not inst.is_instance:
            continue
        obj = inst.object
        if obj.original.hide_render:
            continue
        if obj.type != 'MESH':
            continue
        parent = inst.parent
        if parent is None:
            continue
        orig_parent = parent.original if hasattr(parent, 'original') else parent
        if getattr(orig_parent, 'instance_type', '') != 'COLLECTION':
            continue
        key = (parent.name, tuple(inst.persistent_id))
        result[key] = _to_yup(inst.matrix_world)
    return result


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_scene_binary(filepath: str, context,
                        shutter_open: float = 0.0,
                        shutter_close: float = 0.0) -> None:
    """
    Evaluate the scene via the depsgraph and write the binary scene blob to
    filepath.  The live Blender scene is never modified except for temporary
    frame seeks when motion blur is enabled (always restored before return).

    shutter_open / shutter_close are offsets from the current frame in frames,
    matching the --shutter-open / --shutter-close CLI arguments passed to anacapa.
    When shutter_close > shutter_open, two transform keys are written per object
    at the corresponding USD time codes so collectMotionKeys() in USDLoader
    produces motion-blur data.
    """
    depsgraph = context.evaluated_depsgraph_get()
    scene     = context.scene
    current_frame = scene.frame_current

    do_motion_blur = shutter_close > shutter_open + 1e-5

    mesh_data  = []
    light_data = []
    camera     = None

    # Track which non-instanced objects we have already exported so that
    # objects appearing in depsgraph.object_instances as both a "real" object
    # and an instance of themselves don't get written twice.
    seen_non_instance = set()

    for inst in depsgraph.object_instances:
        obj      = inst.object
        orig     = obj.original

        if orig.hide_render:
            continue

        if obj.type == 'MESH':
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            if mesh is None or len(mesh.polygons) == 0:
                eval_obj.to_mesh_clear()
                continue

            # calc_normals_split() was removed in Blender 4.1; corner_normals
            # replaced it and is always up-to-date after to_mesh().
            if hasattr(mesh, 'calc_normals_split'):
                mesh.calc_normals_split()

            mat_world = _to_yup(inst.matrix_world)

            positions = []
            for v in mesh.vertices:
                positions += [v.co.x, v.co.y, v.co.z]

            normals = []
            if hasattr(mesh, 'corner_normals'):
                for cn in mesh.corner_normals:
                    normals += [cn.vector.x, cn.vector.y, cn.vector.z]
            else:
                for loop in mesh.loops:
                    normals += [loop.normal.x, loop.normal.y, loop.normal.z]

            uvlayers = []
            for uv_layer in mesh.uv_layers:
                uvs = []
                for uv in uv_layer.data:
                    uvs += [uv.uv.x, uv.uv.y]
                uvlayers.append((uv_layer.name, uvs))

            loop_starts  = [p.loop_start     for p in mesh.polygons]
            loop_totals  = [p.loop_total     for p in mesh.polygons]
            mat_indices  = [p.material_index for p in mesh.polygons]
            vert_indices = [l.vertex_index   for l in mesh.loops]

            mat_names    = []
            mat_flags    = []
            mat_colors   = []
            mat_sss_data = []
            for slot in orig.material_slots:
                mat = slot.material
                mat_names.append(mat.name if mat else "")
                is_trans, color = _is_pure_translucent(mat)
                sss = _get_sss_params(mat)
                flags = 0
                if is_trans:
                    flags |= 1
                if sss is not None:
                    flags |= 2
                mat_flags.append(flags)
                mat_colors.extend(color)
                if sss is not None:
                    weight, sc, radius, scale, aniso = sss
                    mat_sss_data.extend([weight] + sc + radius + [scale, aniso])
                else:
                    mat_sss_data.extend([0.0, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 1.0, 0.0])

            mesh_data.append({
                'name':         obj.name,
                'orig_name':    orig.name,
                'is_instance':  inst.is_instance,
                'inst_parent_name': inst.parent.name if inst.is_instance and inst.parent else None,
                'inst_pid':     tuple(inst.persistent_id) if inst.is_instance else None,
                'matrix':       mat_world,       # current-frame transform (single key fallback)
                'positions':    positions,
                'normals':      normals,
                'uvlayers':     uvlayers,
                'loop_starts':  loop_starts,
                'loop_totals':  loop_totals,
                'mat_indices':  mat_indices,
                'vert_indices': vert_indices,
                'mat_names':    mat_names,
                'mat_flags':    mat_flags,
                'mat_colors':   mat_colors,
                'mat_sss_data': mat_sss_data,
            })

            eval_obj.to_mesh_clear()

        elif obj.type == 'LIGHT':
            if inst.is_instance:
                continue
            if orig.name in seen_non_instance:
                continue
            seen_non_instance.add(orig.name)

            ld  = orig.data
            mat_world = _to_yup(obj.matrix_world)

            type_map = {'SUN': 0, 'AREA': 1, 'POINT': 2, 'SPOT': 2}
            ltype    = type_map.get(ld.type, 0)

            params = [0.0, 0.0, 0.0, 0.0]
            if ld.type == 'SUN':
                params[0] = math.degrees(ld.angle)
            elif ld.type == 'AREA':
                params[1] = ld.size
                params[2] = ld.size_y if ld.shape in ('RECTANGLE', 'ELLIPSE') else ld.size
            elif ld.type in ('POINT', 'SPOT'):
                params[3] = ld.shadow_soft_size

            light_data.append({
                'type':      ltype,
                'name':      orig.name,
                'matrix':    mat_world,
                'color':     list(ld.color),
                'intensity': ld.energy / math.pi,
                'normalize': 1,
                'params':    params,
            })

        elif obj.type == 'CAMERA':
            if inst.is_instance:
                continue
            if orig.name in seen_non_instance:
                continue
            seen_non_instance.add(orig.name)

            # Prefer the scene's active camera; fall back to first found.
            if camera is not None and orig != scene.camera:
                continue

            cd        = orig.data
            mat_world = _to_yup(obj.matrix_world)
            camera = {
                'matrix':       mat_world,
                'lens':         cd.lens,
                'sensor_width': cd.sensor_width,
                'sensor_height': (cd.sensor_width / (scene.render.resolution_x / scene.render.resolution_y)
                                  if cd.sensor_fit == 'AUTO'
                                  else cd.sensor_height),
                'clip_start':   cd.clip_start,
                'clip_end':     cd.clip_end,
                'dof_distance': cd.dof.focus_distance,
                'dof_fstop':    cd.dof.aperture_fstop,
            }

    # -------------------------------------------------------------------------
    # Motion blur: sample transforms at shutter-open and shutter-close frames.
    # Only non-instanced objects (matched by orig_name) get per-object motion
    # keys.  Instanced objects (GN, particles) fall back to the current-frame
    # matrix for both keys — per-instance motion blur is not yet supported.
    # -------------------------------------------------------------------------
    if do_motion_blur:
        motion_times = [
            float(current_frame) + shutter_open,
            float(current_frame) + shutter_close,
        ]

        def _frame_set(t):
            fi = int(math.floor(t))
            fs = t - fi
            scene.frame_set(fi, subframe=fs)

        try:
            # Collect transforms at shutter_open
            _frame_set(motion_times[0])
            dg_open = context.evaluated_depsgraph_get()
            xf_open      = _collect_transforms(dg_open)
            xf_inst_open = _collect_instance_transforms(dg_open)

            # Collect transforms at shutter_close
            _frame_set(motion_times[1])
            dg_close = context.evaluated_depsgraph_get()
            xf_close      = _collect_transforms(dg_close)
            xf_inst_close = _collect_instance_transforms(dg_close)
        finally:
            scene.frame_set(current_frame)

        for m in mesh_data:
            if not m['is_instance']:
                # Non-instance: match by orig_name across frames
                oname = m['orig_name']
                mo = xf_open.get(oname,  m['matrix'])
                mc = xf_close.get(oname, m['matrix'])
            elif m['inst_parent_name'] is not None and m['inst_pid'] is not None:
                # Collection instance: key is (parent_name, pid) — unique because
                # each collection instancer has a distinct parent name, and objects
                # within one collection are ordered by persistent_id.
                # Non-collection instances (GN, particles) were excluded from
                # xf_inst_open/close, so their lookup falls back to m['matrix'].
                key = (m['inst_parent_name'], m['inst_pid'])
                mo = xf_inst_open.get(key,  m['matrix'])
                mc = xf_inst_close.get(key, m['matrix'])
            else:
                mo = mc = m['matrix']
            m['motion_matrices'] = [mo, mc]
            m['motion_times']    = motion_times
    else:
        # Static: single key at the current frame time code
        for m in mesh_data:
            m['motion_matrices'] = [m['matrix']]
            m['motion_times']    = [float(current_frame)]

    # -------------------------------------------------------------------------
    # Write binary
    # -------------------------------------------------------------------------
    with open(filepath, 'wb') as f:

        # SceneHeader (32 bytes)
        f.write(struct.pack('<5I12x',
                            0x41434E41,        # magic "ANCA"
                            4,                 # version
                            len(mesh_data),
                            len(light_data),
                            1 if camera else 0))

        # Mesh records
        for m in mesh_data:
            nv  = len(m['positions']) // 3
            nl  = len(m['normals'])   // 3
            np_ = len(m['loop_starts'])
            nuv = len(m['uvlayers'])
            nm  = len(m['mat_names'])
            nmk = len(m['motion_matrices'])

            # MeshHeader (no matrix — matrices follow immediately)
            f.write(struct.pack('<6I', nv, nl, np_, nuv, nm, nmk))

            # Motion keys: matrices then time codes
            for mat in m['motion_matrices']:
                f.write(_pack_matrix(mat))
            f.write(_pack_floats(m['motion_times']))

            # name
            f.write(_pack_str(m['name']))

            # material slot names
            for mname in m['mat_names']:
                f.write(_pack_str(mname))

            # material flags (bit 0 = translucent, bit 1 = sss), translucency colors, sss data
            f.write(_pack_uints(m['mat_flags']))
            f.write(_pack_floats(m['mat_colors']))
            f.write(_pack_floats(m['mat_sss_data']))

            # geometry
            f.write(_pack_floats(m['positions']))
            f.write(_pack_floats(m['normals']))

            # UV layers
            for uv_name, uvs in m['uvlayers']:
                f.write(_pack_str(uv_name))
                f.write(_pack_floats(uvs))

            # topology
            f.write(_pack_uints(m['loop_starts']))
            f.write(_pack_uints(m['loop_totals']))
            f.write(_pack_uints(m['mat_indices']))
            f.write(_pack_uints(m['vert_indices']))

        # Light records
        for l in light_data:
            f.write(struct.pack('<I', l['type']))
            f.write(_pack_matrix(l['matrix']))
            f.write(struct.pack('<3f', *l['color']))
            f.write(struct.pack('<f',  l['intensity']))
            f.write(struct.pack('<I',  l['normalize']))
            f.write(struct.pack('<4f', *l['params']))
            f.write(_pack_str(l['name']))

        # Camera record
        if camera:
            f.write(_pack_matrix(camera['matrix']))
            f.write(struct.pack('<7f',
                                camera['lens'],
                                camera['sensor_width'],
                                camera['sensor_height'],
                                camera['clip_start'],
                                camera['clip_end'],
                                camera['dof_distance'],
                                camera['dof_fstop']))

    print(f"[anacapa] scene binary v4: {len(mesh_data)} meshes "
          f"({'motion blur' if do_motion_blur else 'static'}), "
          f"{len(light_data)} lights, camera={'yes' if camera else 'no'} → {filepath}")
