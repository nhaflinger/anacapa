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
import array
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


def _uniform_gn_scale(m, inst):
    """Fix GN 'Instance on Points' non-uniform scale artifact: col0 carries
    per-point scale but col1/col2 do not.  Detect via sy≈sz but sx≠sy and
    rebuild with uniform scale sx.  Collection instances are skipped."""
    # Skip collection instances — they already have correct uniform scale.
    parent = inst.parent
    if parent is not None:
        orig_parent = parent.original if hasattr(parent, 'original') else parent
        if getattr(orig_parent, 'instance_type', '') == 'COLLECTION':
            return m

    sx = m.col[0].to_3d().length
    sy = m.col[1].to_3d().length
    sz = m.col[2].to_3d().length

    if sy < 1e-6:
        return m

    rel_tol = 0.01
    sy_sz_close = abs(sy - sz) < rel_tol * sy
    sx_differs  = abs(sx - sy) > rel_tol * sy

    if sy_sz_close and sx_differs:
        T, R, _S = m.decompose()
        return Matrix.LocRotScale(T, R, (sx, sx, sx))
    return m


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


def _get_mat_base_color(mat):
    """Return [r, g, b] of the Principled BSDF Base Color constant, or
    fall back to mat.diffuse_color, or [0.8, 0.8, 0.8] if unavailable.
    Only reads constant (unlinked) values; textured inputs fall back to default."""
    if mat is None:
        return [0.8, 0.8, 0.8]
    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue
            sock = node.inputs.get('Base Color')
            if sock and not sock.links:
                return list(sock.default_value[:3])
    if hasattr(mat, 'diffuse_color'):
        return list(mat.diffuse_color[:3])
    return [0.8, 0.8, 0.8]


def _get_sss_params(mat):
    """Return (weight, color[3], radius[3], scale, anisotropy) for SSS, or None.

    Priority:
    1. Custom anacapa_subsurface property set in the Anacapa material panel.
    2. Principled BSDF node 'Subsurface Weight' (or 'Subsurface' in older Blender).

    The custom property exists so artists can assign SSS parameters without
    modifying the Principled BSDF node (which drives the Cycles/OSL appearance).
    """
    if mat is None:
        return None

    # --- Priority 1: custom per-material properties from the Anacapa panel ---
    custom_weight = getattr(mat, 'anacapa_subsurface', 0.0)
    if custom_weight > 0.0:
        sc_raw = getattr(mat, 'anacapa_subsurface_color', None)
        sc = list(sc_raw[:3]) if sc_raw is not None else None
        # If SSS color is the default near-white (unset by user), fall back to
        # the material's actual base_color so sss_weight=1 doesn't override
        # diffuse color with white.
        if sc is None or all(c > 0.74 for c in sc):
            sc = _get_mat_base_color(mat)
        radius_val = float(getattr(mat, 'anacapa_subsurface_radius', 0.1))
        scale      = float(getattr(mat, 'anacapa_subsurface_scale',  1.0))
        aniso      = 0.0
        return (custom_weight, sc, [radius_val, radius_val, radius_val], scale, aniso)

    # --- Priority 2: Principled BSDF node parameters ---
    if not mat.use_nodes or mat.node_tree is None:
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
                        shutter_close: float = 0.0,
                        depsgraph_override=None) -> None:
    """
    Evaluate the scene via the depsgraph and write the binary scene blob to
    filepath.  The live Blender scene is never modified except for temporary
    frame seeks when motion blur is enabled (always restored before return).

    shutter_open / shutter_close are offsets from the current frame in frames,
    matching the --shutter-open / --shutter-close CLI arguments passed to anacapa.
    When shutter_close > shutter_open, two transform keys are written per object
    at the corresponding USD time codes so collectMotionKeys() in USDLoader
    produces motion-blur data.

    depsgraph_override: when provided (e.g. from RenderEngine.render()), use this
    depsgraph for the main frame instead of context.evaluated_depsgraph_get().
    This lets the render engine pass the render-mode depsgraph so GN instance
    counts reflect the full render density rather than the viewport guide count.
    Motion-blur sub-frame depsgraphs still come from context.
    """
    depsgraph = depsgraph_override if depsgraph_override is not None \
                else context.evaluated_depsgraph_get()
    scene     = context.scene
    current_frame = scene.frame_current

    do_motion_blur = shutter_close > shutter_open + 1e-5

    light_data = []
    camera     = None

    # --- GN "Set Material" override detection -----------------------------------
    # When a GN modifier uses a "Set Material" node, the assigned material appears
    # in the realized emitter mesh as a subset (material_index N, N≥1) but is NOT
    # stored on the prototype object's material_slots.  Virtual instances therefore
    # export with the prototype's original material (wrong).
    #
    # Detection: for each GN emitter with virtual instances, find the non-slot-0
    # material index in the realized mesh whose face_count is closest to
    #   instance_count × proto_poly_count
    # If within 50%, treat it as the Set Material override.
    #
    # We also record the filter index so the realized emitter mesh can exclude those
    # faces (avoiding double geometry with the virtual instances).
    _instance_mat_overrides  = {}  # emitter_name → override_mat_name
    _instance_mat_filter_idx = {}  # emitter_name → material_index to strip from emitter mesh
    _gn_emitter_counts  = {}       # emitter_name → virtual instance count
    _gn_emitter_proto   = {}       # emitter_name → prototype bpy.types.Object (original)
    _gn_emitter_eval    = {}       # emitter_name → emitter bpy.types.Object (evaluated)

    for _di in depsgraph.object_instances:
        if _di.is_instance:
            if _di.parent is None:
                continue
            _dp = _di.parent.original if hasattr(_di.parent, 'original') else _di.parent
            if getattr(_dp, 'instance_type', '') == 'COLLECTION':
                continue
            _dn = _dp.name
            _gn_emitter_counts[_dn] = _gn_emitter_counts.get(_dn, 0) + 1
            if _dn not in _gn_emitter_proto:
                _gn_emitter_proto[_dn] = _di.object  # safe bpy.types.Object ref
        else:
            _do = _di.object.original
            if any(m.type == 'NODES' for m in _do.modifiers):
                _gn_emitter_eval[_do.name] = _di.object

    for _dn, _dcount in _gn_emitter_counts.items():
        _dproto = _gn_emitter_proto.get(_dn)
        _demit  = _gn_emitter_eval.get(_dn)
        if _dproto is None or _demit is None:
            continue
        try:
            _dpeval = _dproto.evaluated_get(depsgraph)
            _dpm    = _dpeval.to_mesh()
            _dpp    = len(_dpm.polygons) if _dpm else 0
            _dpeval.to_mesh_clear()
        except Exception:
            _dpp = 0
        if _dpp == 0:
            continue
        try:
            _deeval = _demit.evaluated_get(depsgraph)
            _dem    = _deeval.to_mesh()
            _defc   = {}  # material_index → face count
            if _dem:
                for _dpoly in _dem.polygons:
                    _defc[_dpoly.material_index] = _defc.get(_dpoly.material_index, 0) + 1
            # Use evaluated material_slots: GN "Set Material" adds slots only
            # on the evaluated object; original may have fewer (e.g. 2 slots
            # while GN adds Sprinkles at index 2, causing 0<2<2 → False).
            _de_slots = list(_deeval.material_slots)
            _deeval.to_mesh_clear()
        except Exception:
            continue
        expected = _dcount * _dpp
        best_ratio = 10.0
        best_slot  = -1
        for _dmi, _dfc in _defc.items():
            if _dmi == 0:
                continue  # skip base/emitter material (slot 0)
            _dr = abs(_dfc - expected) / max(expected, 1)
            if _dr < best_ratio:
                best_ratio = _dr
                best_slot  = _dmi
        if best_ratio < 0.5 and 0 < best_slot < len(_de_slots):
            _dmat = _de_slots[best_slot].material
            if _dmat is not None:
                # Always strip the baked GN instances from the emitter mesh to avoid
                # double-rendering them alongside the virtual instances.
                _instance_mat_filter_idx[_dn] = best_slot
                # Always override the virtual-instance material with the one the GN
                # "Set Material" node assigned.  The old proto_has_mat guard was
                # blocking this override when the candy prototype had Bread assigned
                # by the GN; that guard is no longer needed because face-subset
                # ordering is now correct (eval_obj.material_slots).
                _instance_mat_overrides[_dn] = _dmat.name
                _dproto_obj = _gn_emitter_proto.get(_dn)
                _proto_name = (_dproto_obj.material_slots[0].material.name
                               if _dproto_obj and _dproto_obj.material_slots
                               and _dproto_obj.material_slots[0].material else '(none)')
                pass  # GN SetMaterial override applied

    # -------------------------------------------------------------------------
    # Motion blur pre-pass: collect transforms at shutter open/close frames
    # BEFORE the main write loop so they're available during streaming.
    # Only non-instanced and collection-instance objects get per-instance keys;
    # GN instances fall back to the current-frame matrix for both keys.
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
            _frame_set(motion_times[0])
            dg_open = context.evaluated_depsgraph_get()
            xf_open      = _collect_transforms(dg_open)
            xf_inst_open = _collect_instance_transforms(dg_open)

            _frame_set(motion_times[1])
            dg_close = context.evaluated_depsgraph_get()
            xf_close      = _collect_transforms(dg_close)
            xf_inst_close = _collect_instance_transforms(dg_close)
        finally:
            scene.frame_set(current_frame)
    else:
        xf_open = xf_close = xf_inst_open = xf_inst_close = {}
        motion_times = [float(current_frame)]

    # -------------------------------------------------------------------------
    # Single-pass streaming write.
    # All objects are processed exactly once during the depsgraph iteration:
    # mesh geometry is evaluated, written immediately, and discarded — no
    # geometry arrays are held in memory across iterations.  Lights and camera
    # are cheap (no geometry) and collected into small lists written after the
    # mesh loop.  The mesh count is unknown upfront, so we write a placeholder
    # in the header and seek back to patch it after the loop completes.
    # -------------------------------------------------------------------------
    seen_non_instance = set()
    mesh_count = 0
    _inst_total = 0
    # Geometry cache: for GN instances all copies of the same prototype share
    # identical vertex/loop data.  Pack it once to bytes; subsequent instances
    # skip to_mesh() entirely and write the cached blob directly.
    # Key = orig.name (prototype object name).
    _geo_cache = {}  # orig.name → (nv, nl, np_, nuv, geo_bytes)
    # Instance groups: GN instances are NOT written as individual MeshRecords.
    # They are accumulated here and written as compact InstanceGroupRecords
    # (one geometry blob + N transform records per prototype).
    # Key = (orig.name, mat_key) so that instances with different material
    # overrides (GN "Set Material") form separate groups.
    _inst_groups = {}  # (proto_name, mat_key) → group dict

    with open(filepath, 'wb') as f:

        # SceneHeader v5 — all counts are placeholders patched at the end.
        f.write(struct.pack('<5I12x',
                            0x41434E41,   # magic "ANCA"
                            5,            # version
                            0,            # num_meshes       — patched below
                            0,            # num_lights       — patched below
                            0))           # has_camera       — patched below
                            # num_inst_groups at offset 20 (in _pad[0]) also 0

        for inst in depsgraph.object_instances:
            _inst_total += 1
            if _inst_total % 10000 == 0:
                print(f"[anacapa] export: {_inst_total} instances processed, "
                      f"{mesh_count} meshes written...")

            obj  = inst.object
            orig = obj.original

            if orig.hide_render:
                continue

            # ---- MESH --------------------------------------------------------
            if obj.type == 'MESH':
                is_inst = inst.is_instance

                raw_matrix = inst.matrix_world
                if is_inst:
                    raw_matrix = _uniform_gn_scale(raw_matrix, inst)
                mat_world = _to_yup(raw_matrix)

                # Motion matrices (no mesh needed)
                if do_motion_blur:
                    if not is_inst:
                        mo = xf_open.get(orig.name,  mat_world)
                        mc = xf_close.get(orig.name, mat_world)
                    elif inst.parent is not None:
                        key = (inst.parent.name, tuple(inst.persistent_id))
                        mo = xf_inst_open.get(key,  mat_world)
                        mc = xf_inst_close.get(key, mat_world)
                    else:
                        mo = mc = mat_world
                    motion_matrices   = [mo, mc]
                    motion_times_mesh = motion_times
                else:
                    motion_matrices   = [mat_world]
                    motion_times_mesh = [float(current_frame)]

                # Geometry: use cache for GN instances, evaluate otherwise.
                cached = _geo_cache.get(orig.name) if is_inst else None

                if cached is not None:
                    nv, nl, np_, nuv, geo_bytes = cached
                    eval_obj = obj   # obj IS the evaluated prototype for GN instances
                else:
                    eval_obj = obj.evaluated_get(depsgraph)
                    mesh = eval_obj.to_mesh()
                    if mesh is None or len(mesh.polygons) == 0:
                        eval_obj.to_mesh_clear()
                        continue

                    if hasattr(mesh, 'calc_normals_split'):
                        mesh.calc_normals_split()

                    nv  = len(mesh.vertices)
                    nl  = len(mesh.loops)
                    np_ = len(mesh.polygons)
                    nuv = len(mesh.uv_layers)

                    if is_inst:
                        # Fast path for GN instances: foreach_get does bulk C-level
                        # reads instead of per-element Python attribute access.
                        _pos = array.array('f', bytes(nv * 12))
                        mesh.vertices.foreach_get('co', _pos)
                        _nrm = array.array('f', bytes(nl * 12))
                        mesh.loops.foreach_get('normal', _nrm)
                        _ls  = array.array('I', bytes(np_ * 4))
                        _lt  = array.array('I', bytes(np_ * 4))
                        _mi  = array.array('I', bytes(np_ * 4))
                        _vi  = array.array('I', bytes(nl * 4))
                        mesh.polygons.foreach_get('loop_start',     _ls)
                        mesh.polygons.foreach_get('loop_total',     _lt)
                        mesh.polygons.foreach_get('material_index', _mi)
                        mesh.loops.foreach_get('vertex_index',      _vi)

                        _geo = bytearray()
                        _geo += _pos.tobytes()
                        _geo += _nrm.tobytes()
                        for uv_layer in mesh.uv_layers:
                            _uv = array.array('f', bytes(nl * 8))
                            uv_layer.data.foreach_get('uv', _uv)
                            _geo += _pack_str(uv_layer.name)
                            _geo += _uv.tobytes()
                        _geo += _ls.tobytes()
                        _geo += _lt.tobytes()
                        _geo += _mi.tobytes()
                        _geo += _vi.tobytes()
                        geo_bytes = bytes(_geo)
                        _geo_cache[orig.name] = (nv, nl, np_, nuv, geo_bytes)

                    else:
                        # Non-instance path: list comprehensions + optional face filter.
                        positions    = [c for v in mesh.vertices
                                        for c in (v.co.x, v.co.y, v.co.z)]
                        if hasattr(mesh, 'corner_normals'):
                            normals  = [c for cn in mesh.corner_normals
                                        for c in (cn.vector.x, cn.vector.y, cn.vector.z)]
                        else:
                            normals  = [c for loop in mesh.loops
                                        for c in (loop.normal.x, loop.normal.y, loop.normal.z)]
                        uvlayers     = [(uv_layer.name,
                                         [c for uv in uv_layer.data
                                          for c in (uv.uv.x, uv.uv.y)])
                                        for uv_layer in mesh.uv_layers]
                        loop_starts  = [p.loop_start     for p in mesh.polygons]
                        loop_totals  = [p.loop_total     for p in mesh.polygons]
                        mat_indices  = [p.material_index for p in mesh.polygons]
                        vert_indices = [l.vertex_index   for l in mesh.loops]

                        # GN face filter: strip baked instance faces from emitter.
                        _filt = _instance_mat_filter_idx.get(orig.name)
                        if _filt is not None:
                            _new_ls, _new_lt, _new_mi, _new_vi = [], [], [], []
                            _new_nr = []
                            _new_uv_lists = [[] for _ in mesh.uv_layers]
                            _corner_off = 0
                            _cn_list = list(mesh.corner_normals) if hasattr(mesh, 'corner_normals') else None
                            for _fp in mesh.polygons:
                                if _fp.material_index == _filt:
                                    continue
                                _new_ls.append(_corner_off)
                                _new_lt.append(_fp.loop_total)
                                _new_mi.append(_fp.material_index)
                                for _fl in range(_fp.loop_start, _fp.loop_start + _fp.loop_total):
                                    _new_vi.append(mesh.loops[_fl].vertex_index)
                                    if _cn_list:
                                        _cv = _cn_list[_fl].vector
                                        _new_nr += [_cv.x, _cv.y, _cv.z]
                                    else:
                                        _fn = mesh.loops[_fl].normal
                                        _new_nr += [_fn.x, _fn.y, _fn.z]
                                    for _ui, _ul in enumerate(mesh.uv_layers):
                                        _fuv = _ul.data[_fl].uv
                                        _new_uv_lists[_ui] += [_fuv.x, _fuv.y]
                                _corner_off += _fp.loop_total
                            loop_starts  = _new_ls
                            loop_totals  = _new_lt
                            mat_indices  = _new_mi
                            vert_indices = _new_vi
                            normals      = _new_nr
                            uvlayers     = [(mesh.uv_layers[_ui].name, _new_uv_lists[_ui])
                                            for _ui in range(len(mesh.uv_layers))]
                            nv  = len(positions) // 3
                            nl  = len(normals)   // 3
                            np_ = len(loop_starts)
                            nuv = len(uvlayers)

                        _geo = bytearray()
                        _geo += struct.pack(f'<{len(positions)}f', *positions)
                        _geo += struct.pack(f'<{len(normals)}f',   *normals)
                        for uv_name, uvs in uvlayers:
                            _geo += _pack_str(uv_name)
                            _geo += struct.pack(f'<{len(uvs)}f', *uvs)
                        _geo += struct.pack(f'<{np_}I', *loop_starts)
                        _geo += struct.pack(f'<{np_}I', *loop_totals)
                        _geo += struct.pack(f'<{np_}I', *mat_indices)
                        _geo += struct.pack(f'<{nl}I',  *vert_indices)
                        geo_bytes = bytes(_geo)

                    eval_obj.to_mesh_clear()

                # Material info from evaluated material slots.
                mat_names    = []
                mat_flags    = []
                mat_colors   = []
                mat_sss_data = []
                mat_slots = list(eval_obj.material_slots)
                has_real_mats = any(s.material is not None for s in mat_slots)
                if not has_real_mats and is_inst and inst.parent is not None:
                    parent_orig = inst.parent.original if hasattr(inst.parent, 'original') else inst.parent
                    parent_slots = [s for s in parent_orig.material_slots if s.material is not None]
                    if parent_slots:
                        mat_slots = parent_slots
                for slot in mat_slots:
                    mat = slot.material
                    mat_names.append(mat.name if mat else "")
                    is_trans, color = _is_pure_translucent(mat)
                    if not is_trans:
                        color = _get_mat_base_color(mat)
                    sss = _get_sss_params(mat)
                    flags = (1 if is_trans else 0) | (2 if sss is not None else 0)
                    mat_flags.append(flags)
                    mat_colors.extend(color)
                    if sss is not None:
                        weight, sc, radius, scale, aniso = sss
                        mat_sss_data.extend([weight] + sc + radius + [scale, aniso])
                    else:
                        mat_sss_data.extend([0.0, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 1.0, 0.0])

                if is_inst:
                    # GN instance → accumulate into instance group (no per-instance
                    # file write; prototype geometry is written once per group).
                    mat_key = tuple(mat_names)
                    grp_key = (orig.name, mat_key)
                    if grp_key not in _inst_groups:
                        _mt  = raw_matrix.translation
                        _owt = orig.matrix_world.translation
                        print(f"[anacapa] proto '{orig.name}' nv={nv}"
                              f"  v[0]=({_pos[0]:.3f},{_pos[1]:.3f},{_pos[2]:.3f})"
                              f"  inst_t=({_mt.x:.3f},{_mt.y:.3f},{_mt.z:.3f})"
                              f"  orig_mw_t=({_owt.x:.3f},{_owt.y:.3f},{_owt.z:.3f})")
                        import sys; sys.stdout.flush()
                        _inst_groups[grp_key] = {
                            'proto_name':  orig.name,
                            'nv': nv, 'nl': nl, 'np_': np_, 'nuv': nuv,
                            'mat_names':   mat_names,
                            'mat_flags':   mat_flags,
                            'mat_colors':  mat_colors,
                            'mat_sss_data': mat_sss_data,
                            'geo_bytes':   geo_bytes,
                            'nmk':         len(motion_matrices),
                            'motion_times': motion_times_mesh,
                            'instances':   [],
                        }
                    _mdata = bytearray()
                    for _m in motion_matrices:
                        _mdata += _pack_matrix(_m)
                    _inst_groups[grp_key]['instances'].append(
                        (obj.name, bytes(_mdata)))
                else:
                    # Non-instanced object → write MeshRecord immediately.
                    nm  = len(mat_names)
                    nmk = len(motion_matrices)
                    f.write(struct.pack('<6I', nv, nl, np_, nuv, nm, nmk))
                    for _mat in motion_matrices:
                        f.write(_pack_matrix(_mat))
                    f.write(_pack_floats(motion_times_mesh))
                    f.write(_pack_str(obj.name))
                    for mname in mat_names:
                        f.write(_pack_str(mname))
                    f.write(_pack_uints(mat_flags))
                    f.write(_pack_floats(mat_colors))
                    f.write(_pack_floats(mat_sss_data))
                    f.write(geo_bytes)
                    mesh_count += 1

            # ---- LIGHT -------------------------------------------------------
            elif obj.type == 'LIGHT':
                if inst.is_instance:
                    continue
                if orig.name in seen_non_instance:
                    continue
                seen_non_instance.add(orig.name)

                ld = orig.data
                mat_world = _to_yup(obj.matrix_world)
                type_map = {'SUN': 0, 'AREA': 1, 'POINT': 2, 'SPOT': 2}
                ltype  = type_map.get(ld.type, 0)
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

            # ---- CAMERA ------------------------------------------------------
            elif obj.type == 'CAMERA':
                if inst.is_instance:
                    continue
                if orig.name in seen_non_instance:
                    continue
                seen_non_instance.add(orig.name)

                if camera is not None and orig != scene.camera:
                    continue

                cd = orig.data
                mat_world = _to_yup(obj.matrix_world)
                camera = {
                    'matrix':        mat_world,
                    'lens':          cd.lens,
                    'sensor_width':  cd.sensor_width,
                    'sensor_height': (cd.sensor_width /
                                      (scene.render.resolution_x / scene.render.resolution_y)
                                      if cd.sensor_fit == 'AUTO' else cd.sensor_height),
                    'clip_start':    cd.clip_start,
                    'clip_end':      cd.clip_end,
                    'dof_distance':  cd.dof.focus_distance,
                    'dof_fstop':     cd.dof.aperture_fstop,
                }

        # ---- Instance group records (after mesh loop) -----------------------
        for grp in _inst_groups.values():
            num_inst = len(grp['instances'])
            nv  = grp['nv'];  nl = grp['nl'];  np_ = grp['np_'];  nuv = grp['nuv']
            nm  = len(grp['mat_names'])
            nmk = grp['nmk']
            # InstanceGroupHeader: nv, nl, np_, nuv, nm, nmk, num_instances
            f.write(struct.pack('<7I', nv, nl, np_, nuv, nm, nmk, num_inst))
            f.write(_pack_str(grp['proto_name']))
            for mname in grp['mat_names']:
                f.write(_pack_str(mname))
            f.write(_pack_uints(grp['mat_flags']))
            f.write(_pack_floats(grp['mat_colors']))
            f.write(_pack_floats(grp['mat_sss_data']))
            f.write(_pack_floats(grp['motion_times']))
            f.write(grp['geo_bytes'])   # prototype geometry, written once
            for inst_name, mat_bytes in grp['instances']:
                f.write(_pack_str(inst_name))
                f.write(mat_bytes)

        # ---- Lights and camera (after mesh loop) ----------------------------
        for l in light_data:
            f.write(struct.pack('<I', l['type']))
            f.write(_pack_matrix(l['matrix']))
            f.write(struct.pack('<3f', *l['color']))
            f.write(struct.pack('<f',  l['intensity']))
            f.write(struct.pack('<I',  l['normalize']))
            f.write(struct.pack('<4f', *l['params']))
            f.write(_pack_str(l['name']))

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

        # Patch the header now that all counts are known.
        # Offsets: magic(4) + version(4) → num_meshes at 8, then lights, has_camera,
        # num_inst_groups at 20 (was _pad[0]).
        f.seek(8)
        f.write(struct.pack('<4I',
                            mesh_count,
                            len(light_data),
                            1 if camera else 0,
                            len(_inst_groups)))

    total_instances = sum(len(g['instances']) for g in _inst_groups.values())
    print(f"[anacapa] scene binary v5: {mesh_count} meshes, "
          f"{len(_inst_groups)} instance groups ({total_instances} instances), "
          f"{'motion blur' if do_motion_blur else 'static'}, "
          f"{len(light_data)} lights, camera={'yes' if camera else 'no'} → {filepath}")
