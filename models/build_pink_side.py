#!/usr/bin/env python3
"""Pink princess, version A after the confirmed markup.

Satin tube. Each prong rises straight from one point on the band and runs
along the stone edge to a flat tip. No swept shoulders, no ball tips.
Real scale. Stone 7.80 x 7.72 x 5.14 mm. Inside diameter 15.7 mm.
Does not write any other blend.

  blender --background --python build_pink_side.py -- --stage preview
  blender --background --python build_pink_side.py -- --stage final
"""

import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

ROOT = "/cursor/stores/bc-01a16e52-b478-7f93-bb40-e20170555721"
MODEL_DIR = os.path.join(ROOT, "internal", "models")
GEOM_DIR = os.path.join(ROOT, "media", "geometry")
PREVIEW_DIR = "/tmp/pink-side-preview"

# Finger
INNER_D = 15.7
R_IN = INNER_D / 2.0

# Round shank. Equator of the reference, scaled so the hole is 15.7 mm.
TUBE_D = 2.85
TUBE_R = TUBE_D / 2.0
# Air between the top of the round tube and the culet. Must not close.
CULET_GAP = 0.20

# Straight prong wire. The flat tip is the face that replaced the ball.
PRONG_D = 0.92
PRONG_R = PRONG_D / 2.0
TIP_D = 1.40
TIP_R = TIP_D / 2.0
TIP_H = 0.32
TIP_ABOVE = 0.10
# Shaft outer face is flush with the girdle, so the arm does not show past
# the stone from above. The wider flat tip is what sits on the corner.

# Pink princess — IGI LG688527512. Percents are of the 7.72 mm width.
PINK_L, PINK_W, PINK_D = 7.80, 7.72, 5.14
PINK_TABLE = 0.64
PINK_CROWN = 0.145 * PINK_W
PINK_PAV = 0.48 * PINK_W
PINK_GIRDLE = PINK_D - PINK_CROWN - PINK_PAV

Z_CULET = R_IN + TUBE_D + CULET_GAP
Z_GIRDLE_BOT = Z_CULET + PINK_PAV
Z_GIRDLE_TOP = Z_GIRDLE_BOT + PINK_GIRDLE
Z_TABLE = Z_CULET + PINK_D
# Flat face sits just above the table.
Z_TIP = Z_TABLE + TIP_ABOVE


def argv_after_dd():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"
    return scene


def new_mesh(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(map(float, v)) for v in verts], [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def shade_flat(obj):
    for p in obj.data.polygons:
        p.use_smooth = False


def shade_smooth(obj):
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.update()


def world_bbox(obj):
    mw = obj.matrix_world
    pts = [(mw @ v.co) for v in obj.data.vertices]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    return {
        "x": (min(xs), max(xs), max(xs) - min(xs)),
        "y": (min(ys), max(ys), max(ys) - min(ys)),
        "z": (min(zs), max(zs), max(zs) - min(zs)),
    }


def _unique(pts, tol=1e-5):
    out = []
    for p in pts:
        if all(np.linalg.norm(p - q) > tol for q in out):
            out.append(np.array(p, dtype=float))
    return out


def _clip_poly(poly, dist, eps=1e-8):
    out = []
    inters = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        da = dist(a)
        db = dist(b)
        a_in = da <= eps
        b_in = db <= eps
        if a_in and b_in:
            out.append(b)
        elif a_in and not b_in:
            t = da / (da - db)
            p = a + t * (b - a)
            out.append(p)
            inters.append(p)
        elif (not a_in) and b_in:
            t = da / (da - db)
            p = a + t * (b - a)
            out.append(p)
            out.append(b)
            inters.append(p)
    return out, inters


def _sort_cap(points, normal):
    pts = np.array(points, dtype=float)
    c = pts.mean(axis=0)
    n = normal / np.linalg.norm(normal)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    ang = np.arctan2((pts - c) @ v, (pts - c) @ u)
    order = np.argsort(ang)
    return [pts[i] for i in order]


def convex_from_planes(planes, bounds=20.0):
    b = bounds
    faces = [
        [np.array(p, dtype=float) for p in [(b, -b, -b), (b, b, -b), (b, b, b), (b, -b, b)]],
        [np.array(p, dtype=float) for p in [(-b, -b, -b), (-b, -b, b), (-b, b, b), (-b, b, -b)]],
        [np.array(p, dtype=float) for p in [(-b, b, -b), (-b, b, b), (b, b, b), (b, b, -b)]],
        [np.array(p, dtype=float) for p in [(-b, -b, -b), (b, -b, -b), (b, -b, b), (-b, -b, b)]],
        [np.array(p, dtype=float) for p in [(-b, -b, b), (b, -b, b), (b, b, b), (-b, b, b)]],
        [np.array(p, dtype=float) for p in [(-b, -b, -b), (-b, b, -b), (b, b, -b), (b, -b, -b)]],
    ]
    for point, normal in planes:
        nrm = np.array(normal, dtype=float)
        nrm = nrm / np.linalg.norm(nrm)
        pt = np.array(point, dtype=float)
        d0 = float(np.dot(pt, nrm))

        def dist(v, nrm=nrm, d0=d0):
            return float(np.dot(v, nrm) - d0)

        new_faces = []
        cap_pts = []
        for face in faces:
            clipped, inters = _clip_poly(face, dist)
            if len(clipped) >= 3:
                new_faces.append(clipped)
            cap_pts.extend(inters)
        cap_pts = _unique(cap_pts, tol=2e-5)
        if len(cap_pts) >= 3:
            new_faces.append(_sort_cap(cap_pts, nrm))
        faces = new_faces
        if not faces:
            raise RuntimeError("gem clipping removed the solid")
    allpts = np.vstack([p for f in faces for p in f])
    center = allpts.mean(axis=0)
    oriented = []
    for face in faces:
        n = np.cross(face[1] - face[0], face[2] - face[0])
        fc = np.mean(face, axis=0)
        if np.dot(n, fc - center) < 0:
            face = list(reversed(face))
        oriented.append(face)
    return oriented


def faces_to_mesh(name, faces):
    verts = []
    polys = []
    for face in faces:
        ids = []
        for p in face:
            ids.append(len(verts))
            verts.append(p)
        polys.append(tuple(ids))
    obj = new_mesh(name, verts, polys)
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    boundary = sum(1 for e in bm.edges if not e.is_manifold)
    print(f"{name} welded verts={len(bm.verts)} faces={len(bm.faces)} boundary={boundary}")
    bm.to_mesh(me)
    bm.free()
    me.update()
    shade_flat(obj)
    return obj


def princess_planes():
    hx, hy = PINK_L / 2, PINK_W / 2
    tx, ty = hx * PINK_TABLE, hy * PINK_TABLE
    z_gb = PINK_PAV
    z_gt = PINK_PAV + PINK_GIRDLE
    z_table = PINK_D
    planes = []

    def add(point, normal):
        planes.append((np.array(point, dtype=float), np.array(normal, dtype=float)))

    add((0, 0, z_table), (0, 0, 1))
    add((hx, 0, 0), (1, 0, 0))
    add((-hx, 0, 0), (-1, 0, 0))
    add((0, hy, 0), (0, 1, 0))
    add((0, -hy, 0), (0, -1, 0))
    for sx, sy, nx, ny in (
        (1, 0, 1, 0),
        (-1, 0, -1, 0),
        (0, 1, 0, 1),
        (0, -1, 0, -1),
    ):
        if nx != 0:
            add((sx * hx, 0, z_gb), (sx * z_gb, 0, -hx))
        else:
            add((0, sy * hy, z_gb), (0, sy * z_gb, -hy))
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if sx != 0:
            add((sx * tx, 0, z_table), (sx * (z_table - z_gt), 0, (hx - tx)))
        else:
            add((0, sy * ty, z_table), (0, sy * (z_table - z_gt), (hy - ty)))
    inset = 0.42
    for sx in (1, -1):
        for sy in (1, -1):
            gc = (sx * hx, sy * hy, z_gt)
            ex = (sx * tx, sy * ty * inset, z_table)
            ey = (sx * tx * inset, sy * ty, z_table)
            p1 = np.array(ex) - np.array(gc)
            p2 = np.array(ey) - np.array(gc)
            n = np.cross(p1, p2)
            if np.dot(n, np.array([sx, sy, 0.4])) < 0:
                n = -n
            add(gc, n)
    return planes


def circle_pts(center, u, v, radius, n):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append(center + radius * (math.cos(a) * u + math.sin(a) * v))
    return pts


def build_shank():
    """Full round tube. Inner surface is the 15.7 mm circle."""
    bpy.ops.mesh.primitive_torus_add(
        major_radius=R_IN + TUBE_R,
        minor_radius=TUBE_R,
        major_segments=128,
        minor_segments=36,
        location=(0, 0, 0),
    )
    obj = bpy.context.view_layer.objects.active
    obj.name = "Pink Band"
    obj.rotation_euler = (math.radians(90), 0, 0)
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    shade_smooth(obj)
    return obj


def catmull(points, n_per):
    """Open Catmull-Rom through points. points are np arrays."""
    pts = [np.array(p, dtype=float) for p in points]
    if len(pts) < 2:
        return pts
    out = []
    ext = [pts[0]] + pts + [pts[-1]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        steps = n_per if i < len(ext) - 3 else n_per + 1
        for s in range(steps):
            t = s / n_per
            t2 = t * t
            t3 = t2 * t
            out.append(
                0.5
                * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
            )
    return out


def loft_open(name, centers, radii, n_rad=16):
    centers = [np.array(c, dtype=float) for c in centers]
    radii = [float(r) for r in radii]
    loops = []
    prev_u = None
    for i, (c, r) in enumerate(zip(centers, radii)):
        if i == 0:
            tangent = centers[1] - centers[0]
        elif i == len(centers) - 1:
            tangent = centers[-1] - centers[-2]
        else:
            tangent = centers[i + 1] - centers[i - 1]
        ln = float(np.linalg.norm(tangent))
        tangent = tangent / ln if ln > 1e-8 else np.array([0.0, 0.0, 1.0])
        if prev_u is None:
            ref = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(tangent, ref))) > 0.85:
                ref = np.array([1.0, 0.0, 0.0])
            u = np.cross(ref, tangent)
            u = u / np.linalg.norm(u)
        else:
            u = prev_u - tangent * float(np.dot(prev_u, tangent))
            if np.linalg.norm(u) < 1e-6:
                u = prev_u
            u = u / np.linalg.norm(u)
        v = np.cross(tangent, u)
        v = v / np.linalg.norm(v)
        prev_u = u
        loops.append(circle_pts(c, u, v, max(r, 0.08), n_rad))
    n = n_rad
    verts = [p for loop in loops for p in loop]
    faces = []
    for s in range(len(loops) - 1):
        for i in range(n):
            j = (i + 1) % n
            faces.append((s * n + i, s * n + j, (s + 1) * n + j, (s + 1) * n + i))
    for s, sign in ((0, 1), (len(loops) - 1, -1)):
        ids = list(range(s * n, (s + 1) * n))
        center = np.mean([verts[i] for i in ids], axis=0)
        ci = len(verts)
        verts.append(center)
        loop = ids if sign > 0 else list(reversed(ids))
        for i in range(n):
            faces.append((ci, loop[i], loop[(i + 1) % n]))
    return new_mesh(name, verts, faces)


def prong_layout():
    """Post center is just inside the corner. Both prongs on a side share X.

    The side view is one straight post. The plan of the wire stays under the
    flat tip, so the top view is the band, the square, and four faces.
    """
    gx = PINK_L / 2.0 - PRONG_R
    gy = PINK_W / 2.0 - PRONG_R
    r_out = R_IN + TUBE_D
    z_sil = math.sqrt(max(r_out * r_out - gx * gx, 1e-6))
    z_root = z_sil - PRONG_R
    return gx, gy, z_root


def prong_centers(sx, sy, gx, gy, z_root):
    """Buried root, then one straight line up the stone edge to the flat face."""
    x = sx * gx
    y_tip = sy * gy
    # Cross the centreline inside the tube so the pair shares one point.
    root = np.array([x, -sy * 0.40, z_root - 0.20], dtype=float)
    leave = np.array([x, sy * 0.45, z_root], dtype=float)
    corner = np.array([x, y_tip, Z_GIRDLE_BOT - 0.05], dtype=float)
    top = np.array([x, y_tip, Z_TIP], dtype=float)
    centers = [root, leave]
    n_run = 12
    for i in range(1, n_run + 1):
        t = i / n_run
        centers.append((1.0 - t) * leave + t * corner)
    n_up = 8
    for i in range(1, n_up + 1):
        t = i / n_up
        centers.append((1.0 - t) * corner + t * top)
    return centers


def build_prong(sx, sy, gx, gy, z_root):
    centers = prong_centers(sx, sy, gx, gy, z_root)
    obj = loft_open(f"Prong_{sx}_{sy}", centers, [PRONG_R] * len(centers), n_rad=16)
    return obj


def build_tip(sx, sy, gx, gy):
    """Short cylinder. The top is a flat face, not a ball."""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=40,
        radius=TIP_R,
        depth=TIP_H,
        location=(sx * gx, sy * gy, Z_TIP - TIP_H / 2.0),
    )
    obj = bpy.context.view_layer.objects.active
    obj.name = f"Tip_{sx}_{sy}"
    shade_flat(obj)
    return obj


def flatten_tips(obj, gx, gy):
    """Keep the four tip faces planar after the remesh softens them."""
    z_top = Z_TIP
    reach = TIP_R * 0.94
    for v in obj.data.vertices:
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                dx = v.co.x - sx * gx
                dy = v.co.y - sy * gy
                if dx * dx + dy * dy < reach * reach and v.co.z > z_top - 0.22:
                    v.co.z = z_top
    obj.data.update()
    for poly in obj.data.polygons:
        if poly.normal.z < 0.82 or poly.center.z < z_top - 0.12:
            continue
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                dx = poly.center.x - sx * gx
                dy = poly.center.y - sy * gy
                if dx * dx + dy * dy < TIP_R * TIP_R:
                    poly.use_smooth = False
                    break
    obj.data.update()


def bool_op(host, tool, operation):
    mod = host.modifiers.new(name="Bool", type="BOOLEAN")
    mod.operation = operation
    mod.solver = "MANIFOLD"
    mod.object = tool
    bpy.context.view_layer.objects.active = host
    with bpy.context.temp_override(object=host, active_object=host):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(tool, do_unlink=True)
    return host


def apply_modifier(obj, name):
    bpy.context.view_layer.objects.active = obj
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.modifier_apply(modifier=name)


def place_stone(obj, z_culet):
    bb = world_bbox(obj)
    obj.location.z += z_culet - bb["z"][0]
    bpy.context.view_layer.update()


def duplicate_obj(obj, name):
    copy = obj.copy()
    copy.data = obj.data.copy()
    copy.name = name
    bpy.context.scene.collection.objects.link(copy)
    return copy


def build_metal(voxel):
    gx, gy, z_root = prong_layout()
    print(f"PRONG foot x={gx:.3f} y={gy:.3f} z_root={z_root:.3f} tip_z={Z_TIP:.3f}")
    host = build_shank()
    for sx in (1, -1):
        for sy in (1, -1):
            bool_op(host, build_prong(sx, sy, gx, gy, z_root), "UNION")
            bool_op(host, build_tip(sx, sy, gx, gy), "UNION")
    remesh = host.modifiers.new("Remesh", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = voxel
    remesh.use_smooth_shade = True
    apply_modifier(host, remesh.name)
    smooth = host.modifiers.new("Smooth", "SMOOTH")
    smooth.factor = 0.10
    smooth.iterations = 2
    apply_modifier(host, smooth.name)
    # Remesh softens the hole. Cut it back to a true 15.7 mm circle.
    bpy.ops.mesh.primitive_cylinder_add(
        radius=R_IN,
        depth=40,
        location=(0, 0, 0),
        rotation=(math.radians(90), 0, 0),
    )
    finger = bpy.context.view_layer.objects.active
    bool_op(host, finger, "DIFFERENCE")
    shade_smooth(host)
    return host


def clay_material():
    mat = bpy.data.materials.new("Clay")
    mat.use_nodes = True
    mat.diffuse_color = (0.74, 0.74, 0.75, 1)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.74, 0.74, 0.75, 1)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.78
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.25
    return mat


def assign(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def add_world(scene):
    world = bpy.data.worlds.new("ClayWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.11, 0.11, 0.12, 1)
    bg.inputs["Strength"].default_value = 1.0


def add_lights(scene, target):
    specs = [
        ("Key", (16, -26, 22), 28, 55, (1.0, 1.0, 1.0)),
        ("Fill", (-24, -14, 10), 36, 18, (0.95, 0.96, 1.0)),
        ("Rim", (4, 28, 14), 22, 26, (1.0, 1.0, 1.0)),
        ("Top", (0, -6, 32), 24, 14, (1.0, 1.0, 1.0)),
    ]
    target = Vector(target)
    for name, loc, size, power, color in specs:
        light = bpy.data.lights.new(name, "AREA")
        light.shape = "SQUARE"
        light.size = size
        light.energy = power
        light.color = color
        obj = bpy.data.objects.new(name, light)
        obj.location = loc
        scene.collection.objects.link(obj)
        direction = target - obj.location
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def fit_camera(scene, name, view_dir, target, objs, lens=140, margin=1.12):
    view = np.array(view_dir, dtype=float)
    view = view / np.linalg.norm(view)
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(view, up_hint))) > 0.92:
        up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(up_hint, view)
    right = right / np.linalg.norm(right)
    up = np.cross(view, right)
    up = up / np.linalg.norm(up)
    corners = []
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            corners.append(np.array(w, dtype=float))
    target = np.array(target, dtype=float)
    half_w = 18.0
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    half_h = half_w * (res_y / res_x)
    need = 0.05
    for p in corners:
        r = p - target
        sx = abs(float(np.dot(r, right)))
        sy = abs(float(np.dot(r, up)))
        along = float(np.dot(r, view))
        need = max(need, along + sx * lens / half_w, along + sy * lens / half_h)
    dist = need * margin
    loc = target + view * dist
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.sensor_width = 36
    cam_data.clip_start = 0.1
    cam_data.clip_end = 5000
    cam = bpy.data.objects.new(name, cam_data)
    scene.collection.objects.link(cam)
    cam.matrix_world = __import__("mathutils").Matrix(
        (
            (right[0], up[0], view[0], loc[0]),
            (right[1], up[1], view[1], loc[1]),
            (right[2], up[2], view[2], loc[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return cam


def measure(band, stone):
    """Band cross-section, straight prong, flat tip, and the air under the culet."""
    gx, gy, z_root = prong_layout()
    mw = band.matrix_world
    pts = np.array([(mw @ v.co)[:] for v in band.data.vertices], dtype=float)
    # Bottom of the shank, on the centre plane: outer surface vs the hole.
    bottom = pts[(np.abs(pts[:, 0]) < 0.35) & (np.abs(pts[:, 1]) < 0.45) & (pts[:, 2] < -R_IN)]
    z_outer = float(bottom[:, 2].min()) if len(bottom) else float("nan")
    band_d = -z_outer - R_IN

    # Inner radius. Should sit on 7.85 after the cylinder cut.
    r_xz = np.hypot(pts[:, 0], pts[:, 2])
    inner = float(r_xz.min())

    # Vertical shaft just below the girdle, where the stone no longer fills it.
    z_slab = Z_GIRDLE_BOT - 0.55
    slab = pts[np.abs(pts[:, 2] - z_slab) < 0.14]
    prong_ds = []
    if len(slab):
        for sx in (1, -1):
            for sy in (1, -1):
                cl = slab[
                    (np.abs(slab[:, 0] - sx * gx) < 0.7)
                    & (np.abs(slab[:, 1] - sy * gy) < 0.7)
                ]
                if len(cl) < 6:
                    continue
                prong_ds.append(float(cl[:, 0].max() - cl[:, 0].min()))
    prong_d = float(np.median(prong_ds)) if prong_ds else float("nan")

    tip_ds = []
    tip_tops = []
    for sx in (1, -1):
        for sy in (1, -1):
            face = pts[
                (np.abs(pts[:, 0] - sx * gx) < TIP_R)
                & (np.abs(pts[:, 1] - sy * gy) < TIP_R)
                & (np.abs(pts[:, 2] - Z_TIP) < 0.08)
            ]
            if len(face) < 6:
                continue
            span = max(float(face[:, 0].max() - face[:, 0].min()), float(face[:, 1].max() - face[:, 1].min()))
            tip_ds.append(span)
            tip_tops.append(float(face[:, 2].max()))
    tip_d = float(np.median(tip_ds)) if tip_ds else float("nan")

    # How far the lower arms stick past the stone. The flat tips are allowed to.
    stone_bb = world_bbox(stone)
    arm = pts[(pts[:, 2] > z_root + 0.4) & (pts[:, 2] < Z_GIRDLE_BOT - 0.15) & (np.abs(pts[:, 0]) > 1.5)]
    past_x = float(np.max(np.abs(arm[:, 0])) - stone_bb["x"][1]) if len(arm) else float("nan")
    past_y = float(np.max(np.abs(arm[:, 1])) - stone_bb["y"][1]) if len(arm) else float("nan")

    # Culet is the stone's lowest point. Clearance is air down to metal.
    z_culet = stone_bb["z"][0]
    under = pts[(np.abs(pts[:, 0]) < 0.6) & (np.abs(pts[:, 1]) < 0.8) & (pts[:, 2] < z_culet)]
    if len(under):
        z_metal = float(under[:, 2].max())
        gap = z_culet - z_metal
    else:
        z_metal = float("nan")
        gap = float("nan")

    print(
        f"MEASURE band_diameter={band_d:.3f} prong_diameter={prong_d:.3f} "
        f"tip_size={tip_d:.3f} culet_clearance={gap:.3f}"
    )
    print(
        f"MEASURE inner_radius={inner:.3f} target={R_IN:.3f} "
        f"z_culet={z_culet:.3f} z_metal={z_metal:.3f} past_x={past_x:.3f} past_y={past_y:.3f}"
    )
    print(f"MEASURE stone {stone_bb}")
    print(f"MEASURE band {world_bbox(band)}")
    return {
        "band_diameter": band_d,
        "prong_diameter": prong_d,
        "tip_size": tip_d,
        "culet_clearance": gap,
        "inner_radius": inner,
        "past_x": past_x,
        "past_y": past_y,
    }


def setup_render(scene, stage):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 2
    scene.cycles.samples = 24 if stage == "preview" else 64
    scene.render.threads = 0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.35
    px = 1200 if stage == "preview" else 1800
    scene.render.resolution_x = px
    scene.render.resolution_y = px
    scene.render.resolution_percentage = 100


def render_cameras(scene, cams, paths):
    for cam, path in zip(cams, paths):
        scene.camera = cam
        scene.render.filepath = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("RENDER", path)
        bpy.ops.render.render(write_still=True)
        print("WROTE", path, os.path.getsize(path))


def main():
    args = argv_after_dd()
    stage = "preview"
    for i, a in enumerate(args):
        if a == "--stage" and i + 1 < len(args):
            stage = args[i + 1]
    voxel = 0.16 if stage == "preview" else 0.10
    scene = reset_scene()
    setup_render(scene, stage)

    stone = faces_to_mesh("Pink Stone", convex_from_planes(princess_planes()))
    place_stone(stone, Z_CULET)
    bb = world_bbox(stone)
    print(
        f"STONE L={bb['x'][2]:.3f} W={bb['y'][2]:.3f} D={bb['z'][2]:.3f} "
        f"z=({bb['z'][0]:.3f}..{bb['z'][1]:.3f})"
    )
    print(
        f"DESIGN tube={TUBE_D:.2f} prong={PRONG_D:.2f} tip={TIP_D:.2f} "
        f"gap={CULET_GAP:.2f} z_culet={Z_CULET:.3f} z_tip={Z_TIP:.3f}"
    )

    band = build_metal(voxel)
    # Seat the prongs on the stone so the meshes do not occupy the same volume.
    cutter = duplicate_obj(stone, "Seat")
    bool_op(band, cutter, "DIFFERENCE")
    shade_smooth(band)
    gx, gy, _z_root = prong_layout()
    flatten_tips(band, gx, gy)
    shade_flat(stone)

    stats = measure(band, stone)

    clay = clay_material()
    assign(band, clay)
    assign(stone, clay)
    add_world(scene)
    target = (0.0, 0.0, (bb["z"][0] + bb["z"][1]) * 0.45)
    add_lights(scene, target)

    objs = [band, stone]
    mins = np.array([1e9, 1e9, 1e9])
    maxs = -mins.copy()
    for o in objs:
        for corner in o.bound_box:
            w = np.array(o.matrix_world @ Vector(corner), dtype=float)
            mins = np.minimum(mins, w)
            maxs = np.maximum(maxs, w)
    framed = (mins + maxs) / 2.0
    views = [
        ("side", (0.0, -1.0, 0.02), framed, 160, 1.08),
        ("three-quarter", (1.05, -1.25, 0.72), framed, 130, 1.10),
        ("top", (0.0, -0.05, 1.0), framed, 150, 1.10),
    ]
    if stage == "preview":
        out_dir = PREVIEW_DIR
    else:
        out_dir = GEOM_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    cams = []
    paths = []
    for label, direction, tgt, lens, margin in views:
        cams.append(fit_camera(scene, f"Cam {label}", direction, tgt, objs, lens=lens, margin=margin))
        paths.append(os.path.join(out_dir, f"pink-side-scale-{label}.png"))

    blend_path = os.path.join(MODEL_DIR, "pink-side-scale.blend")
    # Only the final pass writes the blend the brief asked for, so a coarse
    # preview cannot replace it. A finished preview still saves if asked.
    if stage == "final":
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print("SAVED", blend_path)
    render_cameras(scene, cams, paths)
    print("STATS", stats)


if __name__ == "__main__":
    main()
