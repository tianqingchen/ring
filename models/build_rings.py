#!/usr/bin/env python3
"""Build three millimetre-accurate ring models and render approval views.

Run:
  blender --background --python build_rings.py -- --ring all --stage mesh
  blender --background --python build_rings.py -- --ring all --stage preview
  blender --background --python build_rings.py -- --ring pink --stage color
  blender --background --python build_rings.py -- --ring all --stage final
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
RENDER_DIR = os.path.join(ROOT, "media", "renders")
PREVIEW_DIR = "/tmp/ringprev"

# Finger
INNER_D = 15.7
R_IN = INNER_D / 2.0

# Pink princess — IGI LG688527512. Percents are of the 7.72 mm width.
PINK_L, PINK_W, PINK_D = 7.80, 7.72, 5.14
PINK_TABLE = 0.64
PINK_CROWN = 0.145 * PINK_W          # 1.1194
PINK_PAV = 0.48 * PINK_W             # 3.7056
PINK_GIRDLE = PINK_D - PINK_CROWN - PINK_PAV  # 0.315
PINK_TUBE = 2.80
PINK_PRONG_D = 1.10
PINK_PRONG_ABOVE = 0.80              # cylinder tip above girdle top
PINK_CULET_GAP = 1.00                # culet above the outside of the full round band

# Blue emerald — IGI LG756680165. Percents are of the 5.86 mm width.
BLUE_L, BLUE_W, BLUE_D = 8.57, 5.86, 3.53
BLUE_TABLE = 0.61
BLUE_CROWN = 0.15 * BLUE_W           # 0.879
BLUE_PAV = 0.405 * BLUE_W            # 2.3733
BLUE_GIRDLE = BLUE_D - BLUE_CROWN - BLUE_PAV  # 0.2777
BLUE_CHAMFER = 0.95
# Cert says the culet is "Long" and does not give a percentage.
BLUE_KEEL_L = 3.40
BLUE_KEEL_W = 0.62

BLUE_A_TUBE = 2.70                   # round shank before it tapers
BLUE_A_CLEAR = 1.05                  # culet above the finger; top half of the stone stays out

BLUE_B_TUBE = 2.15
BLUE_B_GAP = 0.60                    # culet above the crown of the tube
BLUE_B_GRIP_W = 2.35                 # along the short end (Y)
BLUE_B_GRIP_T = 1.25                 # thickness (X)
BLUE_B_GRIP_ABOVE = 0.50
BLUE_B_GRIP_BITE = 0.30

BLENDER = "/tmp/blender-dl/blender-4.5.15-linux-x64/blender"


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


def shade_metal(obj, angle_deg=48):
    """Smooth the tube, keep creases at caps and hard intersections."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-5, edges=bm.edges)
    thr = math.radians(angle_deg)
    sharp = []
    for e in bm.edges:
        if len(e.link_faces) == 2:
            n0 = e.link_faces[0].normal
            n1 = e.link_faces[1].normal
            if n0.length < 1e-8 or n1.length < 1e-8:
                continue
            ang = n0.angle(n1)
            if ang >= thr:
                sharp.append(e)
    bmesh.ops.split_edges(bm, edges=sharp)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.update()


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


def min_axis_radius(obj):
    mw = obj.matrix_world
    rmin = 1e9
    for v in obj.data.vertices:
        p = mw @ v.co
        rmin = min(rmin, math.hypot(p.x, p.z))
    return rmin


# ---------------------------------------------------------------------------
# Gems
# ---------------------------------------------------------------------------

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
    # Reorient outward
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
    # The clipper emits one vertex per corner, so the shell is not a volume
    # until those corners are welded. Volume absorption only tints a closed mesh.
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
    z_culet = 0.0
    planes = []

    def add(point, normal):
        planes.append((np.array(point, dtype=float), np.array(normal, dtype=float)))

    add((0, 0, z_table), (0, 0, 1))
    add((hx, 0, 0), (1, 0, 0))
    add((-hx, 0, 0), (-1, 0, 0))
    add((0, hy, 0), (0, 1, 0))
    add((0, -hy, 0), (0, -1, 0))

    # Pavilion mains: girdle edge to the pointed culet. Profile depth is 48%.
    for sx, sy, nx, ny in (
        (1, 0, 1, 0),
        (-1, 0, -1, 0),
        (0, 1, 0, 1),
        (0, -1, 0, -1),
    ):
        # Plane through culet and the girdle edge. Outward normal tilts up-out.
        if nx != 0:
            # contains Y, through (sx*hx, 0, z_gb) and origin
            add((sx * hx, 0, z_gb), (sx * z_gb, 0, -hx))
        else:
            add((0, sy * hy, z_gb), (0, sy * z_gb, -hy))

    # Crown bezels: table edge to girdle edge. Crown height is 14.5%.
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if sx != 0:
            # outward ~ (positive x, 0, positive z) for +X
            add((sx * tx, 0, z_table), (sx * (z_table - z_gt), 0, (hx - tx)))
        else:
            add((0, sy * ty, z_table), (0, sy * (z_table - z_gt), (hy - ty)))

    # Crown corner facets. They keep the girdle corner and chamfer the table corner
    # so the centre profile (64% table, 14.5% crown) stays put.
    inset = 0.42
    for sx in (1, -1):
        for sy in (1, -1):
            gc = (sx * hx, sy * hy, z_gt)
            ex = (sx * tx, sy * ty * inset, z_table)
            ey = (sx * tx * inset, sy * ty, z_table)
            p1 = np.array(ex) - np.array(gc)
            p2 = np.array(ey) - np.array(gc)
            n = np.cross(p1, p2)
            # outward should point away from origin-ish
            if np.dot(n, np.array([sx, sy, 0.4])) < 0:
                n = -n
            add(gc, n)
    return planes


def build_princess():
    obj = faces_to_mesh("Pink Stone", convex_from_planes(princess_planes()))
    bb = world_bbox(obj)
    print(
        f"PINK STONE bbox L={bb['x'][2]:.3f} W={bb['y'][2]:.3f} D={bb['z'][2]:.3f} "
        f"(target {PINK_L:.2f} x {PINK_W:.2f} x {PINK_D:.2f})"
    )
    return obj


def octagon(hx, hy, c, z):
    c = min(c, hx * 0.95, hy * 0.95)
    return [
        np.array([hx, hy - c, z]),
        np.array([hx - c, hy, z]),
        np.array([-(hx - c), hy, z]),
        np.array([-hx, hy - c, z]),
        np.array([-hx, -(hy - c), z]),
        np.array([-(hx - c), -hy, z]),
        np.array([hx - c, -hy, z]),
        np.array([hx, -(hy - c), z]),
    ]


def lerp_ring(a, b, s, z):
    return [((1 - s) * p[:2] + s * q[:2]).tolist() + [z] for p, q in zip(a, b)]


def stitch(rings):
    """rings: list of equal-length point lists. Returns verts, side faces. Caps separate."""
    n = len(rings[0])
    verts = [p for ring in rings for p in ring]
    faces = []
    for s in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            faces.append((s * n + i, s * n + j, (s + 1) * n + j, (s + 1) * n + i))
    return verts, faces


def build_emerald(name):
    hx, hy = BLUE_L / 2, BLUE_W / 2
    c = BLUE_CHAMFER
    girdle = octagon(hx, hy, c, 0.0)
    table = octagon(hx * BLUE_TABLE, hy * BLUE_TABLE, c * BLUE_TABLE, 0.0)
    keel = octagon(BLUE_KEEL_L / 2, BLUE_KEEL_W / 2, 0.16, 0.0)

    z_gb = BLUE_PAV
    z_gt = BLUE_PAV + BLUE_GIRDLE
    z_table = BLUE_D

    # Crown: three step rows. Lower row steeper, upper row shallower.
    crown_s = [0.0, 0.30, 0.62, 1.0]
    crown_z = [0.0, 0.42, 0.74, 1.0]
    crown_rings = []
    for s, zf in zip(crown_s, crown_z):
        ring = lerp_ring(girdle, table, s, z_gt + BLUE_CROWN * zf)
        crown_rings.append(ring)

    pav_s = [0.0, 0.26, 0.50, 0.76, 1.0]
    pav_z = [0.0, 0.22, 0.48, 0.74, 1.0]
    pav_rings = []
    for s, zf in zip(pav_s, pav_z):
        ring = lerp_ring(girdle, keel, s, z_gb * (1.0 - zf))
        pav_rings.append(ring)

    # Stack from keel up to table so the mesh is ordered, but stitch each section.
    # Pavilion rings are keel-last; reverse so index 0 is girdle for a shared girdle.
    # Build as one stack: keel ... girdle bottom, girdle top ... table.
    pav_up = list(reversed(pav_rings))  # keel at 0, girdle bottom at end
    # Shift girdle-bottom ring onto z_gb (already) and crown starts at z_gt.
    rings = pav_up + crown_rings
    verts, faces = stitch([[np.array(p) for p in ring] for ring in rings])
    n = 8
    # caps
    keel_ids = list(range(n))
    table_ids = list(range(len(verts) - n, len(verts)))
    faces.append(tuple(keel_ids))
    faces.append(tuple(reversed(table_ids)))
    obj = new_mesh(name, verts, faces)
    shade_flat(obj)
    bb = world_bbox(obj)
    print(
        f"{name} bbox L={bb['x'][2]:.3f} W={bb['y'][2]:.3f} D={bb['z'][2]:.3f} "
        f"(target {BLUE_L:.2f} x {BLUE_W:.2f} x {BLUE_D:.2f})"
    )
    return obj


# ---------------------------------------------------------------------------
# Metal geometry
# ---------------------------------------------------------------------------

def circle_pts(center, u, v, radius, n):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append(center + radius * (math.cos(a) * u + math.sin(a) * v))
    return pts


def partial_torus(phi0, phi1, n_arc, n_rad, major, minor):
    phis = np.linspace(phi0, phi1, n_arc)
    loops = []
    for phi in phis:
        radial = np.array([math.sin(phi), 0.0, math.cos(phi)])
        center = major * radial
        loops.append(circle_pts(center, radial, np.array([0.0, 1.0, 0.0]), minor, n_rad))
    verts = [p for loop in loops for p in loop]
    faces = []
    for s in range(n_arc - 1):
        for i in range(n_rad):
            j = (i + 1) % n_rad
            a = s * n_rad + i
            b = s * n_rad + j
            c = (s + 1) * n_rad + j
            d = (s + 1) * n_rad + i
            faces.append((a, b, c, d))
    # End caps
    for s, sign in ((0, 1), (n_arc - 1, -1)):
        ids = list(range(s * n_rad, (s + 1) * n_rad))
        center = np.mean([verts[i] for i in ids], axis=0)
        ci = len(verts)
        verts.append(center)
        loop = ids if sign > 0 else list(reversed(ids))
        # sign chooses winding; recalc_face_normals fixes it if the mesh is closed
        for i in range(n_rad):
            faces.append((ci, loop[i], loop[(i + 1) % n_rad]))
    return new_mesh("PartialTorus", verts, faces)


def loft_tube(name, centers, radii, n_rad=14):
    """Closed swept tube through an explicit polyline."""
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
    return solid_loft(name, loops)


def bool_op(host, tool, operation):
    mod = host.modifiers.new(name="Bool", type="BOOLEAN")
    mod.operation = operation
    mod.solver = "MANIFOLD"
    mod.object = tool
    with bpy.context.temp_override(object=host, active_object=host):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(tool, do_unlink=True)
    return host


def full_torus(name, tube_d):
    major = R_IN + tube_d / 2
    minor = tube_d / 2
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major,
        minor_radius=minor,
        major_segments=128,
        minor_segments=36,
        location=(0, 0, 0),
    )
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.rotation_euler = (math.radians(90), 0, 0)
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return obj


def solid_loft(name, rings):
    """Closed loft. rings are loops of equal count, first and last capped."""
    n = len(rings[0])
    verts = []
    for ring in rings:
        for p in ring:
            verts.append(np.array(p, dtype=float))
    faces = []
    for s in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            faces.append((s * n + i, s * n + j, (s + 1) * n + j, (s + 1) * n + i))
    faces.append(tuple(range(n)))
    last = len(rings) - 1
    faces.append(tuple(reversed(range(last * n, (last + 1) * n))))
    return new_mesh(name, verts, faces)


def _torus_dist(x, y, z, major):
    radial = math.hypot(x, z)
    if radial < 1e-8:
        return math.hypot(y, major)
    cx = major * x / radial
    cz = major * z / radial
    return math.sqrt((x - cx) ** 2 + y * y + (z - cz) ** 2)


def _pink_prong_centers(sx, sy, z_axis, z_edge, z_tip, gx, gy, y_exit):
    """One wire from the shared point on the band, out to the corner, then up the edge.

    X never changes, so the side view is a straight post of constant thickness.
    """
    x = sx * gx
    y_tip = sy * gy
    y_leave = sy * y_exit
    centers = []
    # Buried in the tube, crossing the centreline so both corners share one root.
    y0 = -sy * 0.30
    for i in range(7):
        t = i / 6.0
        centers.append(np.array([x, (1.0 - t) * y0 + t * y_leave, z_axis]))
    # Leave the band and arrive vertical at the stone edge.
    n_rise = 16
    dy = abs(y_tip) - y_exit
    dz = z_edge - z_axis
    for i in range(1, n_rise + 1):
        a = (i / n_rise) * (math.pi / 2.0)
        y = sy * (y_exit + dy * math.sin(a))
        z = z_axis + dz * (1.0 - math.cos(a))
        centers.append(np.array([x, y, z]))
    n_up = 8
    for i in range(1, n_up + 1):
        t = i / n_up
        centers.append(np.array([x, y_tip, (1.0 - t) * z_edge + t * z_tip]))
    return centers


def build_pink_band(z_edge, z_tip):
    """Even round shank, closed under the stone. Prongs grow from one point per side."""
    pr = PINK_PRONG_D / 2.0
    major = R_IN + PINK_TUBE / 2.0
    minor = PINK_TUBE / 2.0
    r_out = major + minor
    host = full_torus("Pink Band", PINK_TUBE)
    # Stay flush with the stone's side so the foot is hidden from above.
    # Sit on the corner along the finger so each tip reads as a circle.
    gx = PINK_L / 2.0 - pr
    gy = PINK_W / 2.0 - pr * 0.08
    z_sil = math.sqrt(max(r_out * r_out - gx * gx, 1e-6))
    z_axis = z_sil - pr
    y_exit = 0.35
    while y_exit < gy - 0.6 and _torus_dist(gx, y_exit, z_axis, major) < minor - 0.02:
        y_exit += 0.05
    print(
        f"pink foot x={gx:.3f} y={gy:.3f} z={z_axis:.3f} "
        f"exit_y={y_exit:.3f} edge_z={z_edge:.3f} tip_z={z_tip:.3f}"
    )
    for sx in (1, -1):
        for sy in (1, -1):
            centers = _pink_prong_centers(sx, sy, z_axis, z_edge, z_tip, gx, gy, y_exit)
            tool = loft_tube(f"Prong_{sx}_{sy}", centers, [pr] * len(centers), n_rad=14)
            bool_op(host, tool, "UNION")
    shade_metal(host, 62)
    claw = 0
    wide = 0.0
    for v in host.data.vertices:
        if abs(v.co.x) > 5.0 and abs(v.co.y) > 2.0:
            claw += 1
        if abs(v.co.x) > 6.0:
            wide = max(wide, abs(v.co.y))
    print(f"pink claw vertices {claw} shank half-width {wide:.3f}")
    return host


def _superellipse(hx, hy, z, n=40, power=2.0):
    pts = []
    e = 2.0 / power
    for i in range(n):
        a = 2.0 * math.pi * i / n
        c, s = math.cos(a), math.sin(a)
        pts.append(
            np.array(
                [
                    hx * math.copysign(abs(c) ** e, c),
                    hy * math.copysign(abs(s) ** e, s),
                    z,
                ]
            )
        )
    return pts


def _apply_modifier(obj, name):
    bpy.context.view_layer.objects.active = obj
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.modifier_apply(modifier=name)


def build_blue_integrated(stone_z_culet):
    """One tapered band. The head widens until the emerald sits inside it.

    The socket is cut on the stone's midline, so the top half of the stone
    is the only part that leaves the metal.
    """
    z_culet = stone_z_culet
    z_table = z_culet + BLUE_D
    rim_z = z_culet + BLUE_D * 0.5

    # Full round shank. The head is unioned over the top so there is no cut end to leave a slit.
    host = full_torus("Blue Integrated Band", BLUE_A_TUBE)

    slices = [
        (6.4, 8.4, 2.05, 2.0),
        (7.7, 7.3, 2.75, 2.1),
        (8.8, 6.55, 3.45, 2.25),
        (9.7, 6.15, 4.00, 2.45),
        (rim_z, 5.75, 4.40, 2.7),
    ]
    rings = [_superellipse(hx, hy, z, power=p) for z, hx, hy, p in slices]
    head = solid_loft("Head", rings)
    bool_op(host, head, "UNION")

    remesh = host.modifiers.new("Remesh", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = 0.18
    remesh.use_smooth_shade = True
    _apply_modifier(host, remesh.name)
    smooth = host.modifiers.new("Smooth", "SMOOTH")
    smooth.factor = 0.55
    smooth.iterations = 12
    _apply_modifier(host, smooth.name)

    # Remesh softens the hole. Cut it back to a true 15.7 mm circle.
    bpy.ops.mesh.primitive_cylinder_add(
        radius=R_IN + 0.05,
        depth=36,
        location=(0, 0, 0),
        rotation=(math.radians(90), 0, 0),
    )
    finger = bpy.context.view_layer.objects.active
    bool_op(host, finger, "DIFFERENCE")

    cutter = build_emerald("Seat Cut")
    place_stone(cutter, z_culet)
    bool_op(host, cutter, "DIFFERENCE")

    for poly in host.data.polygons:
        poly.use_smooth = True
    host.data.update()
    print(f"INTEGRATED rim_z={rim_z:.3f} table_z={z_table:.3f} exposed={z_table - rim_z:.3f}")
    return host, rim_z, z_table


def build_blue_grips(z_culet):
    host = full_torus("Blue Grips Band", BLUE_B_TUBE)
    z_gb = z_culet + BLUE_PAV
    z_gt = z_gb + BLUE_GIRDLE
    top = z_gt + BLUE_B_GRIP_ABOVE
    tube_top = R_IN + BLUE_B_TUBE
    bottom = tube_top - 1.35
    height = top - bottom
    z_mid = (top + bottom) / 2
    hx = BLUE_L / 2
    # Inner face bites the short end.
    center_x_mag = hx + BLUE_B_GRIP_T / 2 - BLUE_B_GRIP_BITE
    for sx in (1, -1):
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(sx * center_x_mag, 0.0, z_mid)
        )
        grip = bpy.context.view_layer.objects.active
        grip.name = f"Grip_{sx}"
        grip.scale = (BLUE_B_GRIP_T, BLUE_B_GRIP_W, height)
        with bpy.context.temp_override(object=grip, active_object=grip):
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bev = grip.modifiers.new("Bevel", "BEVEL")
            bev.width = 0.12
            bev.segments = 3
            bev.affect = "EDGES"
            bpy.ops.object.modifier_apply(modifier="Bevel")
        bool_op(host, grip, "UNION")
    shade_metal(host, 48)
    return host


# ---------------------------------------------------------------------------
# Materials, lights, cameras
# ---------------------------------------------------------------------------

def metal_material():
    mat = bpy.data.materials.new("Platinum Satin")
    mat.use_nodes = True
    mat.diffuse_color = (0.75, 0.74, 0.70, 1)
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.78, 0.77, 0.74, 1)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.42
    if "Anisotropic" in bsdf.inputs:
        bsdf.inputs["Anisotropic"].default_value = 0.05
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.2
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.55
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.04
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def gem_material(name, absorb_rgba, density, viewport, transmit=(1, 1, 1, 1)):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = viewport
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    # Base color tints refraction. Specular stays white, so the stone still flashes.
    bsdf.inputs["Base Color"].default_value = transmit
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.015
    bsdf.inputs["IOR"].default_value = 2.417
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 1.0
    if "Dispersion" in bsdf.inputs:
        bsdf.inputs["Dispersion"].default_value = 0.018
    out = nt.nodes.get("Material Output")
    absorb = nt.nodes.new("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = absorb_rgba
    absorb.inputs["Density"].default_value = density
    nt.links.new(absorb.outputs["Volume"], out.inputs["Volume"])
    return mat


def assign(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def add_world(scene):
    world = bpy.data.worlds.new("Studio")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.84, 0.84, 0.85, 1)
    bg.inputs["Strength"].default_value = 0.55


def add_lights(scene, target):
    specs = [
        ("Key", (22, -34, 36), 48, 40, (1.0, 0.98, 0.96)),
        ("Fill", (-32, -18, 16), 60, 14, (0.92, 0.95, 1.0)),
        ("Rim", (6, 36, 22), 28, 22, (1.0, 0.98, 0.96)),
        ("Warm", (28, 12, 8), 14, 12, (1.0, 0.48, 0.22)),
        # Feeds the pavilion so the body stays bright instead of black windows.
        ("Lift", (4, -26, 5), 42, 34, (1.0, 0.80, 0.66)),
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


def add_ground(scene, z):
    bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, z))
    obj = bpy.context.view_layer.objects.active
    obj.name = "Ground"
    obj.is_shadow_catcher = True
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    mat.diffuse_color = (0.84, 0.84, 0.85, 1)
    assign(obj, mat)
    return obj


def fit_camera(scene, name, view_dir, target, objs, lens=120, margin=1.18):
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
    half_w = 18.0  # sensor 36 / 2
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
    # Columns are camera X (right), Y (up), Z (back, toward the camera from the target).
    cam.matrix_world = bpy.types.Matrix if False else __import__("mathutils").Matrix(
        (
            (right[0], up[0], view[0], loc[0]),
            (right[1], up[1], view[1], loc[1]),
            (right[2], up[2], view[2], loc[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return cam


def setup_render(scene, stage):
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    # Punchy keeps the fancy-color body from going gray. Metal stays satin.
    scene.view_settings.look = "AgX - Punchy"
    scene.view_settings.exposure = 0.2
    if stage == "preview":
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x = 1100
        scene.render.resolution_y = 1100
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.studio_light = "studio.sl"
        scene.render.resolution_percentage = 100
        return
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 16
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 8
    scene.cycles.transmission_bounces = 12
    scene.cycles.volume_bounces = 2
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.blur_glossy = 0.35
    scene.cycles.sample_clamp_indirect = 20
    scene.render.threads = 4
    if stage == "color":
        scene.render.resolution_x = 700
        scene.render.resolution_y = 700
        scene.cycles.samples = 48
    else:
        scene.render.resolution_x = 1800
        scene.render.resolution_y = 1800
        scene.cycles.samples = 96
    scene.render.resolution_percentage = 100


def render_cameras(scene, cams, out_paths):
    for cam, path in zip(cams, out_paths):
        scene.camera = cam
        scene.render.filepath = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("RENDER", path)
        bpy.ops.render.render(write_still=True)
        print("WROTE", path, os.path.getsize(path))


# ---------------------------------------------------------------------------
# Rings
# ---------------------------------------------------------------------------

def place_stone(obj, z_culet):
    # Local gem has its lowest point at z=0 (culet / keel).
    bb = world_bbox(obj)
    obj.location.z += z_culet - bb["z"][0]
    bpy.context.view_layer.update()


def build_ring(kind):
    scene = bpy.context.scene
    metal = metal_material()
    if kind == "pink":
        stone = build_princess()
        z_crown = R_IN + PINK_TUBE
        z_culet = z_crown + PINK_CULET_GAP
        place_stone(stone, z_culet)
        bb = world_bbox(stone)
        z_gb = bb["z"][0] + PINK_PAV
        z_gt = z_gb + PINK_GIRDLE
        z_tip = z_gt + PINK_PRONG_ABOVE
        print(f"pink crown {z_crown:.3f} culet {z_culet:.3f} gap {z_culet - z_crown:.3f}")
        band = build_pink_band(z_gb, z_tip)
        assign(stone, gem_material(
            "Pink Diamond",
            # In this unit scale the color is what survives, and density is per millimetre.
            # Warm reddish pink: red passes, green and blue are eaten, pavilion goes orange.
            (1.0, 0.40, 0.14, 1),
            0.16,
            (0.90, 0.28, 0.16, 1),
            transmit=(1.0, 0.62, 0.40, 1),
        ))
        assign(band, metal)
        target = (0, 0, (bb["z"][0] + bb["z"][1]) * 0.5)
        bottom = world_bbox(band)["z"][0]
        slug = "pink-prongs"
        blend = "pink-prongs.blend"
    elif kind == "integrated":
        stone = build_emerald("Blue Stone")
        z_culet = R_IN + BLUE_A_CLEAR
        place_stone(stone, z_culet)
        band, rim_z, z_table = build_blue_integrated(z_culet)
        assign(stone, gem_material(
            "Blue Diamond",
            # Surviving color is teal. Density is per millimetre at this unit scale.
            (0.10, 0.72, 0.78, 1),
            0.55,
            (0.20, 0.55, 0.62, 1),
            transmit=(0.42, 0.82, 0.88, 1),
        ))
        assign(band, metal)
        bb = world_bbox(stone)
        target = (0, 0, (bb["z"][0] + bb["z"][1]) * 0.5)
        bottom = world_bbox(band)["z"][0]
        slug = "blue-integrated"
        blend = "blue-integrated.blend"
        print(f"INTEGRATED rim_z={rim_z:.3f} table_z={z_table:.3f} proud={z_table - rim_z:.3f}")
    elif kind == "grips":
        stone = build_emerald("Blue Stone")
        tube_top = R_IN + BLUE_B_TUBE
        z_culet = tube_top + BLUE_B_GAP
        place_stone(stone, z_culet)
        band = build_blue_grips(z_culet)
        assign(stone, gem_material(
            "Blue Diamond",
            (0.10, 0.72, 0.78, 1),
            0.55,
            (0.20, 0.55, 0.62, 1),
            transmit=(0.42, 0.82, 0.88, 1),
        ))
        assign(band, metal)
        bb = world_bbox(stone)
        target = (0, 0, (bb["z"][0] + bb["z"][1]) * 0.5)
        bottom = world_bbox(band)["z"][0]
        slug = "blue-grips"
        blend = "blue-grips.blend"
    else:
        raise ValueError(kind)

    print(f"{kind} band inner radius {min_axis_radius(band):.3f} (target {R_IN:.3f})")
    print(f"{kind} band bbox", world_bbox(band))
    print(f"{kind} stone bbox", world_bbox(stone))
    add_world(scene)
    add_lights(scene, target)
    add_ground(scene, bottom - 0.05)
    return scene, slug, blend, target, [band, stone]


def main():
    args = argv_after_dd()
    ring = "all"
    stage = "mesh"
    for i, a in enumerate(args):
        if a == "--ring" and i + 1 < len(args):
            ring = args[i + 1]
        if a == "--stage" and i + 1 < len(args):
            stage = args[i + 1]
    rings = ["pink", "integrated", "grips"] if ring == "all" else [ring]
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RENDER_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    for kind in rings:
        scene = reset_scene()
        setup_render(scene, stage if stage != "mesh" else "preview")
        scene, slug, blend, target, objs = build_ring(kind)
        mins = np.array([1e9, 1e9, 1e9])
        maxs = -mins.copy()
        for o in objs:
            for corner in o.bound_box:
                w = np.array(o.matrix_world @ Vector(corner), dtype=float)
                mins = np.minimum(mins, w)
                maxs = np.maximum(maxs, w)
        framed = (mins + maxs) / 2.0
        views = [
            ("three-quarter", (1.15, -1.35, 0.78), framed, 120, 1.10),
            ("top", (0.0, -0.02, 1.0), framed, 150, 1.12),
            ("side", (0.0, -1.0, 0.0), framed, 140, 1.08),
        ]
        cams = []
        paths = []
        for label, direction, tgt, lens, margin in views:
            cam = fit_camera(scene, f"Cam {label}", direction, tgt, objs, lens=lens, margin=margin)
            cams.append(cam)
            if stage == "final":
                paths.append(os.path.join(RENDER_DIR, f"{slug}-{label}.png"))
            elif stage == "color" and label != "three-quarter":
                continue
            else:
                paths.append(os.path.join(PREVIEW_DIR, f"{slug}-{label}.png"))
        if stage == "color":
            cams = cams[:1]
            paths = paths[:1]
        blend_path = os.path.join(MODEL_DIR, blend)
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print("SAVED", blend_path)
        if stage in ("preview", "color", "final"):
            # Ground is a shadow catcher; workbench would draw it as a huge plane.
            ground = bpy.data.objects.get("Ground")
            if stage == "preview" and ground is not None:
                ground.hide_render = True
            render_cameras(scene, cams, paths)


if __name__ == "__main__":
    main()
