"""
Core pipeline: image path → LegoModel JSON.

Stages:
  1. load_image        – read and validate the file
  2. interpret_image   – placeholder for LLM scene description
  3. decompose         – placeholder for primitive decomposition
  4. build_model       – assemble LegoModel and return as JSON
"""
import base64
import json
import math
from collections import deque
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path

import numpy as np

from schema import (
    BASEPLATE_DEPTH,
    BASEPLATE_WIDTH,
    GRID_SIZE,
    BoundingBox,
    CuboidDimensions,
    CylinderDimensions,
    Face,
    LegoModel,
    Part,
    PartDimensions,
    Position,
    PrimitiveType,
    Shape,
    WedgeDimensions,
)


# ---------------------------------------------------------------------------
# Part-world state  (Z=UP coordinate system)
# ---------------------------------------------------------------------------

@dataclass
class PartState:
    """
    Geometric working state for one part.

    Before apply_rotation:
      center — post-scale, post-placement, pre-rotation world-space center.
      anchor — pre-rotation pivot (original attachment anchor after scale+placement);
               None for roots and parts with no parent_face.

    After apply_rotation:
      center — final world-space position used for voxelization.
      anchor — transformed connection point (R_acc @ pre_anchor + t_acc);
               not the original attachment anchor.

    dims: local canonical extents ALWAYS — never world-space AABB.

    world_R: None before apply_rotation; set to accumulated 3×3 rotation matrix
    R_acc by apply_rotation. Required by voxelize to transform world-space voxel
    candidates back to local space for containment testing.
    """
    uid: str
    dims: PartDimensions
    center: tuple[float, float, float]
    anchor: tuple[float, float, float] | None = None
    world_R: np.ndarray | None = None
    parent_surface_normal: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Graph validation  (Part-world, Z=UP)
# ---------------------------------------------------------------------------

def validate_graph(parts: list[Part]) -> list[Part]:
    """
    Enforce a valid parent-child tree over a list of Parts.

    Two passes:
      1. Fix missing parents — any part whose parent_part uid does not exist
         in the list is re-rooted (parent_part set to None).
      2. Break cycles — DFS detects back-edges; the node that closes a cycle
         is re-rooted, which is the minimal change needed to break it.

    Mutations are applied in-place; the same list is returned.
    Safe for small part counts (< ~100); recursion depth = tree depth.
    """
    by_uid: dict[str, Part] = {p.uid: p for p in parts}

    # Pass 1: fix references to non-existent parents.
    for p in parts:
        if p.parent_part is not None and p.parent_part not in by_uid:
            p.parent_part = None

    # Pass 2: detect and break cycles with a DFS.
    # Node states: 0 = unvisited, 1 = in current path, 2 = fully visited.
    state: dict[str, int] = {uid: 0 for uid in by_uid}

    def _visit(uid: str) -> None:
        if state[uid] != 0:
            return
        state[uid] = 1
        parent_uid = by_uid[uid].parent_part
        if parent_uid:
            if state[parent_uid] == 1:
                # Back-edge found — re-root this node to break the cycle.
                by_uid[uid].parent_part = None
            elif state[parent_uid] == 0:
                _visit(parent_uid)
        state[uid] = 2

    for uid in list(by_uid):
        _visit(uid)

    return parts


# ---------------------------------------------------------------------------
# Critical-closure and single-root enforcement  (Part-world, Z=UP)
# ---------------------------------------------------------------------------

def enforce_critical_closure(parts: list[Part]) -> list[Part]:
    """
    Ensure every ancestor of a critical part is also critical.
    Mutates parts in-place; returns the same list.
    """
    by_uid = {p.uid: p for p in parts}
    must_be_critical: set[str] = set()
    for p in parts:
        if p.critical:
            uid = p.parent_part
            while uid is not None and uid not in must_be_critical:
                must_be_critical.add(uid)
                uid = by_uid[uid].parent_part if uid in by_uid else None
    for p in parts:
        if p.uid in must_be_critical:
            p.critical = True
    return parts


def _part_volume(part: Part) -> float:
    """Approximate volume of a part in its local canonical frame."""
    d = part.dimensions
    w, dep, h = d.width, d.depth, d.height
    if part.primitive_type == PrimitiveType.CUBOID:
        return w * dep * h
    if part.primitive_type == PrimitiveType.ELLIPSOID:
        return (math.pi / 6) * w * dep * h
    if part.primitive_type == PrimitiveType.CYLINDER:
        r = w / 2
        return math.pi * r * r * h
    if part.primitive_type == PrimitiveType.CONE_FRUSTUM:
        R = w / 2
        top_r = part.top_radius if part.top_radius is not None else 0.0
        return (math.pi * h / 3) * (R * R + R * top_r + top_r * top_r)
    return w * dep * h  # fallback


def enforce_single_root(parts: list[Part]) -> list[Part]:
    """
    Ensure exactly one root part. If multiple roots exist, select the primary
    via (critical → largest volume → smallest UID) and reattach all others
    to it with parent_part=primary.uid, parent_face=None.
    Mutates parts in-place; returns the same list.
    """
    roots = [p for p in parts if p.parent_part is None]
    if len(roots) <= 1:
        return parts
    critical_roots = [p for p in roots if p.critical]
    candidates = critical_roots if critical_roots else roots
    primary = min(candidates, key=lambda p: (-_part_volume(p), p.uid))
    for p in roots:
        if p.uid != primary.uid:
            p.parent_part = primary.uid
            p.parent_face = None
    return parts


# ---------------------------------------------------------------------------
# Attachment  (Part-world, Z=UP)
# ---------------------------------------------------------------------------

# Unit outward normals for each bounding-cuboid face in Z=UP space.
_FACE_NORMAL: dict[str, tuple[float, float, float]] = {
    "top":    ( 0.0,  0.0, +1.0),
    "bottom": ( 0.0,  0.0, -1.0),
    "front":  ( 0.0, +1.0,  0.0),
    "back":   ( 0.0, -1.0,  0.0),
    "right":  (+1.0,  0.0,  0.0),
    "left":   (-1.0,  0.0,  0.0),
}


def _face_center(
    parent_center: tuple[float, float, float],
    parent_dims: PartDimensions,
    face: str,
) -> tuple[float, float, float]:
    """Center point of the named face on the parent's bounding cuboid."""
    cx, cy, cz = parent_center
    nx, ny, nz = _FACE_NORMAL[face]
    return (
        cx + nx * parent_dims.width / 2,
        cy + ny * parent_dims.depth / 2,
        cz + nz * parent_dims.height / 2,
    )


def _face_offset_vector(
    face: str,
    offset_u: float,
    offset_v: float,
    parent_dims: PartDimensions,
) -> tuple[float, float, float]:
    """
    2D face offset: (u, v) placement on a parent bounding-cuboid face.

    Each face is a 2D surface. offset_u and offset_v are both in [-1, 1],
    mapping to [-half_extent, +half_extent] along the two in-plane axes.

    Axis mapping per face (X=left/right, Y=front/back, Z=up):

      top    (XY plane, normal +Z)  →  u=X, v=Y
      bottom (XY plane, normal -Z)  →  u=X, v=Y
      front  (XZ plane, normal +Y)  →  u=X, v=Z
      back   (XZ plane, normal -Y)  →  u=X, v=Z
      left   (YZ plane, normal -X)  →  u=Y, v=Z
      right  (YZ plane, normal +X)  →  u=Y, v=Z
    """
    w, d, h = parent_dims.width / 2, parent_dims.depth / 2, parent_dims.height / 2
    if face in ("top", "bottom"):
        return (offset_u * w, offset_v * d, 0.0)
    elif face in ("front", "back"):
        return (offset_u * w, 0.0, offset_v * h)
    else:  # left, right
        return (0.0, offset_u * d, offset_v * h)


def _face_anchor(
    parent_center: tuple[float, float, float],
    parent_dims: PartDimensions,
    face: str,
    offset_u: float,
    offset_v: float = 0.0,
) -> tuple[float, float, float]:
    """Anchor point on the parent face: face center + 2D in-plane offset."""
    fc = _face_center(parent_center, parent_dims, face)
    dv = _face_offset_vector(face, offset_u, offset_v, parent_dims)
    return (fc[0] + dv[0], fc[1] + dv[1], fc[2] + dv[2])


def _surface_normal_at_anchor(
    parent: Part,
    parent_center: tuple[float, float, float],
    anchor: tuple[float, float, float],
    parent_face: str,
) -> np.ndarray:
    """
    Outward surface normal at the anchor point on the parent primitive.

    For top/bottom faces and cuboid parents, returns the axis-aligned face
    normal (identical to the bounding-box normal).

    For side faces on curved primitives (cylinder, cone_frustum, ellipsoid),
    returns the radial outward normal at the anchor's angular position so
    that children (windows, doors) orient tangent to the curved surface.
    """
    # Top/bottom: surface normal is ±Z for every shape.
    if parent_face in ("top", "bottom"):
        return np.array(_FACE_NORMAL[parent_face], dtype=float)

    ptype = parent.primitive_type

    # Cuboid: bounding-box normal is the surface normal.
    if ptype == PrimitiveType.CUBOID:
        return np.array(_FACE_NORMAL[parent_face], dtype=float)

    cx, cy, cz = parent_center
    ax, ay, az = anchor
    dx, dy = ax - cx, ay - cy

    if ptype in (PrimitiveType.CYLINDER, PrimitiveType.CONE_FRUSTUM):
        # Radial outward in XY plane.
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            return np.array(_FACE_NORMAL[parent_face], dtype=float)
        return np.array([dx / length, dy / length, 0.0])

    if ptype == PrimitiveType.ELLIPSOID:
        # Gradient of (x/a)²+(y/b)²+(z/c)²=1 gives outward normal.
        d = parent.dimensions
        a2 = (d.width / 2) ** 2 or 1.0
        b2 = (d.depth / 2) ** 2 or 1.0
        c2 = (d.height / 2) ** 2 or 1.0
        dz = az - cz
        gx, gy, gz = dx / a2, dy / b2, dz / c2
        length = math.sqrt(gx * gx + gy * gy + gz * gz)
        if length < 1e-9:
            return np.array(_FACE_NORMAL[parent_face], dtype=float)
        return np.array([gx / length, gy / length, gz / length])

    # Fallback: axis-aligned face normal.
    return np.array(_FACE_NORMAL[parent_face], dtype=float)


def _child_center_from_anchor(
    anchor: tuple[float, float, float],
    child_dims: PartDimensions,
    parent_face: str,
    child_face: str | None = None,
    surface_normal: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Place the child center so its contact face is flush with the anchor.

    Push direction is along the surface normal (or the axis-aligned parent
    face normal when no surface normal is provided).  Push distance is half
    the child's extent along the child_face normal axis.
    """
    if surface_normal is not None:
        nx, ny, nz = float(surface_normal[0]), float(surface_normal[1]), float(surface_normal[2])
    else:
        nx, ny, nz = _FACE_NORMAL[parent_face]
    # Determine push distance from child_face (which child dim faces the parent)
    cf = child_face or parent_face
    cnx, cny, cnz = _FACE_NORMAL[cf]
    half_along_normal = (
        abs(cnx) * child_dims.width / 2
        + abs(cny) * child_dims.depth / 2
        + abs(cnz) * child_dims.height / 2
    )
    return (
        anchor[0] + nx * half_along_normal,
        anchor[1] + ny * half_along_normal,
        anchor[2] + nz * half_along_normal,
    )


def _topological_order(parts: list[Part]) -> list[Part]:
    """
    BFS from root parts (parent_part=None) down through children.
    Parents always precede their children in the returned list.
    Assumes validate_graph has already eliminated cycles and dangling refs.
    """
    children: dict[str, list[Part]] = {p.uid: [] for p in parts}
    roots: list[Part] = []
    for p in parts:
        if p.parent_part is None:
            roots.append(p)
        else:
            children[p.parent_part].append(p)

    ordered: list[Part] = []
    queue = list(roots)
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        queue.extend(children[node.uid])
    return ordered


def attach_parts(parts: list[Part]) -> list[PartState]:
    """
    Compute world-space centers and attachment anchors for all parts (Z=UP).

    Root parts are placed at (0, 0, 0) with anchor=None. Grounding is deferred
    to final_placement. Children with an parent_face get their center placed
    flush against the parent face; the attachment anchor is stored on each
    returned PartState. Children with no parent_face fall back to the parent
    center with anchor=None.

    Call validate_graph before this function to ensure a clean tree.
    """
    by_uid_part = {p.uid: p for p in parts}
    states: dict[str, PartState] = {}
    ordered = _topological_order(parts)

    for p in ordered:
        if p.parent_part is None:
            states[p.uid] = PartState(
                uid=p.uid, dims=p.dimensions, center=(0.0, 0.0, 0.0)
            )
            continue

        parent = states[p.parent_part]

        if p.parent_face is None:
            states[p.uid] = PartState(
                uid=p.uid, dims=p.dimensions, center=parent.center
            )
            continue

        anchor = _face_anchor(
            parent.center, parent.dims, p.parent_face,
            p.attachment_offset, p.attachment_offset_v,
        )
        parent_part = by_uid_part[p.parent_part]
        sn = _surface_normal_at_anchor(
            parent_part, parent.center, anchor, p.parent_face,
        )
        center = _child_center_from_anchor(
            anchor, p.dimensions, p.parent_face, p.child_face,
            surface_normal=sn,
        )
        states[p.uid] = PartState(
            uid=p.uid, dims=p.dimensions, center=center, anchor=anchor,
            parent_surface_normal=sn,
        )

    return [states[p.uid] for p in ordered]


# ---------------------------------------------------------------------------
# Scaling  (Part-world, Z=UP)
# ---------------------------------------------------------------------------

# Maximum extents the assembled model may occupy (in Part-world units).
PART_BOUND_X = 100.0   # left/right
PART_BOUND_Y = 100.0   # front/back
PART_BOUND_Z = 100.0   # up


def _global_bbox(
    states: list[PartState],
) -> tuple[float, float, float, float, float, float]:
    """
    Return (min_x, min_y, min_z, max_x, max_y, max_z) over all parts.
    Each part's box is its center ± half its dimensions on each axis.
    Returns all zeros for an empty list.
    """
    if not states:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    min_x = min_y = min_z =  float("inf")
    max_x = max_y = max_z = -float("inf")
    for s in states:
        cx, cy, cz = s.center
        hx = s.dims.width / 2
        hy = s.dims.depth / 2
        hz = s.dims.height / 2
        min_x = min(min_x, cx - hx);  max_x = max(max_x, cx + hx)
        min_y = min(min_y, cy - hy);  max_y = max(max_y, cy + hy)
        min_z = min(min_z, cz - hz);  max_z = max(max_z, cz + hz)
    return min_x, min_y, min_z, max_x, max_y, max_z


def _uniform_scale_factor(
    min_x: float, min_y: float, min_z: float,
    max_x: float, max_y: float, max_z: float,
) -> float:
    """
    Uniform scale to fit within PART_BOUND_*, with a 0.5× buffer for rotation
    headroom (s = 0.5 × min(50/ex, 50/ey, 100/ez)).
    Axes with zero extent are skipped to avoid division by zero.
    Returns 0.5 if all extents are zero (degenerate input).
    """
    ratios: list[float] = []
    if (ex := max_x - min_x) > 0:
        ratios.append(PART_BOUND_X / ex)
    if (ey := max_y - min_y) > 0:
        ratios.append(PART_BOUND_Y / ey)
    if (ez := max_z - min_z) > 0:
        ratios.append(PART_BOUND_Z / ez)
    s_fit = min(ratios) if ratios else 1.0
    # Rotation buffer: 1/√2 ≈ 0.707 handles worst-case 45° diagonal expansion.
    # 0.5 was overly conservative.
    return 0.7 * s_fit


def _states_for_scale(
    parts: list[Part],
    states: list[PartState],
) -> list[PartState]:
    """
    Return a filtered/adjusted list of PartStates for scale computation.

    - Non-critical parts with any dimension < 0.05 are excluded.
    - Critical parts with any dimension < 0.05 are thickened to 0.05 for
      those dimensions (copy only — input states are not mutated).
    - All other parts are included unchanged.
    """
    THRESH = 0.05
    by_uid_part = {p.uid: p for p in parts}
    result: list[PartState] = []
    for s in states:
        p = by_uid_part.get(s.uid)
        d = s.dims
        small = d.width < THRESH or d.depth < THRESH or d.height < THRESH
        if not small:
            result.append(s)
            continue
        if p is None or not p.critical:
            continue  # exclude non-critical tiny parts
        # Critical: thicken sub-threshold dims in a copy.
        result.append(PartState(
            uid=s.uid,
            dims=PartDimensions(
                width=max(d.width, THRESH),
                depth=max(d.depth, THRESH),
                height=max(d.height, THRESH),
            ),
            center=s.center,
            anchor=s.anchor,
        ))
    return result


def compute_scale(parts: list[Part], states: list[PartState]) -> float:
    """Compute the uniform scale factor from the pruning-aware subset of states."""
    effective = _states_for_scale(parts, states)
    if not effective:
        return 1.0
    return _uniform_scale_factor(*_global_bbox(effective))


def apply_scale(states: list[PartState], s: float) -> list[PartState]:
    """Apply uniform scale s to dims, centers, and anchors (None anchor passthrough)."""
    result = []
    for st in states:
        d = st.dims
        new_anchor = None
        if st.anchor is not None:
            new_anchor = (st.anchor[0] * s, st.anchor[1] * s, st.anchor[2] * s)
        result.append(PartState(
            uid=st.uid,
            dims=PartDimensions(
                width=d.width * s,
                depth=d.depth * s,
                height=d.height * s,
            ),
            center=(st.center[0] * s, st.center[1] * s, st.center[2] * s),
            anchor=new_anchor,
        ))
    return result


def final_placement(states: list[PartState]) -> list[PartState]:
    """
    Translate the model so it is horizontally centered at (25, 25) and
    grounded with min_z = 0. Applied BEFORE voxelization in continuous space.
    Applies to centers and anchors (None anchor passthrough).
    """
    if not states:
        return []
    min_x, min_y, min_z, max_x, max_y, _ = _global_bbox(states)
    tx = 50.0 - (min_x + max_x) / 2
    ty = 50.0 - (min_y + max_y) / 2
    tz = -min_z
    result = []
    for st in states:
        new_anchor = None
        if st.anchor is not None:
            new_anchor = (
                st.anchor[0] + tx,
                st.anchor[1] + ty,
                st.anchor[2] + tz,
            )
        result.append(PartState(
            uid=st.uid,
            dims=st.dims,
            center=(st.center[0] + tx, st.center[1] + ty, st.center[2] + tz),
            anchor=new_anchor,
        ))
    return result


# ---------------------------------------------------------------------------
# Rotation  (Part-world, Z=UP)
# ---------------------------------------------------------------------------

@dataclass
class _Transform:
    """Accumulated rigid transform: maps pre-rotation point p → R @ p + t."""
    R: np.ndarray  # 3×3 rotation matrix (column-vector convention)
    t: np.ndarray  # 3-vector translation


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Build a 3×3 rotation matrix from Euler angles (degrees).
    Applied Z → Y → X (spec §15, column-vector convention): R = Rx @ Ry @ Rz.
    """
    rx_r, ry_r, rz_r = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(rx_r), math.sin(rx_r)
    cy, sy = math.cos(ry_r), math.sin(ry_r)
    cz, sz = math.cos(rz_r), math.sin(rz_r)
    Rx = np.array([[1,   0,   0 ],
                   [0,  cx,  -sx],
                   [0,  sx,   cx]], dtype=float)
    Ry = np.array([[ cy,  0,  sy],
                   [  0,  1,   0],
                   [-sy,  0,  cy]], dtype=float)
    Rz = np.array([[cz, -sz,  0 ],
                   [sz,  cz,  0 ],
                   [ 0,   0,  1 ]], dtype=float)
    return Rx @ Ry @ Rz


def _apply_transform(
    tf: _Transform, p: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Apply a _Transform to a 3D point."""
    v = tf.R @ np.array(p, dtype=float) + tf.t
    return (float(v[0]), float(v[1]), float(v[2]))


def _canonical_root_rotation(part: Part) -> np.ndarray:
    """
    Rotation that aligns the root's primary axis to +Z.

    - Cylinder / cone_frustum: central axis is already height (Z) → identity.
    - Cuboid / ellipsoid: longest dimension is rotated to Z.

    For cone_frustum the wider base is already at -Z in local space (see
    _containment), so no flip is needed.
    """
    ptype = part.primitive_type
    if ptype in (PrimitiveType.CYLINDER, PrimitiveType.CONE_FRUSTUM):
        return np.eye(3)

    d = part.dimensions
    w, dp, h = d.width, d.depth, d.height
    if h >= w and h >= dp:
        return np.eye(3)                       # height already longest → Z
    if w >= dp:
        return _rotation_matrix(0, -90, 0)     # width (X) → Z
    return _rotation_matrix(90, 0, 0)          # depth (Y) → Z


def _face_align_rotation(
    parent_face: str,
    child_face: str | None,
    surface_normal: np.ndarray | None = None,
) -> np.ndarray:
    """
    Rotation that orients a child so its *child_face* normal opposes the
    parent surface normal (faces point toward each other).

    When *surface_normal* is provided (curved parent), it replaces the
    axis-aligned parent-face normal so that children sit tangent to curved
    surfaces (cylinders, frustums, ellipsoids).

    When child_face is None or already opposes the target, returns identity.
    """
    if child_face is None:
        return np.eye(3)

    if surface_normal is not None:
        n_p = surface_normal
    else:
        n_p = np.array(_FACE_NORMAL[parent_face], dtype=float)
    n_c = np.array(_FACE_NORMAL[child_face], dtype=float)
    target = -n_p  # desired direction for child_face normal

    # Already aligned — identity.
    if np.allclose(n_c, target):
        return np.eye(3)

    # Opposite direction — 180° around a perpendicular axis.
    if np.allclose(n_c, -target):
        perp = np.array([0, 0, 1.0]) if abs(n_c[2]) < 0.5 else np.array([1, 0, 0.0])
        return 2 * np.outer(perp, perp) - np.eye(3)

    # General case: Rodrigues rotation from n_c to target.
    v = np.cross(n_c, target)
    s = np.linalg.norm(v)
    c = float(np.dot(n_c, target))
    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]], dtype=float)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def apply_rotation(parts: list[Part], states: list[PartState]) -> list[PartState]:
    """
    Apply hierarchical rotations to all parts in topological order.

    For each part the effective local rotation is:
        R_own = R_llm  @  R_canonical
    where R_canonical is:
      - Root: _canonical_root_rotation (aligns primary axis to +Z)
      - Child with parent_face: _face_align_rotation (aligns child_face
        to oppose parent_face)
      - Child without parent_face: identity

    R_llm is the LLM-specified Euler rotation (a delta from the canonical pose).

    Each part then accumulates its parent's transform and applies R_own.
    Root: pivots around its own center.
    Child with anchor: pivots around the parent-transformed anchor point.
    Child without anchor: pivots around the parent-transformed center.

    Updates center and anchor. dims are unchanged (local canonical extents).
    """
    by_uid_part = {p.uid: p for p in parts}
    by_uid_state = {s.uid: s for s in states}
    transforms: dict[str, _Transform] = {}
    out: dict[str, PartState] = {}

    for p in _topological_order(parts):
        s = by_uid_state[p.uid]
        rot = p.rotation
        R_llm = _rotation_matrix(rot.rx, rot.ry, rot.rz)

        # Canonical pre-rotation
        if p.parent_part is None:
            R_canon = _canonical_root_rotation(p)
        elif p.parent_face is not None:
            R_canon = _face_align_rotation(
                p.parent_face, p.child_face,
                surface_normal=s.parent_surface_normal,
            )
        else:
            R_canon = np.eye(3)

        R_own = R_llm @ R_canon

        if p.parent_part is None:
            # Root: pivots around its own center — center is invariant.
            c = np.array(s.center, dtype=float)
            R_acc = R_own
            t_acc = c - R_own @ c
        else:
            R_par = transforms[p.parent_part].R
            t_par = transforms[p.parent_part].t

            if s.anchor is not None:
                # Pivot around parent-transformed anchor.
                A = np.array(s.anchor, dtype=float)
                A_world = R_par @ A + t_par
                R_acc = R_own @ R_par
                t_acc = R_own @ t_par + A_world - R_own @ A_world
            else:
                # No anchor: pivot around parent-transformed center.
                C = np.array(s.center, dtype=float)
                C_world = R_par @ C + t_par
                R_acc = R_own @ R_par
                t_acc = R_own @ t_par + C_world - R_own @ C_world

        transforms[p.uid] = _Transform(R=R_acc, t=t_acc)
        tf = transforms[p.uid]
        new_center = _apply_transform(tf, s.center)
        new_anchor = _apply_transform(tf, s.anchor) if s.anchor is not None else None
        out[p.uid] = PartState(
            uid=p.uid, dims=s.dims, center=new_center, anchor=new_anchor,
            world_R=R_acc,
        )

    return [out[s.uid] for s in states if s.uid in out]


def _reground(states: list[PartState]) -> list[PartState]:
    """Shift all parts so the model's lowest voxel-extent sits at z=0.

    After rotation, parts may have moved below z=0. This computes the
    post-rotation global bbox and applies a vertical shift.
    """
    if not states:
        return []
    _, _, min_z, _, _, _ = _global_bbox(states)
    if abs(min_z) < 1e-6:
        return states
    tz = -min_z
    result = []
    for s in states:
        new_center = (s.center[0], s.center[1], s.center[2] + tz)
        new_anchor = None
        if s.anchor is not None:
            new_anchor = (s.anchor[0], s.anchor[1], s.anchor[2] + tz)
        result.append(PartState(
            uid=s.uid, dims=s.dims, center=new_center,
            anchor=new_anchor, world_R=s.world_R,
        ))
    return result


# ---------------------------------------------------------------------------
# Voxelization  (Part-world, Z=UP)
# ---------------------------------------------------------------------------

# 1× grid dimensions (world-space units = voxels after final_placement).
GRID_X: int = 100
GRID_Y: int = 100
GRID_Z: int = 100

# Sparse claims map: 1× voxel coord → frozenset of claiming part UIDs.
VoxelCoord = tuple[int, int, int]
VoxelClaims = dict[VoxelCoord, frozenset[str]]


def _containment(
    part: Part,
    dims: PartDimensions,
    lx: np.ndarray,
    ly: np.ndarray,
    lz: np.ndarray,
) -> np.ndarray:
    """
    Return a boolean mask (same shape as lx/ly/lz) indicating which sample
    points lie inside the primitive's local-space geometry.

    All inputs are in local canonical space (origin at part center, axes
    aligned with width/depth/height). Half-extents are derived internally.
    """
    hx = dims.width  / 2
    hy = dims.depth  / 2
    hz = dims.height / 2
    ptype = part.primitive_type

    if ptype == PrimitiveType.CUBOID:
        return (np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(lz) <= hz)

    if ptype == PrimitiveType.ELLIPSOID:
        return (lx / hx) ** 2 + (ly / hy) ** 2 + (lz / hz) ** 2 <= 1.0

    if ptype == PrimitiveType.CYLINDER:
        # Circular/elliptical cross-section in XY, axis along Z.
        return ((lx / hx) ** 2 + (ly / hy) ** 2 <= 1.0) & (np.abs(lz) <= hz)

    if ptype == PrimitiveType.CONE_FRUSTUM:
        # Axis along Z; base (radius = hx) at z = -hz, top at z = +hz.
        top_r = part.top_radius if part.top_radius is not None else 0.0
        t = np.clip((lz + hz) / (2 * hz), 0.0, 1.0)  # 0 at base, 1 at top
        r_at_z = hx + (top_r - hx) * t
        return (np.abs(lz) <= hz) & (np.sqrt(lx ** 2 + ly ** 2) <= r_at_z)

    # Unknown type — treat as cuboid.
    return (np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(lz) <= hz)


def voxelize(parts: list[Part], states: list[PartState]) -> VoxelClaims:
    """
    Rasterize all parts into a 2× bitmask grid, then downsample to a 1× claims map.

    Internally uses a (100, 100, 200) int64 bitmask grid (2× per axis). For each part:
      1. Compute world-space AABB using |world_R| @ half_extents.
      2. Clamp candidate range to 2× grid bounds.
      3. Sample each candidate 2× voxel center; transform to local space via
         world_R.T @ (p - center); run primitive containment test.
      4. OR bit `idx` (0-based) into every 2× voxel the part occupies.
         Multiple parts that cover the same 2× voxel accumulate their bits;
         no part can overwrite another.

    Downsampling (2×→1×):
      Reshape the 2× grid to (X, Y, Z, 8) and bitwise-OR all 8 sub-voxel
      bitmasks per block. Each set bit i in the combined value means states[i]
      claims that 1× voxel. Empty blocks produce no entry.

    Output is a sparse VoxelClaims dict. All claimants for each voxel are
    preserved. apply_ownership resolves conflicts; this function does not.

    Supports up to 62 parts (int64 bitmask, sign bit reserved).
    """
    _SCALE = 2
    assert len(states) < 63, f"voxelize supports at most 62 parts, got {len(states)}"
    grid2 = np.zeros((GRID_X * _SCALE, GRID_Y * _SCALE, GRID_Z * _SCALE), dtype=np.int64)
    by_uid = {p.uid: p for p in parts}

    for idx, s in enumerate(states):
        part = by_uid.get(s.uid)
        if part is None:
            continue
        R = s.world_R if s.world_R is not None else np.eye(3)
        c = np.array(s.center, dtype=float)
        d = s.dims
        half = np.array([d.width / 2, d.depth / 2, d.height / 2])

        # World-space AABB of the rotated primitive.
        aabb_half = np.abs(R) @ half

        # 2× voxel index range, clamped to grid bounds.
        lo = np.floor((c - aabb_half) * _SCALE).astype(int)
        hi = np.ceil( (c + aabb_half) * _SCALE).astype(int) + 1
        lo = np.clip(lo, 0, [GRID_X*_SCALE - 1, GRID_Y*_SCALE - 1, GRID_Z*_SCALE - 1])
        hi = np.clip(hi, 0, [GRID_X*_SCALE,     GRID_Y*_SCALE,     GRID_Z*_SCALE    ])

        if np.any(lo >= hi):
            continue  # entirely outside grid

        # Sample centers of 2× voxels in world space.
        xs = (np.arange(lo[0], hi[0]) + 0.5) / _SCALE
        ys = (np.arange(lo[1], hi[1]) + 0.5) / _SCALE
        zs = (np.arange(lo[2], hi[2]) + 0.5) / _SCALE
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")

        # Transform to local space: p_local = R.T @ (p_world - center).
        # In batched row-vector form: pts_local = pts @ R  (where pts = p - c).
        pts = np.stack([gx - c[0], gy - c[1], gz - c[2]], axis=-1)  # (..., 3)
        pts_local = pts @ R  # equivalent to (R.T @ col_vec) for each point

        inside = _containment(part, d, pts_local[..., 0], pts_local[..., 1], pts_local[..., 2])
        # OR bit idx into all occupied 2× voxels. Each part accumulates independently;
        # later parts do not erase earlier ones.
        grid2[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][inside] |= np.int64(1) << idx

    # Downsample 2×→1×: bitwise-OR all 8 sub-voxel bitmasks per 1× block.
    # Reshape to (X, 2, Y, 2, Z, 2) → transpose to (X, Y, Z, 2, 2, 2) → flatten last 3.
    combined = np.bitwise_or.reduce(
        grid2
        .reshape(GRID_X, _SCALE, GRID_Y, _SCALE, GRID_Z, _SCALE)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(GRID_X, GRID_Y, GRID_Z, _SCALE ** 3),
        axis=-1,
    )  # shape (GRID_X, GRID_Y, GRID_Z), dtype int64

    uid_by_idx = [s.uid for s in states]  # 0-indexed: uid_by_idx[i] = states[i].uid
    occ_i, occ_j, occ_k = np.where(combined != 0)
    claims: VoxelClaims = {}
    for i, j, k in zip(occ_i.tolist(), occ_j.tolist(), occ_k.tolist()):
        mask = int(combined[i, j, k])
        claims[(i, j, k)] = frozenset(
            uid_by_idx[b] for b in range(len(states)) if (mask >> b) & 1
        )
    return claims


# ---------------------------------------------------------------------------
# Ownership  (Part-world, Z=UP)
# ---------------------------------------------------------------------------

def _build_ancestors(parts: list[Part]) -> dict[str, set[str]]:
    """Return a map uid → frozenset of all ancestor UIDs (parent, grandparent, …)."""
    by_uid = {p.uid: p for p in parts}
    ancestors: dict[str, set[str]] = {}
    for p in parts:
        anc: set[str] = set()
        uid = p.parent_part
        while uid is not None and uid in by_uid and uid not in anc:
            anc.add(uid)
            uid = by_uid[uid].parent_part
        ancestors[p.uid] = anc
    return ancestors


def apply_ownership(
    parts: list[Part],
    states: list[PartState],
    claims: VoxelClaims,
) -> np.ndarray:
    """
    Resolve each contested voxel in claims to a single owner.

    Priority (applied in order):
      1. Descendant beats ancestor (any depth).
      2. Critical beats non-critical.
      3. Volume rule (< 20% threshold — part loses voxels it under-represents).
      4. Smallest UID tiebreak.

    Returns a (50, 50, 100) int32 grid: 0 = empty, n = states[n-1] owns voxel.
    Uncontested voxels are assigned directly without applying the priority chain.
    """
    grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
    state_idx = {s.uid: i + 1 for i, s in enumerate(states)}
    ancestors_of = _build_ancestors(parts)
    is_critical = {p.uid: p.critical for p in parts}
    vol_by_uid = {p.uid: _part_volume(p) for p in parts}

    def _compare(a_uid: str, b_uid: str) -> int:
        # Rule 1: descendant beats ancestor (any depth).
        if b_uid in ancestors_of.get(a_uid, set()):
            return -1  # a is descendant of b → a wins
        if a_uid in ancestors_of.get(b_uid, set()):
            return 1   # b is descendant of a → b wins
        # Rule 2: critical beats non-critical.
        ca = is_critical.get(a_uid, False)
        cb = is_critical.get(b_uid, False)
        if ca and not cb:
            return -1
        if cb and not ca:
            return 1
        # Rule 3: volume rule (smaller wins if < 20% of larger).
        va = vol_by_uid.get(a_uid, 0.0)
        vb = vol_by_uid.get(b_uid, 0.0)
        if va < 0.2 * vb:
            return -1
        if vb < 0.2 * va:
            return 1
        # Rule 4: smallest UID tiebreak.
        if a_uid < b_uid:
            return -1
        if a_uid > b_uid:
            return 1
        return 0

    for (i, j, k), claimants in claims.items():
        if not claimants:
            continue
        winner = min(claimants, key=cmp_to_key(_compare))
        idx = state_idx.get(winner)
        if idx is not None:
            grid[i, j, k] = idx

    return grid


# ---------------------------------------------------------------------------
# Critical restoration  (Part-world, Z=UP)
# ---------------------------------------------------------------------------

def critical_restoration(
    parts: list[Part],
    states: list[PartState],
    grid: np.ndarray,
) -> np.ndarray:
    """
    Restore exactly one voxel for any critical part that owns zero voxels.

    The restored voxel is placed at floor(center), clamped to grid bounds.
    Overwrites whatever occupies that voxel. Non-critical parts are untouched.
    Modifies grid in-place and returns it.
    """
    state_idx = {s.uid: i + 1 for i, s in enumerate(states)}
    by_uid_state = {s.uid: s for s in states}

    for p in parts:
        if not p.critical:
            continue
        idx = state_idx.get(p.uid)
        if idx is None:
            continue
        if np.any(grid == idx):
            continue  # already represented
        s = by_uid_state.get(p.uid)
        if s is None:
            continue
        cx, cy, cz = s.center
        i = max(0, min(GRID_X - 1, int(math.floor(cx))))
        j = max(0, min(GRID_Y - 1, int(math.floor(cy))))
        k = max(0, min(GRID_Z - 1, int(math.floor(cz))))
        grid[i, j, k] = idx

    return grid


# ---------------------------------------------------------------------------
# Connectivity enforcement  (Part-world, Z=UP)
# ---------------------------------------------------------------------------

# Maps each attach-face name to (axis_index, direction_sign) for bridge selection.
# axis 0=X, 1=Y, 2=Z; direction +1 = positive side of parent, -1 = negative side.
_FACE_AXIS_DIR: dict[str, tuple[int, int]] = {
    "top":    (2, +1),
    "bottom": (2, -1),
    "front":  (1, +1),
    "back":   (1, -1),
    "right":  (0, +1),
    "left":   (0, -1),
}

# 6-connectivity (face-adjacent only)
_NEIGHBORS_6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# 26-connectivity (Chebyshev ≤ 1: face + edge + corner neighbors) — spec §3.3
_NEIGHBORS_26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def _flood_fill(grid: np.ndarray) -> np.ndarray:
    """
    26-connected (Chebyshev ≤ 1) flood fill over all occupied voxels.
    Returns a label array (same shape as grid): 0 = empty, ≥1 = component ID.
    """
    labels = np.zeros_like(grid)
    comp_id = 0
    for start_arr in np.argwhere(grid > 0):
        si, sj, sk = int(start_arr[0]), int(start_arr[1]), int(start_arr[2])
        if labels[si, sj, sk] != 0:
            continue
        comp_id += 1
        labels[si, sj, sk] = comp_id
        q: deque[tuple[int, int, int]] = deque([(si, sj, sk)])
        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in _NEIGHBORS_26:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < GRID_X and 0 <= ny < GRID_Y and 0 <= nz < GRID_Z:
                    if grid[nx, ny, nz] != 0 and labels[nx, ny, nz] == 0:
                        labels[nx, ny, nz] = comp_id
                        q.append((nx, ny, nz))
    return labels


def _voxels_of(grid: np.ndarray, part_idx: int) -> list[tuple[int, int, int]]:
    """Return list of (i, j, k) coordinates owned by part_idx."""
    ii, jj, kk = np.where(grid == part_idx)
    return list(zip(ii.tolist(), jj.tolist(), kk.tolist()))


def _lay_bridge(
    grid: np.ndarray,
    src: tuple[int, int, int],
    dst: tuple[int, int, int],
    bridge_idx: int,
    critical_idxs: set[int],
) -> None:
    """
    Lay an axis-aligned bridge from src toward dst (exclusive of both endpoints).

    Walks X → Y → Z in order. Overwrites empty and non-critical voxels with
    bridge_idx; critical voxels are left unchanged (they already contribute to
    connectivity where they occur on the path).
    """
    x, y, z = src
    tx, ty, tz = dst

    def _step(pos: int, target: int) -> int:
        if pos < target:
            return pos + 1
        if pos > target:
            return pos - 1
        return pos

    while (x, y, z) != (tx, ty, tz):
        if x != tx:
            x = _step(x, tx)
        elif y != ty:
            y = _step(y, ty)
        else:
            z = _step(z, tz)

        if (x, y, z) == (tx, ty, tz):
            break  # don't overwrite destination voxel

        if grid[x, y, z] not in critical_idxs:
            grid[x, y, z] = bridge_idx


def enforce_connectivity(
    parts: list[Part],
    states: list[PartState],
    grid: np.ndarray,
) -> np.ndarray:
    """
    Ensure all critical parts are 6-connected in the voxel grid.

    Pass 1 — Bridge disconnected critical parent-child pairs (topological order):
      For each critical child not yet connected to its critical parent, select
      a parent voxel biased toward the attach face and the nearest child voxel,
      then lay an axis-aligned bridge (parent_idx fills gaps). Critical voxels
      along the path are left unchanged; they still maintain connectivity.

    Pass 2 — Remove non-critical disconnected components:
      After bridging, flood-fill to find the main critical component. Voxels
      belonging to other components are zeroed unless they are critical.

    Returns the (possibly new) grid array.
    """
    state_idx = {s.uid: i + 1 for i, s in enumerate(states)}
    by_uid_part = {p.uid: p for p in parts}
    critical_uids = {p.uid for p in parts if p.critical}
    critical_idxs = {state_idx[uid] for uid in critical_uids if uid in state_idx}

    # --- Pass 1: bridge disconnected parent-child pairs ---
    # Bridge ALL parent-child pairs (not just critical) to prevent
    # non-critical parts from being pruned as disconnected in Pass 2.
    for p in _topological_order(parts):
        if p.parent_part is None:
            continue
        parent = by_uid_part.get(p.parent_part)
        if parent is None:
            continue

        child_idx = state_idx.get(p.uid)
        parent_idx = state_idx.get(p.parent_part)
        if child_idx is None or parent_idx is None:
            continue

        child_voxels = _voxels_of(grid, child_idx)
        parent_voxels = _voxels_of(grid, parent_idx)
        if not child_voxels or not parent_voxels:
            continue

        labels = _flood_fill(grid)
        parent_labels = {labels[v] for v in parent_voxels}
        child_labels = {labels[v] for v in child_voxels}
        if parent_labels & child_labels:
            continue  # already connected

        # Select source: parent voxels biased toward the attach-face direction.
        if p.parent_face is not None:
            axis, direction = _FACE_AXIS_DIR[p.parent_face]
            n_top = max(1, len(parent_voxels) // 5)
            parent_sorted = sorted(
                parent_voxels, key=lambda v: direction * v[axis], reverse=True
            )
            top_parents = parent_sorted[:n_top]
        else:
            top_parents = parent_voxels

        child_center = np.array(child_voxels, dtype=float).mean(axis=0)
        src = min(
            top_parents,
            key=lambda v: float(np.sum((np.array(v, dtype=float) - child_center) ** 2)),
        )

        # Select destination: child voxel closest to chosen source.
        src_arr = np.array(src, dtype=float)
        dst = min(
            child_voxels,
            key=lambda v: float(np.sum((np.array(v, dtype=float) - src_arr) ** 2)),
        )

        _lay_bridge(grid, src, dst, parent_idx, critical_idxs)

    # --- Pass 2: remove non-critical disconnected components ---
    labels = _flood_fill(grid)

    critical_label_counts: dict[int, int] = {}
    for uid in critical_uids:
        idx = state_idx.get(uid)
        if idx is None:
            continue
        for v in _voxels_of(grid, idx):
            lbl = int(labels[v])
            if lbl > 0:
                critical_label_counts[lbl] = critical_label_counts.get(lbl, 0) + 1

    if not critical_label_counts:
        return grid

    main_label = max(critical_label_counts, key=lambda lbl: critical_label_counts[lbl])

    # Zero out non-critical voxels outside the main component.
    not_main = (labels != main_label) & (labels != 0)
    not_critical = ~np.isin(grid, list(critical_idxs)) if critical_idxs else np.ones_like(grid, dtype=bool)
    return np.where(not_main & not_critical, 0, grid).astype(np.int32)


# ---------------------------------------------------------------------------
# Part-world orchestration  (single authoritative execution path)
# ---------------------------------------------------------------------------

def build_part_world(parts: list[Part]) -> np.ndarray:
    """
    Full Part-world geometry pipeline. This is the only authoritative path
    from a validated Part list to a resolved voxel grid.

    Steps (executed in order — do NOT call these individually in production):
      1. validate_graph           — fix dangling parents, break cycles
      2. enforce_critical_closure — promote ancestors of critical parts
      3. enforce_single_root      — select one root; reattach extras
      4. attach_parts             — compute world-space centers + anchors
      5. compute_scale            — derive uniform scale with 0.5× rotation buffer
      6. apply_scale              — scale dims, centers, anchors
      7. final_placement          — center at (25,25), ground at z=0
      8. apply_rotation           — hierarchical scene-graph rotations; sets world_R
      9. voxelize                 — rasterize claims (VoxelClaims)
     10. apply_ownership          — resolve contested voxels to single owners
     11. critical_restoration     — restore one voxel for vanished critical parts
     12. enforce_connectivity     — bridge disconnected critical parts; prune orphans

    Returns a (50, 50, 100) int32 ndarray: 0 = empty, n = states[n-1] owns voxel.
    """
    validate_graph(parts)
    enforce_critical_closure(parts)
    enforce_single_root(parts)
    states = attach_parts(parts)
    s = compute_scale(parts, states)
    states = apply_scale(states, s)
    states = final_placement(states)
    states = apply_rotation(parts, states)
    # Re-ground after rotation: rotation can shift parts below z=0
    states = _reground(states)
    claims = voxelize(parts, states)
    grid = apply_ownership(parts, states, claims)
    grid = critical_restoration(parts, states, grid)
    grid = enforce_connectivity(parts, states, grid)
    return grid


# ---------------------------------------------------------------------------
# Public entry point  (thin wrapper — all logic stays in build_part_world)
# ---------------------------------------------------------------------------

def run_part_world(
    parts: list[Part],
    debug: bool = False,
) -> "np.ndarray | dict":
    """
    Public entry point for the Part-world geometry pipeline.

    Always deep-copies the input so the caller's Part list is never mutated.

    Args:
        parts: list of Part objects describing the model.
        debug: if False (default), returns the final (50, 50, 100) int32 voxel
               grid. If True, returns a dict with intermediate artifacts:
                 'grid'           — final voxel grid (50×50×100 int32)
                 'states'         — list[PartState] after all pipeline stages
                 'scale'          — uniform scale factor applied
                 'voxel_counts'   — {uid: count} after ownership resolution
                 'total_occupied' — total occupied voxels in the final grid

    Returns:
        np.ndarray (debug=False) or dict (debug=True).
    """
    import copy
    p = copy.deepcopy(parts)

    if not debug:
        return build_part_world(p)

    # Debug path: run the same stages as build_part_world, capturing intermediates.
    # All geometric logic lives in the individual stage functions — nothing duplicated.
    validate_graph(p)
    enforce_critical_closure(p)
    enforce_single_root(p)
    states = attach_parts(p)
    scale = compute_scale(p, states)
    states = apply_scale(states, scale)
    states = final_placement(states)
    states = apply_rotation(p, states)
    claims = voxelize(p, states)
    grid = apply_ownership(p, states, claims)
    grid = critical_restoration(p, states, grid)
    grid = enforce_connectivity(p, states, grid)

    return {
        "grid": grid,
        "states": states,
        "scale": scale,
        "voxel_counts": {s.uid: int((grid == i + 1).sum()) for i, s in enumerate(states)},
        "total_occupied": int((grid > 0).sum()),
    }


# ---------------------------------------------------------------------------
# Stage 1: Image loading
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> dict:
    """Read image from disk, return a dict with metadata and base64 payload."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    raw = path.read_bytes()
    return {
        "filename": path.name,
        "format": path.suffix.lstrip(".").lower(),
        "size_bytes": len(raw),
        "base64": base64.b64encode(raw).decode("utf-8"),
    }


# ---------------------------------------------------------------------------
# Stage 2: LLM image interpretation (placeholder)
# ---------------------------------------------------------------------------

def interpret_image(image_data: dict) -> dict:
    """
    Placeholder: send image to an LLM and return a structured scene description.

    Expected output shape:
    {
        "subject": str,          # e.g. "a red racing car"
        "category": str,         # broad class, e.g. "vehicle", "animal", "building"
        "symmetry": str,         # "symmetric" | "asymmetric"
        "objects": [
            {
                "label": str,            # e.g. "body", "wheel", "roof"
                "color": str,            # hex, e.g. "#FF0000"
                "relative_size": str,    # "small" | "medium" | "large"
                "primitive_hint": str,   # "cuboid" | "cylinder" | "wedge"
            }
        ]
    }
    """
    # TODO: call Claude / vision model with image_data["base64"]
    return {
        "subject": "a red car (stub)",
        "category": "vehicle",
        "symmetry": "symmetric",
        "objects": [
            {"label": "body",  "color": "#CC0000", "relative_size": "large",  "primitive_hint": "cuboid",   "relative_position": "center"},
            {"label": "wheel", "color": "#222222", "relative_size": "medium", "primitive_hint": "cylinder", "relative_position": "under"},
            {"label": "roof",  "color": "#CC0000", "relative_size": "small",  "primitive_hint": "wedge",    "relative_position": "on_top"},
        ],
    }


# ---------------------------------------------------------------------------
# Stage 3: Primitive decomposition (placeholder)
# ---------------------------------------------------------------------------

_SIZE_DIMS = {
    #              (width, height, depth)   used for cuboid + wedge
    "small":       (2, 2, 2),
    "medium":      (4, 3, 4),
    "large":       (6, 6, 6),
}

_SIZE_CYLINDER = {
    #              (diameter, height)
    "small":       (2, 2),
    "medium":      (4, 3),
    "large":       (6, 6),
}

_HINT_TO_TYPE = {
    "cuboid":   PrimitiveType.CUBOID,
    "cylinder": PrimitiveType.CYLINDER,
    "wedge":    PrimitiveType.WEDGE,
}


def _make_dimensions(hint: str, size: str):
    if hint == "cylinder":
        diameter, height = _SIZE_CYLINDER.get(size, _SIZE_CYLINDER["medium"])
        return CylinderDimensions(diameter=diameter, height=height)
    w, h, d = _SIZE_DIMS.get(size, _SIZE_DIMS["medium"])
    if hint == "wedge":
        return WedgeDimensions(width=w, height=h, depth=d)
    return CuboidDimensions(width=w, height=h, depth=d)


def _footprint(dims) -> tuple[int, int]:
    """Return (width_x, depth_z) for any dimensions type."""
    if isinstance(dims, CylinderDimensions):
        return dims.diameter, dims.depth_studs if dims.depth_studs is not None else dims.diameter
    return dims.width, dims.depth


def _y_offset(rel_pos: str, dims, body_height: int) -> tuple[int, int]:
    """
    Return (y, x_nudge) for a shape given its relative_position.

    - "center" / default: y=0, no x nudge
    - "under":            y=0, x nudge of +1 stud (offset within slot)
    - "on_top":           y=body_height, no x nudge
    """
    if rel_pos == "on_top":
        return body_height, 0
    if rel_pos == "under":
        return 0, 1
    return 0, 0  # "center" or unrecognised


def decompose_into_primitives(scene: dict) -> list[Shape]:
    """
    Convert a scene description into a list of Shapes.

    Slot distribution: objects spread evenly along X, centred on Z.
    Y placement: driven by each object's optional relative_position field.
      - "center" (default): y=0
      - "under":            y=0, +1 stud x-nudge within slot
      - "on_top":           y = height of the "body" shape
    Dimensions: determined by relative_size + primitive_hint.
    TODO: replace with real decomposition logic (rule-based or LLM-assisted).
    """
    objects = scene.get("objects", [])
    n = len(objects)

    # Pre-pass: resolve all dims so on_top can reference body height.
    all_dims = [
        _make_dimensions(obj.get("primitive_hint", "cuboid"), obj.get("relative_size", "medium"))
        for obj in objects
    ]
    body_height = next(
        (d.height for obj, d in zip(objects, all_dims) if obj.get("label") == "body"),
        0,
    )

    # Find the body's slot origin. All grouped objects share it.
    # Fall back to per-object slot distribution when no body is present.
    body_index = next(
        (i for i, obj in enumerate(objects) if obj.get("label") == "body"),
        None,
    )
    slot_width = BASEPLATE_WIDTH // max(n, 1)
    if body_index is not None:
        body_fw, _ = _footprint(all_dims[body_index])
        body_slot_x = slot_width * body_index + (slot_width - body_fw) // 2
    else:
        body_slot_x = None

    shapes = []

    for i, (obj, dims) in enumerate(zip(objects, all_dims)):
        fw, fd = _footprint(dims)
        hint = obj.get("primitive_hint", "cuboid")
        rel_pos = obj.get("relative_position", "center")
        y, x_nudge = _y_offset(rel_pos, dims, body_height)

        if body_slot_x is not None:
            # All objects anchor to the body's slot; only x_nudge varies within it.
            x = body_slot_x + x_nudge
        else:
            x = slot_width * i + (slot_width - fw) // 2 + x_nudge
        z = (BASEPLATE_DEPTH - fd) // 2

        shapes.append(
            Shape(
                id=f"shape_{i}",
                type=_HINT_TO_TYPE.get(hint, PrimitiveType.CUBOID),
                position=Position(x=x, y=y, z=z),
                dimensions=dims,
                rotation=0,
                color=obj.get("color", "#AAAAAA"),
                label=obj.get("label", ""),
            )
        )
    return shapes


# ---------------------------------------------------------------------------
# Stage 4: Model assembly
# ---------------------------------------------------------------------------

def _shape_extents(shape: Shape) -> tuple[int, int, int]:
    """Return (width_x, height_y, depth_z) occupied studs/plates for a shape."""
    d = shape.dimensions
    if isinstance(d, (CuboidDimensions, WedgeDimensions)):
        return d.width, d.height, d.depth
    if isinstance(d, CylinderDimensions):
        return d.diameter, d.height, d.depth_studs if d.depth_studs is not None else d.diameter
    raise TypeError(f"Unknown dimensions type: {type(d)}")


def build_model(image_filename: str, shapes: list[Shape]) -> LegoModel:
    """Wrap shapes in a LegoModel with a tight bounding box."""
    if not shapes:
        bbox = BoundingBox(x=0, y=0, z=0, width=0, height=0, depth=0)
    else:
        x_starts, y_starts, z_starts = [], [], []
        x_ends,   y_ends,   z_ends   = [], [], []
        for s in shapes:
            ex, ey, ez = _shape_extents(s)
            x_starts.append(s.position.x);       x_ends.append(s.position.x + ex)
            y_starts.append(s.position.y);       y_ends.append(s.position.y + ey)
            z_starts.append(s.position.z);       z_ends.append(s.position.z + ez)
        bbox = BoundingBox(
            x=min(x_starts), y=min(y_starts), z=min(z_starts),
            width=max(x_ends)  - min(x_starts),
            height=max(y_ends) - min(y_starts),
            depth=max(z_ends)  - min(z_starts),
        )

    return LegoModel(
        source_image=image_filename,
        bounding_box=bbox,
        shapes=shapes,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(image_path: str) -> str:
    """Full pipeline. Returns a JSON string."""
    image_data = load_image(image_path)
    scene = interpret_image(image_data)
    shapes = decompose_into_primitives(scene)
    model = build_model(image_data["filename"], shapes)
    return model.model_dump_json(indent=2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <image_path>")
        sys.exit(1)
    print(run(sys.argv[1]))
