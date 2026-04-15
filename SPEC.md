# lego-ai-builder Spec

## Overview

Text or image description -> 9-stage pipeline (LLM + Code) -> voxel grid -> 3D render.

Server: FastAPI on port 8000 (`uvicorn server:app --reload --port 8000`)
Frontend: Three.js 0.161.0, served from `static/`

---

## Architecture

```
Input (text/image) ─► Semantic + Intent (LLM) ─► Cleanup (Code) ─► Voxelize (Code)
                                                       ▲                    │
                                                       │                    ▼
                                              Revise (Code) ◄─ Critique (LLM) ◄─ Validate (Code)
                                                       │
                                                       ▼
                                              User Feedback ─► Finalize
```

### Role Split

| Layer | Responsibility |
|---|---|
| **LLM** | Meaning, structure, intent, critique, user feedback translation |
| **Code** | Geometry, constraints, enforcement, voxelization, validation |

### Pipeline Stages

| Stage | Name | Execution | Purpose |
|---|---|---|---|
| 0 | Input | System | Collect raw input, package into bundle |
| 1 | Semantic + Intent Generation | LLM | Understand object, define parts + metadata tags |
| 2 | Structural Cleanup / Regularization | Code | Normalize, rotate, enforce symmetry, scale, place |
| 3 | Geometry Realization + Validation | Code | Voxelize, resolve ownership, render projections, detect/repair hard failures (merges, floaters, thin loss) |
| 4 | *(reserved)* | | |
| 5 | Semantic Validation | LLM | Perceptual critique, suggest correction intents |
| 6 | Bounded Revision | Code | Execute intents via deterministic operators |
| 7 | User Feedback | User + LLM + Code | Human-in-loop refinement |
| 8 | Finalization | Code | Lock/export approved state |

### Loops

**Auto loop** (max 1 revision iteration): Initial pass through 2 → 3 is not counted. After Gemini validates (5) and proposes revisions (6), the pipeline re-runs 2 → 3 **once**. No further auto-revision after that. Pipeline tracks a `revision_done` flag — set after Stage 6 completes, checked before Stage 5 invokes Gemini.
- Exit early if: no meaningful improvement, changes below threshold, risk > benefit

**User loop**: 7 → 5 (feedback mode) → 6 → rebuild (2→3) → render → user sees result
- Single LLM pass per feedback round, no additional auto-revision
- User can submit more feedback (next iteration) or accept the result
- Re-enter Stage 1 only if object identity changes (e.g., "make it a car instead of a spider")

**Group metadata persists through loops.** Stage 1 defines groups; Stage 2 expands them into individual instances. Group membership annotations (which group, source vs mirror, index) are preserved on instances through Stage 5/6 edits and user loop iterations. Stage 5/6 edits target individual instances — group-wide symmetric edits are not enforced automatically (future enhancement). But group metadata surviving means a subsequent pass can still reason about group structure.

### Core Flow

Meaning → Intent → Regularize → Realize → Measure → Critique → Controlled Revision → User → Lock

---

## Stage 0 — Input Acquisition

**Purpose:** Collect the raw user-provided input and package it into a normalized bundle. No interpretation, no preprocessing.

### InputBundle

| Field | Type | Description |
|---|---|---|
| `image` | `bytes \| None` | Raw image data, untouched |
| `image_mime` | `str \| None` | MIME type: `"image/jpeg"`, `"image/png"`, or `"image/webp"` |
| `image_ext` | `str \| None` | Original file extension: `"jpg"`, `"jpeg"`, `"png"`, `"webp"` |
| `text` | `str \| None` | User text description, untouched |
| `modality` | `str` | `"image"` \| `"text"` \| `"both"` — derived from what is present |
| `grid_size` | `tuple[int,int,int]` | System default `(100, 100, 100)` — width, depth, height |

### Validation Rules

- At least one of `image` or `text` must be non-null
- If `image` is not None: `image_mime` must be non-null and in `{"image/jpeg", "image/png", "image/webp"}`
- If `image` is not None: `image_ext` must be non-null and in `{"jpg", "jpeg", "png", "webp"}`
- If `text` is not None: `text.strip()` must be non-empty
- `modality` is derived, not user-set: `"both"` if both present, else whichever is present
- `grid_size` defaults to `(100, 100, 100)` — may become configurable later

### MIME / Extension Mapping

| Extension | MIME Type |
|---|---|
| `jpg`, `jpeg` | `image/jpeg` |
| `png` | `image/png` |
| `webp` | `image/webp` |

### What Stage 0 does NOT do

- Object recognition or semantic interpretation
- Geometry generation
- Conflict resolution between image and text
- Image preprocessing (background removal, cropping, segmentation)
- If preprocessing is needed later, it becomes a distinct stage between 0 and 1

---

## Stage 1 — Semantic + Intent Generation

**Purpose:** Interpret the input into a structured, semantically meaningful representation that downstream deterministic code can reliably process. Stage 1 is a "controlled interpretation layer" — it describes what the object is and how its parts relate, not exact geometry.

**Execution:** LLM (single call, taxonomy/world knowledge is implicit)

**Input:** `InputBundle` from Stage 0

### Design Philosophy

- The LLM interprets the input into structured signals; deterministic code enforces geometry
- Prefer clarity over precision — no exact coordinates, angles, or voxel placements
- Prefer consistent structure over exhaustive detail
- No collision resolution, spacing, or symmetry enforcement — that belongs to Stage 2
- The LLM brings its own world knowledge (e.g., mammals have 4 legs) — no taxonomy table in the prompt
- The prompt defines the output schema and valid enums; the LLM fills them in

### Output: IntentModel

#### 1. Entity

| Field | Type | Description |
|---|---|---|
| `entity` | `str` | Normalized object name — the LLM's natural language label (e.g., "tarantula", "fire truck"). Used for LLM reasoning and validation, not for driving geometry logic |

#### 2. Category

| Field | Type | Valid Values |
|---|---|---|
| `category` | `str` | `animal`, `plant`, `building`, `vehicle`, `robot`, `humanoid`, `artifact`, `natural_feature`, `other` |

Category is for semantic grouping, reporting, and validation support. Deterministic geometry logic is driven by resolved structural attributes, not category.

#### 3. Parts List

A list of major structural components. Each part represents a meaningful piece of the object — not overly granular, not overly abstract.

| Field | Type | Description |
|---|---|---|
| `part_name` | `str` | Human-readable name, **must be unique** across all parts (e.g., "body", "left_front_leg_1", "rear_wheel_left") |
| `primitive_type` | `str` | `cuboid`, `cylinder`, `ellipsoid`, `cone_frustum` |
| `dimensions` | `object` | Primitive-specific dimensions in root-normalized object space (see below) |
| `parent` | `str \| null` | `part_name` of parent, or null for root. Must reference an existing unique part_name |
| `critical` | `bool` | `true` = essential for recognition, `false` = decorative/secondary |
| `color_id` | `str` | Exact color from the palette (see Color Palette section) |
| `attachment` | `Attachment` | Where this part attaches to its parent (see below) |
| `orientation` | `str` | Direction the child extends from the attachment point, in parent's local frame (see below) |
| `group` | `str \| null` | Group name if part of a repeated set (e.g., "left_legs", "wheels") |
| `support_role` | `str` | `primary_support`, `secondary_support`, `non_support` (see section 7) |
| `ground_contact` | `str` | `required`, `optional`, `none` (see section 7) |

##### Dimensions

All dimensions are in root-normalized object space (root's largest dimension = 100). Unified fields for all primitives:

```json
{ "width": float, "depth": float, "height": float }
```

All values are **full extents** (not radii or half-extents). Interpretation per primitive:

| Primitive | `width` (X) | `depth` (Y) | `height` (Z) | Extra |
|---|---|---|---|---|
| `cuboid` | X extent | Y extent | Z extent | — |
| `cylinder` | base diameter | base diameter | Z extent | `width` ≈ `depth` |
| `ellipsoid` | X diameter | Y diameter | Z diameter | — |
| `cone_frustum` | base diameter | base diameter | Z extent | `top_radius` on part |

The LLM outputs explicit numeric values — no coarse labels or parent-relative ratios. Stage 2 receives these directly.

#### 4. Attachment

Attachment is defined by **two anchor points** — one on the parent, one on the child. Each anchor specifies a face and a region on that face, using the respective primitive's local coordinate system.

| Field | Type | Description |
|---|---|---|
| `parent_face` | `str` | Which face of the parent: `top`, `bottom`, `front`, `back`, `left`, `right` |
| `parent_region` | `str` | Where on that face (see valid regions below) |
| `child_face` | `str` | Which face of the child: `top`, `bottom`, `front`, `back`, `left`, `right` |
| `child_region` | `str` | Where on that face (see valid regions below) |

Stage 2 code aligns the child's specified face to the parent's specified face and matches the anchor points. This fully determines position and orientation — no separate orientation field needed.

##### Examples

A leg hanging from a body:
```
parent_face: "bottom",  parent_region: "front_left"
child_face:  "top",     child_region:  "center"
```

A wheel on the side of a car:
```
parent_face: "left",    parent_region: "bottom_front"
child_face:  "right",   child_region:  "center"
```

Eyes on the front of a head:
```
parent_face: "front",   parent_region: "top_left"
child_face:  "back",    child_region:  "center"
```

##### Local Coordinate System (for both parent and child)

- +X = right, -X = left
- +Y = front, -Y = back
- +Z = top, -Z = bottom

Each primitive uses its own local frame with this convention. The face/region labels reference that frame.

##### Valid Regions Per Face

Every face has 9 valid regions: 4 edges, 4 corners, 1 center. Region labels use the primitive's local axes for the two dimensions that span the selected face.

| Face | Edges | Corners | Center |
|---|---|---|---|
| `top` / `bottom` | `front`, `back`, `left`, `right` | `front_left`, `front_right`, `back_left`, `back_right` | `center` |
| `front` / `back` | `top`, `bottom`, `left`, `right` | `top_left`, `top_right`, `bottom_left`, `bottom_right` | `center` |
| `left` / `right` | `top`, `bottom`, `front`, `back` | `top_front`, `top_back`, `bottom_front`, `bottom_back` | `center` |

The face determines which 2D surface we're on. The region picks a position on that surface. Two-part key — face narrows to a surface, region picks a spot.

##### Non-Cuboid Attachment Resolution

For non-cuboid primitives (cylinder, ellipsoid, cone frustum), attachment semantics are interpreted using the primitive's **axis-aligned bounding box** as the reference frame:

1. The face + region selects a point on the bounding box (same rules as cuboid)
2. That point is projected **inward** toward the primitive's center or central axis
3. The first intersection with the actual primitive surface becomes the true attachment point

This ensures all primitives share the same face/region vocabulary while anchoring to geometrically meaningful surface points. For example, "left, center" on a cylinder resolves to a point on the curved surface, not floating at the bounding box edge.

#### 5. Orientation

Orientation defines the direction of the **child's local +Z axis** in the **parent's local coordinate system**. Each label maps to a direction vector composed of parent axes.

| Field | Type | Description |
|---|---|---|
| `orientation` | `str` | Direction of child's +Z axis, expressed in parent's local frame |

##### Valid Values

**Single-axis** (child +Z aligns with one parent axis):

| Label | Direction Vector |
|---|---|
| `top` | +Z |
| `bottom` | -Z |
| `front` | +Y |
| `back` | -Y |
| `left` | -X |
| `right` | +X |

**Two-axis compounds** (child +Z points between two parent axes, normalized):

| Label | Direction Vector |
|---|---|
| `top_front` | +Z +Y |
| `top_back` | +Z -Y |
| `top_left` | +Z -X |
| `top_right` | +Z +X |
| `bottom_front` | -Z +Y |
| `bottom_back` | -Z -Y |
| `bottom_left` | -Z -X |
| `bottom_right` | -Z +X |
| `front_left` | +Y -X |
| `front_right` | +Y +X |
| `back_left` | -Y -X |
| `back_right` | -Y +X |

**Three-axis compounds** (child +Z points between three parent axes, normalized):

`top_front_left`, `top_front_right`, `top_back_left`, `top_back_right`,
`bottom_front_left`, `bottom_front_right`, `bottom_back_left`, `bottom_back_right`

Total: 6 + 12 + 8 = 26 possible directions.

##### How Stage 2 Uses This

1. Map the label to a direction vector (e.g., `top_left` → `(-1, 0, 1)`)
2. Normalize the vector
3. Rotate the child so its local +Z axis aligns with this vector
4. Keep the attachment point fixed

Stage 2 may further refine angles based on structural constraints (spacing, collision avoidance, etc.).

##### Examples

A leg extending straight down from a body:
```
orientation: "bottom"          → child +Z points toward parent -Z
```

A spider leg angling down, outward, and forward:
```
orientation: "bottom_left_front"  → child +Z points toward (-X, +Y, -Z) normalized
```

A tail angling up and backward:
```
orientation: "top_back"        → child +Z points toward (+Z, -Y) normalized
```

#### 6. Structural Grouping / Repetition

Parts that belong to repeated sets share a `group` name. Attachment and orientation define where a single part connects and how it points. They do NOT define how multiple parts in a group are spaced. Distribution is a separate group-level concept.

The LLM should NOT generate explicit coordinates or manually place each member. It describes the distribution pattern; deterministic code computes exact anchor points.

##### Per-Part Fields

| Field | Type | Description |
|---|---|---|
| `group` | `str \| null` | Group name (on each part). Null if not part of a group |

##### Group Definition

| Field | Type | Description |
|---|---|---|
| `group_name` | `str` | Matches the `group` field on member parts |
| `count` | `int` | Number of parts in this group |
| `distribution` | `Distribution \| null` | How members are spatially arranged. Null if this is a derived (mirror) group |
| `mirror_of` | `str \| null` | Group name this is a mirror of. Null for source groups |
| `mirror_axis` | `str \| null` | `x`, `y`, or `z`. Override for mirror axis. If null, inferred from source group's attachment face |

**Source vs derived groups:**
- **Source group** — fully specified: distribution, count, etc.
- **Derived group** — references source via `mirror_of`, inherits distribution + count, code mirrors positions
- Derived groups may override fields if asymmetry is needed (e.g., different count on one side)

##### Distribution

A composable set of parameters that Stage 2 code uses to compute individual anchor points for each group member.

| Field | Type | Valid Values | Description |
|---|---|---|---|
| `domain` | `str` | `edge`, `face`, `radial` | Where the group lives relative to parent |
| `pattern` | `str` | `line`, `band`, `grid`, `arc`, `ring` | How members are arranged |
| `axis` | `str \| null` | `x`, `y`, `z` | Direction of distribution in parent's local frame. Required for `line`, `band`. Must be valid for the selected face |
| `spacing` | `str` | `uniform_full`, `uniform_inset`, `center_cluster` | How parts are spaced within the domain |
| `spread` | `str` | `narrow`, `medium`, `wide` | How much of the domain is occupied |
| `orientation_mode` | `str` | `parallel`, `radial` | How instances are oriented relative to attachment position (default: `parallel`) |

##### Domain

| Value | Meaning |
|---|---|
| `edge` | Along an edge of a face |
| `face` | Across a face (2D area) |
| `radial` | Around a central point on a face |

##### Pattern

| Value | Meaning |
|---|---|
| `line` | Single axis arrangement |
| `band` | Strip across a face |
| `grid` | 2D array |
| `arc` | Partial circular arrangement |
| `ring` | Full circular arrangement |

##### Spacing

| Value | Meaning |
|---|---|
| `uniform_full` | Evenly distributed edge-to-edge or face-wide |
| `uniform_inset` | Evenly distributed with margin at boundaries |
| `center_cluster` | Grouped tightly around center |

##### Spread

| Value | Meaning |
|---|---|
| `narrow` | Occupies a small portion of the domain |
| `medium` | Occupies a moderate portion |
| `wide` | Occupies most of the domain |

##### Orientation Mode

| Value | Meaning | Use case |
|---|---|---|
| `parallel` | All instances share the part type's base rotation. No per-instance rotation adjustment. | Wheels, windows, fence posts — parts that should all face the same direction |
| `radial` | Each instance is rotated around the parent face's normal axis based on its offset position. The angle is `atan2(attachment_offset_v, attachment_offset)` applied on top of the base rotation. | Legs, spokes, petals, arms — parts that should splay outward from the attachment center |

**Radial computation (Stage 2):**
1. For each instance with offset `(u, v)` on the parent face, compute angle `θ = atan2(v, u)`
2. Determine the face normal axis (e.g., bottom face normal = -Z, so rotation is around Z)
3. Add `θ` to the appropriate rotation axis of the part type's base rotation
4. Result: each instance points away from the face center toward its offset position

| Parent face | Normal axis | Radial rotation applied to |
|---|---|---|
| `top` / `bottom` | Z | `rz` |
| `front` / `back` | Y | `ry` |
| `left` / `right` | X | `rx` |

##### Key Relationships

- **Attachment** on member parts defines the nominal location of the group (face + region)
- **Distribution** defines how multiple members occupy that space around the attachment
- **Orientation** remains per-part and defines direction of extension, not position within the group

##### Examples

Spider legs (4 on left side):
```
group_name: "left_legs"
count: 4
distribution:
  domain: "face"
  pattern: "line"
  axis: "y"
  spacing: "uniform_inset"
  spread: "wide"
```

Windows on a building front:
```
group_name: "front_windows"
count: 6
distribution:
  domain: "face"
  pattern: "grid"
  axis: null
  spacing: "uniform_full"
  spread: "wide"
```

Flower petals:
```
group_name: "petals"
count: 5
distribution:
  domain: "radial"
  pattern: "ring"
  axis: null
  spacing: "uniform_full"
  spread: "medium"
```

Spider with mirrored legs:
```
groups:
  - group_name: "left_legs"
    count: 4
    mirror_of: null
    mirror_axis: null
    distribution:
      domain: "face"
      pattern: "line"
      axis: "y"
      spacing: "uniform_inset"
      spread: "wide"

  - group_name: "right_legs"
    count: 4
    mirror_of: "left_legs"
    mirror_axis: null            ← inferred as X (source attaches to left face)
    distribution: null           ← inherited + mirrored from left_legs
```

##### Group Ordering

Groups have an inherent order. Members are indexed 0 → count-1.

| Pattern | Default Order |
|---|---|
| `line` / `band` | Positive axis direction: `x` = left→right, `y` = back→front, `z` = bottom→top |
| `grid` | Row-major: primary axis first, then secondary |
| `ring` / `arc` | Counterclockwise when viewed from the positive normal of the face |
| `stack` | Bottom→top (+Z) |

##### Mirroring

Derived groups (via `mirror_of`) inherit distribution from the source and mirror positions across an axis.

**Mirror axis inference** (when `mirror_axis` is null):

| Source attachment face | Inferred mirror axis |
|---|---|
| `left` or `right` | `x` |
| `front` or `back` | `y` |
| `top` or `bottom` | `z` |

**Index ordering in mirror groups:**
- Mirror groups preserve index order by default (index 0 maps to index 0)
- This works when the mirror axis differs from the ordering axis (common case: mirror X, order along Y)
- If mirroring occurs along the same axis as ordering, code applies index reversal automatically — the LLM does not need to handle this

**Design benefits:**
- LLM only defines one side explicitly — reduced burden, fewer tokens
- Guaranteed symmetry — modify source, mirror regenerates automatically
- Controlled exceptions — LLM can override fields on derived groups for intentional asymmetry

#### 7. Support, Grounding, and Elevation

Defines how parts interact with the ground and with each other in terms of support and elevation. These are structural intent signals, not physics simulation.

Attachment and orientation define relationships between parts. Support and grounding define how the structure relates to the ground and which parts bear weight.

##### Part-Level Signals

| Field | Type | Valid Values | Description |
|---|---|---|---|
| `support_role` | `str` | `primary_support`, `secondary_support`, `non_support` | Whether this part contributes to supporting the object |
| `ground_contact` | `str` | `required`, `optional`, `none` | Whether this part should touch the ground plane (Z=0) |

**Support role:**

| Value | Meaning |
|---|---|
| `primary_support` | Main load-bearing element (legs, wheels, base) |
| `secondary_support` | Contributes to support but not critical |
| `non_support` | Does not support the structure (body, wings, decorations) |

**Ground contact:**

| Value | Meaning |
|---|---|
| `required` | Must touch ground (feet, wheels, base) |
| `optional` | May touch ground but not required |
| `none` | Should not touch ground (body, wings, head) |

Groups often share support roles (e.g., all legs are `primary_support` + `required` contact). The LLM assigns per part; Stage 2 code may enforce consistency across groups.

##### Object-Level Elevation

| Field | Type | Valid Values | Description |
|---|---|---|---|
| `elevation` | `str` | `elevated_above_support`, `directly_grounded`, `mixed` | How the main mass relates to support parts |

| Value | Meaning | Examples |
|---|---|---|
| `elevated_above_support` | Main body sits above support parts | Animals, vehicles, chairs, tables |
| `directly_grounded` | Main mass rests on ground/base | Rocks, buildings, pyramids |
| `mixed` | Partial contact with ground and supports | Trees (trunk grounded, canopy elevated) |

##### How Stage 2 Uses This

- Snap `required` ground contact parts to Z=0
- Maintain elevation relationships (body above legs)
- Ensure `none` ground contact parts do not intersect ground unintentionally
- Detect and correct floating or collapsed structures
- Maintain connectivity between support and supported parts

##### Examples

Spider: legs = `primary_support` / `required`, body = `non_support` / `none`, elevation = `elevated_above_support`

Car: wheels = `primary_support` / `required`, chassis = `non_support` / `none`, elevation = `elevated_above_support`

Rock: base = `primary_support` / `required`, rest = `non_support` / `none`, elevation = `directly_grounded`

##### Constraints

- Do NOT specify exact heights, distances, or forces
- Do NOT simulate physics or stability
- Support signals guide geometry cleanup and validation, not initial placement
- These work together with attachment/orientation, not as overrides

#### 8. High-Level Structural Attributes

To be defined separately. These are resolved signals (symmetry, repetition patterns, support structure, posture, etc.) that drive deterministic logic in Stage 2. The LLM fills these in based on its own world knowledge and observation of the input.

### Primitive Local Coordinate System (Canonical Convention)

All primitives are defined in their own local coordinate system before any rotation or placement.

**General axes:**
- +X / -X = right / left
- +Y / -Y = front / back
- +Z / -Z = top / bottom

#### Cuboid
- Dimensions along X, Y, Z
- Longest dimension aligned with X, second longest with Y, shortest with Z (canonical normalization)
- 6 faces: +Z (top), -Z (bottom), +X (right), -X (left), +Y (front), -Y (back)
- Center at geometric center

#### Cylinder
- Axis aligned along Z
- +Z = top circular face, -Z = bottom circular face
- Side surface wraps around X-Y plane

#### Ellipsoid
- Semi-axes along X, Y, Z
- Same face conventions as cuboid
- Center at geometric center

#### Cone Frustum (Truncated Cone)
- Axis aligned along Z
- -Z = bottom (larger base), +Z = top (smaller face)

**Important:** Canonical orientation defines a stable local reference frame for attachment semantics. It does NOT constrain final orientation in the scene — after rotation, a cuboid's longest axis may point along any direction.

### What Stage 1 does NOT do

- Produce exact coordinates, angles, or voxel placements
- Resolve collisions, spacing, or symmetry precisely
- Enforce geometric constraints
- Select from a taxonomy table (LLM uses implicit world knowledge)
- Compute scale or grid placement

Stage 1 output is the foundation for all downstream geometry, validation, and refinement.

---

## Stage 2 — Structural Cleanup / Regularization

**Purpose:** Convert Stage 1's semantic intent model into concrete geometry that fits within the voxel grid. All conversion from semantic signals to numeric values happens here.

**Execution:** Code (deterministic)

**Input:** `IntentModel` from Stage 1

**Output:** Cleaned geometry with world-space positions, dimensions, and rotations — ready for voxelization.

### Design Philosophy

- Stage 1 defines intent (structure, relationships, direction)
- Stage 2 enforces geometry (placement, spacing, elevation, connectivity)
- Prefer deterministic defaults where semantic detail is insufficient
- No LLM calls — all conversions are rule-based
- Stage 2 uses a **working occupancy grid** for collision scoring — this is a temporary sampling of primitive geometry, not the definitive voxelization. It is discarded after Stage 2 completes. Stage 3 performs the definitive voxelization for rendering, validation, and downstream stages.

#### Known Limitation: Single-Parent Tree Only (v1)

The pipeline assumes a strict single-parent tree. Each part has exactly one `parent_part`; transforms propagate downward through unique parentage.

**Closed structures** (e.g., 4 pillars supporting a shared roof) cannot be represented natively — the roof must be parented to one pillar, with other pillar-roof contacts being geometric only (voxels touch, no transform edge). If an edit moves the parent pillar, the roof moves with it; other contacts break visually and are caught by the Stage 5 validation loop.

**v1 scope:** Objects that don't fit a single-parent tree are out of scope. The Stage 1 prompt should hint Gemini to pick the most structurally central parent when a part contacts multiple siblings. Part groups (collapsing closed sub-structures into atomic nodes) are a future enhancement.

### Sub-Stages

| Sub-stage | Purpose |
|---|---|
| 2.1 | Graph integrity + hierarchy preprocessing |
| 2.2 | Semantic → numeric conversion (regions, spread fractions) |
| 2.3 | Attachment resolution + group/spread instantiation |
| 2.4 | Orientation application + initial transform assembly |
| 2.5 | Support shaping + elevation + grounding |
| 2.6 | Collision detection + resolution (using temporary working occupancy) |

### Root Normalization

Stage 1 outputs all dimensions in root-normalized object space, where the root part's largest primitive-specific axis-aligned dimension = **100 units**.

The normalization dimension depends on primitive type:

| Primitive | Largest dimension = max of |
|---|---|
| Cuboid | `length_x`, `length_y`, `length_z` |
| Cylinder | `height`, `2 × radius` |
| Ellipsoid | `axis_x`, `axis_y`, `axis_z` |
| Cone Frustum | `height`, `2 × base_radius`, `2 × top_radius` |

**Constraints:**
- Do NOT use diagonal lengths
- Do NOT mix radius with full-length dimensions
- Always compare like-for-like linear extents

### Primitive Dimension Conventions

All dimensions are full lengths unless explicitly noted as radius.

| Primitive | Dimensions |
|---|---|
| Cuboid | `length_x`, `length_y`, `length_z` (full extents) |
| Cylinder | `height` (full extent), `radius` |
| Ellipsoid | `axis_x`, `axis_y`, `axis_z` (full axis lengths, not radii) |
| Cone Frustum | `height` (full extent), `base_radius`, `top_radius` |

All parts are defined directly in this normalized object-space unit system. Stage 2 receives explicit numeric dimensions — no size conversion is needed.

### 2.1 — Graph Integrity + Hierarchy Preprocessing

- Validate parent-child graph: fix dangling refs, break cycles
- Enforce single root (prefer critical + largest volume)
- Promote ancestors of critical (`essential`) parts to critical
- Compute hierarchy metrics: downstream dependent count and depth for each node

This runs first because all subsequent stages depend on a valid tree.

### 2.2 — Semantic → Numeric Conversion

Dimensions are now provided directly by Stage 1 in root-normalized object space — no size or aspect conversion is needed. This stage converts the remaining semantic signals to numeric values.

#### Region → Coordinate Mapping

Regions map to normalized (u, v) coordinates on the face, where (0,0) is one corner and (1,1) is the opposite.

| Region type | Coordinate rule |
|---|---|
| `center` | (0.5, 0.5) |
| Edge (e.g., `top`) | Midpoint of that edge (e.g., (0.5, 1.0)) |
| Corner (e.g., `top_left`) | Exact corner (e.g., (0.0, 1.0)) |

The two face-spanning axes determine which axis maps to u and v:

| Face | u axis | v axis |
|---|---|---|
| `top` / `bottom` | X (left→right, 0→1) | Y (back→front, 0→1) |
| `front` / `back` | X (left→right, 0→1) | Z (bottom→top, 0→1) |
| `left` / `right` | Y (back→front, 0→1) | Z (bottom→top, 0→1) |

#### Spread → Domain Fraction

`spread` maps to how much of the face/edge the group occupies.

| Value | Fraction of domain |
|---|---|
| `narrow` | ~0.3 |
| `medium` | ~0.6 |
| `wide` | ~0.9 |

### 2.3 — Attachment Resolution + Group/Spread Instantiation

#### Group/Spread Instantiation

For each group:
1. Read distribution parameters (domain, pattern, axis, spacing, spread)
2. Compute anchor points for each member based on pattern and count
3. For mirror groups: inherit source distribution, mirror across inferred or explicit axis
4. Assign parts to positions by group index order
5. Feed computed per-child positions into attachment resolution below

Group spread produces per-child offsets that become the `resolved_offset` for each member. After this step, each child has its own unique attachment location.

#### Two-Level Attachment Model

**A. Semantic attachment (from Stage 1)**

Stage 1 outputs four fields per part:
- `parent_face`: which face of the parent (`top`, `bottom`, `front`, `back`, `left`, `right`)
- `parent_region`: where on that face (9 valid regions: `center`, 4 edges, 4 corners)
- `child_face`: which face of the child
- `child_region`: where on that face (same 9 regions)

This is coarse intent — not yet a resolved 3D anchor.

**B. Region dimensionality**

The region determines the degrees of freedom for offset resolution:

| Region type | Examples | Dimensionality | Offset shape |
|---|---|---|---|
| center | `center` | 2D | Free within face |
| edge | `top`, `left`, `front`, etc. | 1D | Along edge axis |
| corner | `top_left`, `bottom_front`, etc. | 0D | Fixed point |

Stage 2 derives dimensionality from the region — no separate `attachment_type` field is stored.

**C. Resolved attachment (computed in Stage 2)**

For each child, Stage 2 computes and stores:
- `parent_anchor`: 3D point on the parent primitive surface
- `child_anchor`: 3D point on the child primitive surface
- `resolved_offset`: offset within the region's domain (see below)

Resolution steps:
1. Compute `parent_anchor` from `parent_face` + `parent_region`
2. Compute `child_anchor` from `child_face` + `child_region`
3. Align `child_anchor` to `parent_anchor` (child moves to parent)
4. Compute resulting child transform

Gemini defines coarse attachment and group spread only. Stage 2 resolves exact per-child offsets and anchors on BOTH sides. After spread resolution, different children in the same group no longer share the same attachment point.

#### Resolved Offset Definitions

Offsets are defined by the region's dimensionality (derived from `parent_region` or `child_region`).

**Center region (2D — face attachment)**

Two offsets in the face plane, following the face's axis mapping:

| Face | Offset axes |
|---|---|
| `top` / `bottom` | (x, y) |
| `front` / `back` | (x, z) |
| `left` / `right` | (y, z) |

Offsets are normalized: (0,0) = face center, ±1 = face edges.

**Edge region (1D — edge attachment)**

One scalar offset along the edge axis.

Offsets follow positive axis direction:
- X edge → left to right
- Y edge → back to front
- Z edge → bottom to top

Offset is normalized: 0 = edge midpoint, ±1 = edge endpoints.

**Corner region (0D — corner attachment)**

No translational offset. Corner is a single point. Only orientation / rotation matters after that.

#### Anchor Resolution to 3D

For each part, Stage 2 computes two 3D anchor points: one on the parent surface, one on the child surface.

**Parent-side anchor**

Map `parent_face` + `parent_region` to a 3D point on the parent primitive.

Cuboid:
- Direct mapping from face + region coordinates to 3D
- Literal face rectangle

Cylinder / Frustum / Cone / Ellipsoid:
- Use a unified abstraction:
  1. Choose a point on the bounding-box surface patch (same face/region rules as cuboid)
  2. Project inward to the actual primitive surface

Projection direction:
- cylinder / frustum / cone → inward toward central axis
- ellipsoid → inward toward center

**Child-side anchor**

Same computation on the child primitive using `child_face` + `child_region`. The child-side anchor determines the contact point on the child.

**Alignment**

Once both anchors are computed:
1. Rotate the child so that `child_face` normal opposes `parent_face` normal (faces point toward each other)
2. Translate the child so `child_anchor` coincides with `parent_anchor`
3. Store the resulting transform

This fully determines the child's position and orientation relative to the parent. The `orientation` field (section 5 of Stage 1) may further adjust the child's +Z direction after initial face alignment.

#### Attachment Validity Rule

Parent-child overlap is valid only if it occurs at the intended attachment interface.

Overlap between two primitives is considered VALID if the resolved `parent_anchor`:
- lies within the overlap volume, OR
- is within 2 working-grid cells of the overlap volume

This tolerance handles intentional embedding/insertion, sampling artifacts in the working occupancy grid, and minor anchor discretization mismatch.

All other parent-child overlap is INVALID. Sibling, cousin, and other non-parent-child overlap is always invalid.

Implications:
- Collision detection must compute overlap volume explicitly
- Attachment validity is determined by a point-in-overlap test with tolerance

#### Local 2D Movement Spaces

Movement within an attachment domain is represented in a local 2D control space, then converted back to 3D. The 2D space is only a control space for movement and budgeting. Collision checks still happen in rebuilt 3D / global geometry.

By primitive / domain:

| Primitive / Domain | Control patch shape |
|---|---|
| cuboid face | rectangular patch |
| cylinder side | rectangular patch on bounding-box face, projected inward to cylinder |
| frustum / cone side | tapered patch on bounding pyramid side, projected inward |
| ellipsoid | bounding-box face patch, projected inward to center |
| cylinder / frustum caps | disk-like domains if needed |

For frustum / cone, the side patch narrows toward the tip. For ellipsoid, an ellipse-like effective domain results after projection, but bounding-face patch + inward projection is the controlling abstraction.

Movement happens in the parent's local 2D control patch. Every candidate point must be converted back to an exact 3D surface point. Overlap is never evaluated only in 2D.

#### Allowable Movement Domains

Movement must preserve attachment. The region dimensionality (derived from `parent_region`) determines the movement domain.

Primary rule: prefer sliding within the attachment domain. Use rotation only when sliding is unavailable or insufficient.

**By region type:**

| Region type | Primary movement | Fallback |
|---|---|---|
| Corner (0D) | None | Rotation only |
| Edge (1D) | Slide along edge axis | Rotate around edge anchor / edge axis |
| Center on flat face (2D) | Slide within face plane (no movement along normal) | — |
| Center on curved face (2D) | Slide along surface via 2D control patch + projection | Rotation if sliding fails |

Disallowed:
- Breaking contact
- Leaving the movement domain
- Pushing deeper into the parent

Movement limits are governed by domain boundaries.

**Symmetry during repair:**
- First try group-preserving symmetric movement
- If unresolved, allow single-member movement as fallback

### 2.4 — Orientation Application + Initial Transform Assembly

1. Map orientation label to direction vector (see Stage 1 section 5)
2. Normalize the vector
3. After face alignment from 2.3 (child_face opposes parent_face), apply additional rotation so the child's local +Z aligns with the orientation vector
4. Attachment point remains the pivot — it stays fixed
5. Rotations are hierarchical — children inherit all ancestor transforms
6. Application order: Z → Y → X
7. Store the composed transform (face alignment + orientation) as the initial world transform for each part

### 2.5 — Support Shaping + Elevation + Grounding

#### Grounding

- Parts with `ground_contact: "required"` → snap lowest point to Z=0
- Parts with `ground_contact: "none"` → must not intersect Z=0 plane

#### Elevation

| Value | Rule |
|---|---|
| `elevated_above_support` | Body elevation = height of support parts (derived from support geometry after shaping) |
| `directly_grounded` | Root/base part snapped to Z=0 |
| `mixed` | Apply per-part ground contact rules individually |

#### Support Shaping

The LLM defines structure (parts + hierarchy). Code determines shaping behavior based on that structure.

**Case 1 — Unsegmented support parts** (single primitive, e.g., one cuboid leg):
- Treat as straight support
- Apply default incline based on orientation
- Derive body elevation from support length + ground contact constraint
- Do NOT infer joints or articulation

**Case 2 — Segmented support parts** (multi-primitive chain inferred from hierarchy, e.g., upper_leg → lower_leg → foot):
- Treat as articulated support
- Apply default articulation profile:
  - Proximal segments → establish elevation and outward spread
  - Distal segment(s) → descend to ground contact
- Exact angles derived to satisfy:
  - Ground contact constraint
  - Connectivity between segments
  - Collision avoidance
- No explicit joint mechanics from LLM required

### 2.6 — Collision Detection + Resolution

#### Working Occupancy Grid

Stage 2 does NOT perform definitive voxelization. Instead, it samples primitive geometry into a temporary working occupancy grid used only for:
- Overlap volume estimates
- Overlap extents along x/y/z
- Global overlap checks (coarse side-effect monitoring)
- Local collision scoring during candidate movement

**Resolution:** 1 working grid cell = 1 normalized unit (root largest dimension = 100). The grid is dynamically sized to the bounding box of the assembled geometry — it is NOT a fixed 100³ cube.

This grid is derived from the current primitive geometry and transformed subtree state, and is recomputed as needed during collision resolution. It is discarded when Stage 2 completes.

References to "voxels" in collision resolution (probe ±1 voxel, 1 voxel steps, etc.) refer to cells in this working grid, not the final output grid. Stage 2 object-space reasoning and Stage 3 final scale-to-fit are separate concerns — do not collapse the normalized scale to match the final grid dimensions.

#### Default Constants (Tunable)

These are initial defaults and may be tuned based on empirical behavior.

| Constant | Value | Description |
|---|---|---|
| Attachment validity tolerance | 2 working grid cells | Max distance from `parent_anchor` to overlap volume for valid classification |
| Acceptable overlap threshold | 5% of smaller primitive volume | Target for invalid overlap reduction — not zero |
| Allowable domain size | 25% of parent attachment surface | Movement range within the attachment domain |
| Rotation step (short radius) | 5° | Used when rotation radius < 7 grid cells |
| Rotation step (long radius) | 2.5° | Used when rotation radius ≥ 7 grid cells |

#### Hierarchy Metrics

Two metrics are computed in 2.1 and used here:

**Downstream dependent count** (primary movement signal):
- Total number of descendant parts below a node, excluding the node itself
- Example: a body with head(1), arms(2), hands(2), fingers(10), legs(2), feet(2) → downstream count = 19

**Depth:**
- Number of edges from root to node (root = 0)
- Used for structural ordering, but downstream dependent count is the more important signal

#### Collision Classification

After geometry is assembled and sampled into the working occupancy grid, overlaps are classified:

| Type | Definition |
|---|---|
| **Valid (structural)** | Overlap where the resolved `parent_anchor` is within the overlap volume or within 2 grid cells of it (see Attachment Validity Rule in 2.3) |
| **Invalid (interference)** | Any overlap not meeting the validity test — including sibling collisions, group collisions, and unintended penetrations |

#### Overlap Threshold

The system does not chase perfect zero overlap.

Target: reduce invalid overlap below a threshold based on the smaller part's volume (see Default Constants below).

This threshold applies to invalid overlap only, not valid structural overlap.

#### Collision Resolution Ordering

Collisions are processed in an order that minimizes later rework:
- Start with collisions where the involved primitives have higher downstream dependent counts and are closer to the root
- Resolving higher-impact collisions first reduces repeated corrections later

This ordering is separate from the per-pair decision of which part to move.

#### Part Selection for a Collision Pair

For a colliding pair A and B:

1. Choose the part with **fewer** downstream dependents (lighter subtree moves)
2. If tied, use the original Stage 1 part order as tiebreaker

#### Global Overlap Check

During a candidate movement, track:
- **Pair invalid overlap** (primary optimization target)
- **Total global overlap** across the scene (coarse side-effect signal)

Global total overlap may include valid structural overlap — it is used only as a rough side-effect monitor, not an exact optimization target. If pair overlap decreases by L but global overlap decreases by only G, then roughly L − G new overlap was introduced elsewhere.

#### Translation-Based Resolution

For translational movement within a face or edge domain (as defined in 2.3's Allowable Movement Domains):

1. Probe ±1 grid cell in each allowed local direction
2. Measure overlap change ΔV for each probe
3. For each axis, choose the signed direction that reduces overlap best
4. Combine the signed reductions into a single descent direction vector
5. Follow that direction as the movement path

The resulting vector is a descent direction based on observed overlap reduction, not a fixed target point. The direction is computed from initial local probes and followed without full recomputation at every step.

**Straight-line stepping:** Movement along the chosen direction is discretized as unit steps on the working occupancy grid. At each step: advance to the next grid cell that best approximates the continuous direction vector, ensure monotonic movement toward the target (no backtracking), and ensure adjacency between successive positions (no skipping cells). Any standard grid line-walking method that satisfies these properties is acceptable.

Stop when:
- Overlap threshold is reached
- Improvement stalls (ΔV below meaningful change)
- Domain boundary is hit
- Side effects are too large (global overlap increasing)

#### Rotation-Based Resolution

Rotation follows the same local probing logic as translation.

**Adaptive angular increment** based on effective lever arm (see Default Constants below).

**Edge region rotation:**
- Probe +step and −step around the edge axis
- Choose the better signed direction
- Continue in incremental steps
- Stop when threshold is reached, improvement stalls, or angular limit is reached

**Corner region rotation:**
- Corner attachment point is the fixed pivot
- Identify the two axes (u, v) spanning the attachment surface in the parent's local frame
- Probe four candidate rotations: +θ and −θ around u, +θ and −θ around v
- Choose the best rotation axis + direction from these four, then continue in incremental steps

Global overlap check applies to rotation candidates as well.

#### Downstream Transform Propagation

When a primitive is moved during collision resolution, that primitive is the **subtree root** for that operation. The applied transform (translation, rotation, or both) propagates rigidly:

1. **Rigid subtree transform** — the exact same transform applied to the moved part must be applied to ALL descendants
2. **Local relationships preserved** — all descendants retain their local attachment definitions, offsets, and orientations relative to their immediate parent
3. **No independent adjustment** — descendants are NOT individually moved or corrected during this step; they move only as a consequence of the root transform
4. **Global position update** — each descendant's global position and orientation must be recomputed after propagation
5. **Representation update** — `resolved_offset`, `parent_anchor`, `child_anchor`, and any derived global transform state must be updated and stored. The system must NOT rely on stale pre-move values
6. **Full subtree collision evaluation** — all collision checks must use the full transformed subtree, not just the moved primitive alone
7. **Candidate evaluation rule** — every candidate movement must be applied to the full subtree before evaluating overlap improvement. Candidate scoring is never based on the moved primitive alone

#### Representation Consistency

All movements and rotations must update the actual stored representation:
- `resolved_offset` must be updated after movement
- `parent_anchor` and `child_anchor` must be updated
- Resulting transform must be persisted
- No temporary or "virtual" geometry states are allowed

The solver always operates on the latest committed state. Future collision checks and movements use updated anchors, not original ones.

#### Symmetry Handling During Resolution

- Symmetry is enforced during initial placement and group generation (2.3)
- During collision resolution, first try group-preserving symmetric movement
- If unresolved, allow single-member movement as fallback
- Moving one symmetric part does NOT automatically move its counterpart
- Symmetry deviation introduced by a local fix must be measurable and may be used as a secondary scoring signal
- If a local fix causes excessive symmetry drift, the solver may reject it, penalize it, or flag it for later repair

#### Escalation Strategy

For a collision between A and B:

**Step 1:** Move the preferred candidate (from Part Selection above — fewer downstream dependents)

**Step 2:** If unresolved, tiebreak between parent(A) and parent(B). Move the selected parent.

**Step 3:** If Step 2 chose parent(A), try moving B. If Step 2 chose B, tiebreak between parent(A) and parent(B) and move the selected parent.

**Step 4:** If still unresolved, move the remaining untried option among A, B, parent(A), parent(B).

**Step 5 (generalization):** If still unresolved, continue escalating upward — parent → grandparent → root — applying the same tiebreak logic between the two sides at each level.

Example: a finger vs hand collision may require moving the arm. Escalation continues until the collision is resolved or root is reached.

#### Geometry-Aware Signals (Optional Heuristics)

The solver may also use:
- Overlap voxel count
- Overlap bounding box and extents along x/y/z
- Whether overlap is near root/anchor vs distal end
- Primitive type and local axes

Examples: a shallow side strip on a cylinder suggests radial/lateral movement; circular overlap on the bottom of a cylinder suggests axial lift; primitive geometry can help determine whether translation or rotation is the easier move.

These signals are allowed as heuristics but not required for the baseline solver flow.

#### Role of Gemini During Collision Resolution

Gemini is NOT in the inner collision-resolution loop. Deterministic logic is primary.

Optional Gemini use:
- Rank among already-valid candidate fixes
- Assist in rare ambiguous cases
- Provide perceptual tie-breaks

The actual collision search and probing loop is entirely deterministic.

### What Stage 2 does NOT do

- LLM calls of any kind
- Definitive voxelization (that's Stage 3 — Stage 2's working occupancy grid is temporary and discarded)
- Final scaling to voxel space (that's Stage 3)
- Semantic interpretation or object recognition
- Override structural topology from Stage 1 (except graph fixes in 2.1)

Stage 2 output is in root-normalized space (root largest dimension = 100). Stage 3 performs assembled-object bbox normalization before voxelization.

---

## Stage 3 — Voxelization

**Purpose:** Scale the assembled continuous geometry to final voxel space and produce the definitive discrete voxel grid for rendering, validation, and downstream stages.

**Execution:** Code (deterministic)

**Input:** Assembled continuous geometry from Stage 2 (in root-normalized space)

#### Global rule: Additive-only voxel modification

All voxel updates in Stage 3 must be additive:

1. Do **not** overwrite existing voxel contents.
2. Do **not** remove any existing primitive IDs from a voxel.
3. Only **append** new primitive IDs to a voxel's list if not already present.

This applies across all Stage 3 operations — parent-child bridges (3.4), fragment reconnection (3.7), and any future repair steps. No operation may delete or replace existing ownership. Multi-primitive voxels are expected and valid.

#### Global rule: Ordered ownership priority

Each voxel's primitive ID list must always be sorted by ownership priority:

1. **`critical: true` before `critical: false`.**
2. If tied, **smaller volume first.**
3. If tied, **preserve stable order** (Stage 1 declaration order or existing insertion order).

This invariant must be maintained at all times:

- **Adding** a primitive ID: insert at the correct position by priority, do not blindly append.
- **Removing** a primitive ID: remove it; remaining list stays correctly ordered (no re-sort needed).
- **Primary owner**: always the first element in the list — used for color/rendering. No recomputation needed at render time; ownership automatically updates if higher-priority primitives are removed.

### 3.1 — Assembled-Object Bbox Normalization

Stage 2 operates in root-normalized space (root largest primitive dimension = 100). The assembled object's bounding box may exceed 100 units due to spread, orientation, and support geometry. This step aligns the assembled object to near-final voxel scale.

**Steps:**

1. Compute bounding box of the fully assembled Stage 2 output: `bbox_x`, `bbox_y`, `bbox_z`
2. Determine largest extent: `max_dim = max(bbox_x, bbox_y, bbox_z)`
3. Compute uniform scale factor: `scale = 90 / max_dim` (target 90 leaves ~5 units padding on each side in a 100-unit frame)
4. Apply uniform scaling to ALL primitives and their transforms
5. Center object within a 100 × 100 × 100 continuous frame (or align to base if grounding is required)

After this step: 1 continuous unit ≈ 1 voxel. All hierarchy relationships and transforms remain intact (scaling is global and uniform).

### 3.2 — Definitive Voxelization

#### Grid Representation

A single 100 × 100 × 100 grid where each cell stores a list of primitive IDs occupying it:

| Cell contents | Meaning |
|---|---|
| `[]` | Empty |
| `[p]` | Occupied by primitive `p` |
| `[p1, p2, ...]` | Overlap between primitives `p1`, `p2`, etc. |

Implementation: use a list, set, or bitmask of primitive IDs per cell. Do NOT use string encoding for internal representation.

#### Voxelization Process

For each primitive:
1. Iterate over the relevant voxel region (bounding box of the scaled primitive)
2. Compute voxel center for each candidate cell
3. If the voxel center lies inside the primitive: append the primitive ID to that cell's list

This builds the full grid in a single pass — no separate union step is required.

#### Derived Interpretations

- **Occupied voxel:** cell is non-empty
- **Overlap voxel:** cell contains more than one primitive ID (`len > 1`)
- **Per-primitive voxel set:** all cells where a given primitive ID appears

This unifies occupancy, attribution, and overlap detection into one structure. No geometry reasoning happens here — only rasterization. Stage 2 is responsible for correct placement; Stage 3 only discretizes and validates.

### 3.3 — Connectivity Definition

**Purpose:** Define what it means for the final voxelized object to be connected.

**Connectivity target:** Union occupancy — treat every non-empty voxel as part of one global object. Primitive attribution remains available for debugging/analysis, but the connectivity test operates on the combined object, not primitive-by-primitive.

**Connectivity rule — Chebyshev distance ≤ 1:**

Two voxel groups are connected if the minimum Chebyshev distance between any voxel in one group and any voxel in the other is ≤ 1.

| Contact type | Distance | Connected? |
|---|---|---|
| Shared voxel / overlap | 0 | Yes |
| Face contact | 1 | Yes |
| Edge contact | 1 | Yes |
| Corner contact | 1 | Yes |
| 1-voxel near-gap | 1 | Yes |
| Gap > 1 voxel | > 1 | No |

Chebyshev is preferred over strict face-only (Manhattan = 1) adjacency because discretization can convert intended continuous connections into edge/corner contacts or very small gaps.

**Connected components:** Build union occupancy from all non-empty voxels. Compute connected components using the Chebyshev ≤ 1 rule (i.e., 26-connectivity flood fill). The result is a set of disjoint components.

**Grounding:** The main connected object must touch Z = 0. Grounding is checked on the final union occupancy after voxelization.

**Role of primitive IDs in this stage:**

Primitive ID lists stored per voxel are **not** used to define connectivity. They remain available to:
- Inspect which primitives contribute to a component
- Debug overlaps
- Support later repair decisions (Stage 3.4)

### 3.4 — Parent-Child Joint Validation + Repair

**Purpose:** Reinforce every intended hierarchy-defined connection by deterministically bridging each parent-child pair. This reduces false disconnections caused by discretization and simplifies implementation by removing the validation step entirely.

**Rule:** For **every** declared parent-child pair, always add a joint bridge between the resolved attachment points. Do not first test whether the pair is already connected.

#### Bridge construction

For each parent-child pair:

1. Compute a **1-voxel-thick straight line** from the parent attachment point to the child attachment point (Bresenham or equivalent). This is the centerline.
2. **Thicken** the centerline to ensure a minimum **3 wide × 2 tall** cross-section:
   - For each centerline voxel at `(x, y, z)`, also mark `(x±1, y, z)` and `(x, y±1, z)` — giving 1 voxel on either side in the X-Y plane.
   - For each of those voxels, also mark `(x, y, z+1)` — adding 1 voxel above in Z.
3. Mark each bridge voxel with both parent and child primitive IDs (per global additive-only rule).

Keep the bridge local and minimal — do not expand beyond the intended joint region.

#### Global connectivity check (post-bridging)

After all parent-child pairs are bridged:

1. Run the global connectivity check from Stage 3.3 on the updated union occupancy.
2. **Expected outcome:** if all parent-child joints are bridged, the union is usually fully connected.
3. If the union is still disconnected after bridging → treat as a later fallback case. Do **not** apply arbitrary global bridging as the primary strategy.

### 3.5 — Voxel Ownership and Rendering Priority

**Purpose:** Define how multi-primitive voxels are interpreted for rendering/color without collapsing the underlying ownership data.

#### Required per-primitive fields

Each primitive entering Stage 3 must carry:
- `part_int_id` (integer index derived from position in the `states` array; see Stage 5, §5.0 — Part Identifier System)
- `color_id`
- `critical`
- `volume` (or equivalent size measure)

#### Representation

Each voxel stores an **ordered** list of primitive IDs (per global ordered-priority invariant), preserved throughout Stage 3:
- `[]` → empty voxel
- `[P1]` → single-primitive voxel
- `[P1, P2]` → overlap or bridge voxel

A voxel is occupied if its list is non-empty. Multiple IDs indicate geometric overlap, intentional joint/bridge regions, or discretization overlap between connected parts. Ownership is **never collapsed** to a single primitive during processing.

#### Primary ownership and color

Primary owner is always the **first element** in the voxel's ordered list (`critical` flag → volume → stable order). No per-voxel recomputation needed at render time.

**Rendering rule:** voxel color = `color_id` of the primary primitive.

This naturally produces the desired visual behavior — smaller, critical details (eyes, handles, fingers) appear on top of larger parent volumes because they rank higher in the priority sort.

**Bridge voxels** (from Stage 3.4): contain both parent and child IDs; primary owner determined by the same priority invariant; remain shared structurally.

#### Final output per voxel

- **Occupancy:** filled if primitive ID list is non-empty.
- **Render/color owner:** first element in the ordered list; voxel color = its `color_id`.
- **Full primitive ID list:** preserved, available for debugging/analysis export.

### 3.6 — Overlap Resolution via Constrained Re-Solve

**Purpose:** Resolve invalid overlaps by reusing Stage 2's structured resolution logic in a constrained, targeted manner — evaluated in voxel space. This is **not** a new heuristic repair system; it is a constrained re-application of Stage 2 logic.

#### Trigger

Identify overlaps that are **invalid** under the Stage 2 overlap validity rule (§2.6). Only invalid overlaps trigger resolution.

#### Scope

1. Identify the primitives involved in the invalid overlap.
2. Select one as the **moving part** (same selection logic as Stage 2 — hierarchy metrics, critical flag, voxel count).
3. Include its **subtree** (children) in the movement.

Do **not** re-solve the entire object.

#### Resolution process

Re-run Stage 2-style solver steps:

1. **Generate candidate adjustments:**
   - Translations and rotations.
   - Respect attachment constraints.
   - Propagate movement to subtree.

2. **Evaluate each candidate:**
   - Re-voxelize affected primitives locally.
   - Evaluate overlap using Stage 2 validity rule (applied to voxels).

3. **Select best candidate:**
   - Eliminates the invalid overlap.
   - Minimizes displacement from original position.
   - Preserves symmetry when possible.

#### Constraints

- Bounded number of iterations.
- Bounded movement magnitude.
- Must not introduce new invalid overlaps elsewhere.

### 3.7 — Fractured Major-Piece Repair

**Purpose:** If a single primitive's voxels are split into multiple disconnected components (due to discretization, overlap resolution, or geometry), reconnect them using nearest-neighbor bridging.

#### Process

For each primitive whose voxels form more than one connected component (using Stage 3.3 Chebyshev ≤ 1 connectivity):

1. Identify all connected components of that primitive's voxels.
2. Initialize:
   - `connected_set` = {largest component}
   - `remaining_components` = all other components
3. While `remaining_components` is not empty:
   a. For each remaining component C_i, find the closest component C_j in `connected_set` (minimum Chebyshev distance between any voxel pair).
   b. Select the pair (C_i, C_j) with the smallest distance.
   c. Build a **3×2 bridge** between them (same construction as Stage 3.4 — centerline + ±1 X-Y + +1 Z, additive-only insertion).
   d. Move C_i into `connected_set`.
   e. Remove C_i from `remaining_components`.

#### Result

- Components are connected in nearest-neighbor order, avoiding unnatural long bridges that skip intermediate pieces.
- Bridge voxels store the repaired primitive's ID (per global additive-only rule).

### 3.9 — Final Normalization and Export

**Purpose:** Finalize Stage 3 output by applying deterministic normalization and exporting the finished voxel representation.

#### 1. Grounding normalization

1. Compute `min_z` across all occupied voxels in the final union occupancy.
2. Translate the entire voxel mass uniformly:
   - If `min_z < 0` → raise by `|min_z|`.
   - If `min_z > 0` → lower by `min_z`.
   - If `min_z = 0` → no adjustment.

**Post-condition:** The lowest occupied voxel layer must be at `z = 0`.

#### 2. Ownership integrity

- Every non-empty voxel must have a valid, non-empty, ordered primitive ID list.
- The ordering must satisfy the global ownership-priority invariant.
- The first ID is the primary owner.

If malformed ownership is detected: repair ordering if possible, otherwise mark export as degraded.

#### 3. Export artifacts

- **Ownership grid** (source of truth): the 100³ grid with ordered primitive ID lists per voxel.
- **Color/render grid**: derived from ownership grid — each voxel's color = `color_id` of its primary owner (first in list).
- **Metadata:** occupied voxel count, bounding box, final status.

Binary occupancy is always derived from the ownership grid and is **not** stored separately.

#### 4. Connectivity note

Global connectivity is **not** a hard export requirement. Earlier stages (3.4, 3.7) perform bridging/reconnection to improve output quality, but disconnected pieces do not block export. Connectivity is mainly a quality concern for later Gemini validation, not a deterministic structural requirement for export.

---

## Stage 5 — Semantic Validation

**Purpose:** Use Gemini to evaluate whether the generated object matches the user's intended concept.

**Execution:** Mixed — Code (5.0 projection) + LLM (5.1–5.2 validation)

### 5.0 — Voxel Projection (Six Views)

**Purpose:** Produce consistent, deterministic orthographic projections from the final voxel grid for Gemini evaluation.

**Execution:** Code

Generate six orthographic projections directly from the final 100×100×100 voxel grid. No scaling, no normalization, no perspective.

#### Coordinate System

- X: left (0) → right (99)
- Y: back (0) → front (99)
- Z: bottom (0) → top (99)

#### Views and Ray Directions

| View | View direction | Ray traversal | Image horizontal | Image vertical |
|---|---|---|---|---|
| **Front** | −Y | y = 99 → 0 | X (left → right) | Z (top → bottom) |
| **Back** | +Y | y = 0 → 99 | X (left → right) | Z (top → bottom) |
| **Left** | +X | x = 0 → 99 | Y (left → right) | Z (top → bottom) |
| **Right** | −X | x = 99 → 0 | Y (left → right) | Z (top → bottom) |
| **Top** | −Z | z = 99 → 0 | X (left → right) | Y (top → bottom) |
| **Bottom** | +Z | z = 0 → 99 | X (left → right) | Y (top → bottom) |

#### Rendering Rule (per pixel)

For each pixel in a 100×100 canvas:
1. Determine the corresponding `(axis1, axis2)` from the image axes.
2. Traverse along the ray axis in the specified direction.
3. Select the **first voxel** with a non-empty ownership list.
4. Assign:
   - **Pixel color** = `color_id` of the primary owner (first entry in the ordered ownership list).
   - **`part_int_id`** = the primary owner's integer index (pipeline-internal, see Part Identifier System below).

If no occupied voxel is found along the ray:
- Pixel color = **khaki** background.
- `part_int_id` = `null` (black in segmentation image).

**CRITICAL:** The color image and `part_int_id` map MUST be generated in the **same pass** — both are driven by identical ray traversal logic. No separate rendering pass. The `part_int_id` map is then used to compute projection adjacency and render the segmentation images (see Segmentation Color Assignment below).

#### Part Identifier System (Dual ID)

Each part has two identifiers serving different purposes:

| Identifier | Type | Source | Used in |
|---|---|---|---|
| **`uid`** | string (e.g. `"body_0"`, `"leg_front_left_0"`) | Assigned during generation, immutable | Gemini edit actions, structural JSON, all pipeline logic |
| **`part_int_id`** | integer (e.g. `1`, `2`, `3`) | Positional: `i + 1` from `enumerate(states)` | Voxel ownership grid, `part_int_id` map, internal computation |

**`part_int_id` is derived, not stored on the part.** It is assigned at render time from the ordered states array (`states[0]` → 1, `states[1]` → 2, etc.). It is **not stable across pipeline re-runs** if parts are added or deleted — it is purely a compact index for grid storage and view maps.

**`uid` is the stable identifier.** All Gemini edit actions (`translate`, `rotate`, `resize`, `recolor`, `add_part`, `delete`, `toggle_critical`) use `uid`. Gemini sees segmentation colors in the segmentation images and maps them to `uid` via the segmentation legend. `part_int_id` is never exposed to Gemini.

#### Part-ID Map (per view)

For each of the 6 views, generate an additional **100×100 array** alongside the color image:

```
id_map[100][100]  →  null | part_int_id
```

- `null` → background pixel (no occupied voxel along ray).
- integer → `part_int_id` of the visible voxel's primary owner.

This map is **pipeline-internal** — it is NOT sent to Gemini directly. Instead, it is used to:
1. Compute **projection adjacency** for segmentation color assignment (see Segmentation Color Assignment below).
2. Render the **segmentation image** by mapping each `part_int_id` to its assigned segmentation color.

**Purpose:** Intermediate data structure that enables part-aware segmentation images without exposing raw integer arrays to Gemini.

#### Segmentation Image

Each view produces a **separate 100×100 PNG** — a color-coded segmentation image where each pixel's color identifies the visible part.

**Rendering rules:**
- Each pixel is a flat, solid color mapped to its owning part's `uid` via the segmentation legend.
- **Nearest-neighbor rendering** — no anti-aliasing, no blending, no smoothing. Hard pixel boundaries.
- Background (empty) pixels = **black** (`#000000`). Distinct from khaki in the color image.
- No grid overlay, no text overlays, no annotations.

**Why images over JSON/CSV:** A 100×100 PNG costs **258 tokens** in Gemini (single tile, under 384px threshold). The same data as compact CSV costs ~5,000-6,500 tokens per view — a 20× overhead. Additionally, Gemini can spatially reason over an image natively but cannot reliably index into a 10,000-element text array.

**Prompt note for small parts:** Parts occupying fewer than ~20 pixels in a view may not be clearly visible in segmentation maps. The prompt tells Gemini: "Small parts may not appear in segmentation maps. Refer to the structural JSON for their exact positions."

#### Segmentation Color Assignment

Segmentation colors are a **separate palette** from the object's `color_id` values. They have no semantic meaning — their only job is to make adjacent parts visually distinguishable.

**Fixed palette:** 40 pre-defined, maximally-distinct colors (evenly spaced hues in HSL at high saturation, with alternating lightness levels). Hardcoded once. Ordered so that the first N colors are always the most mutually distinct — picking the top 8 gives 8 well-separated colors, picking the top 20 still works.

**Adjacency-aware assignment:**

Color assignment uses projection adjacency — computed from the `part_int_id` maps that are already generated in the ray traversal pass.

1. **Generate all 6 `part_int_id` maps** (already produced during ray traversal — see Rendering Rule above).
2. **Build projection adjacency graph.** For each id map, scan every pixel and check its 4-connected neighbors. If two different `part_int_id` values are neighbors, record that pair. Union adjacency across all 6 views — if parts A and B are pixel-neighbors in *any* view, they need distinct colors.
3. **Greedy graph coloring.** Sort parts by adjacency count (most neighbors first). For each part, assign the first color from the ranked palette that isn't used by any of its projection-adjacent neighbors.

**Why projection adjacency, not voxel adjacency or hierarchy:**
- Two parts can be 3D voxel-neighbors but one is completely occluded in a given view — no contrast needed there.
- Two parts with no 3D contact can be pixel-neighbors in a view because one is behind the other — contrast IS needed there.
- Projection adjacency captures exactly what matters: which parts Gemini will see next to each other in the segmentation image.

**Compute cost:** 6 views × 100 × 100 × 4 neighbor checks = 240k integer comparisons. Trivial. Graph coloring with 40 colors and typical 10-30 parts is always solvable in a single greedy pass.

**Implementation sequence:** Ray traversal → `part_int_id` maps → projection adjacency → color assignment → render segmentation PNGs. Linear dependency chain, no branching or fallbacks.

#### Segmentation Legend (for Gemini)

Gemini receives a text lookup table mapping segmentation colors directly to `uid`:

```
Segmentation map legend:
- red (#FF0000) → uid "body_0", "body" (ellipsoid)
- green (#00FF00) → uid "head_0", "head" (ellipsoid)
- blue (#0000FF) → uid "leg_front_left_0", "front left leg" (cylinder)
...
```

Rules:
- Include both human-readable color name AND hex — Gemini maps natural-language color names more reliably than raw hex codes.
- Each color maps to exactly one `uid`. No duplicates.
- Color assignment is consistent across all 6 views — a part is always the same segmentation color regardless of view.
- Gemini uses `uid` directly in all edit actions. No intermediate ID lookups.

**Note:** `part_int_id` (the 1-based integer index used in the voxel grid and id maps) is **pipeline-internal only**. It is not exposed to Gemini and does not appear in the legend or edit actions.

#### Primary Owner Rule

For both the color image and the segmentation image, the pixel value always comes from the **primary owner** of the voxel — the first entry in the ordered ownership list (per the ownership priority invariant defined in §3.5: `critical` flag → volume → stable order).

This ensures:
- Color image and segmentation image are **perfectly aligned** — same part owns the pixel in both.
- Deterministic attribution: if a transformation removes the primary owner, the next entry in the ownership list automatically becomes the new primary owner. No recomputation needed.
- Empty ownership list → empty voxel (background in both outputs).

#### Canvas

- Fixed **100×100 pixels** per view.
- 1 pixel = 1 voxel projection (no scaling).

#### Grid Overlay

- Thin lines every 10 units.
- Subtle contrast — must not overpower object colors.
- **Purpose for Gemini:** The grid helps Gemini gauge scale and identify empty space. The prompt explicitly tells Gemini: "The khaki background represents empty space. The grid lines mark 10-unit intervals for scale reference. Neither the background nor the grid are part of the object."
- Grid overlay applies to the **color image only**, not the `part_int_id` map.

#### Constraints

- No scaling.
- No smoothing or interpolation.
- No lighting or shading.
- No cropping or centering adjustments.
- No perspective projection.

---

### 5.1 — Gemini Input Specification

**Purpose:** Define what Gemini receives. No internal implementation artifacts are included.

Gemini receives exactly three inputs:
1. **User intent** — original text description + optional image.
2. **Structural representation** — full `part_types` + `instances` JSON from Stage 2 (see 5.2 for exact format).
3. **Six voxel-based views** — orthographic projections from 5.0, reflecting the final Stage 3 output (including bridging, repair, grounding normalization).

Each view is passed as a **separately labeled image attachment** with its view name (Front, Back, Left, Right, Top, Bottom) so Gemini can map spatial reasoning to the correct projection.

**Exclusions:** Do **not** include overlap diagnostics, bridge counts, repair logs, or internal solver details. Gemini evaluates based on visual outcome and intended structure, not implementation artifacts.

### 5.2 — Gemini Validation Prompt

**Purpose:** Define the exact prompt Gemini receives for semantic validation. Gemini identifies deviations from user intent and outputs structured edit operations that the pipeline can execute deterministically.

#### Role Definition

```
You are evaluating a voxelized 3D object against its intended design and proposing bounded, executable corrections.

You are NOT generating a new design.
You are NOT re-describing the object.
You are NOT allowed to suggest a full rebuild.

Your role:
1. FIRST: Look at the six projection views ONLY. Write your "guess":
   - What is the object? (e.g., "a spider", "a person", "a house")
   - What stands out? Note posture, held items, worn items, or distinctive features visible in the silhouette.
   - What is ambiguous or unrecognizable?
   This tests recognizability — if you cannot identify the object, that is itself a critical issue that should drive your edits.
   Do NOT read the structural representation or user intent before completing this step.
2. THEN: Read the user intent and structural representation.
3. Compare your visual impression against the intended design.
4. Output structured edit operations (`translate`, `rotate`, `resize`, `recolor`, `add_part`, `delete`, `toggle_critical`) that the pipeline will execute deterministically.
```

**Pipeline context (included in prompt):**

```
What you're looking at: Earlier in this pipeline, you read a user's description and decomposed it
into a part-based structural representation. That representation was then processed by deterministic
pipeline stages — cleanup, scaling, collision resolution, voxelization — to produce the 3D object
shown in these views. Some of what you see may differ from your original intent due to pipeline
processing (overlap resolution, grounding, bridging). Your job is to identify where the result
diverges from the user's intent and propose targeted fixes. This is your one revision pass — make
it count. If something is fundamentally broken beyond what targeted edits can fix, say so in your
issues list and focus your edits on the highest-impact improvements you can make.
```

**Confidence-driven strategy:**

```
How your confidence level should guide your edits:
- high: Object is recognizable. Focus on fine-tuning — proportions, small placement adjustments,
  color corrections. Fewer edits is fine.
- medium: Object is identifiable but has notable issues. Fix the most impactful proportion,
  placement, or missing-part problems.
- low: Object is difficult to recognize. Prioritize silhouette and overall shape — large resize,
  reattachment, or add/delete to recover the intended form. Detail edits are wasted at this level.
```

**Prompt delivery order:** The six view images are provided FIRST in the message, before the structural representation and user intent. This ensures Gemini's `guess` is an unbiased visual impression, not influenced by knowing what the object is supposed to be.

#### Edit Example

A well-reasoned edit cross-references the visual issue with the structural data:

```
Issue: "Front legs appear stubby in front/side views — current height is 10 but body height is 15,
       legs should be roughly 1.2× body height for a spider"

Edit:  { "action": "resize", "uid": "leg_front_left_0",
         "dimensions": { "width": 4, "depth": 4, "height": 18 }, "top_radius": null }
```

The reasoning references the projection (visual), the structural JSON (current dimensions), and proportional logic (relative to sibling/parent parts). This is the level of justification expected in the `issues` list.

#### Context Provided to Gemini

Gemini receives three inputs:

**1. Original user intent**
- Text description.
- Optional image input (if provided by user).

**2. Structural representation (from Stage 2)**

Provided as JSON in the same `part_types` + `instances` format used during generation:

```json
{
  "entity_name": "spider",
  "category": "living_being",
  "part_types": [
    {
      "type_id": "body",
      "part_name": "body",
      "primitive_type": "ellipsoid",
      "dimensions": { "width": 30, "depth": 20, "height": 15 },
      "rotation": { "rx": 0, "ry": 0, "rz": 0 },
      "top_radius": null,
      "critical": true,
      "color_id": "black"
    }
  ],
  "instances": [
    {
      "uid": "body_0",
      "type_id": "body",
      "parent_part": null,
      "parent_face": null,
      "child_face": null,
      "attachment_offset": 0,
      "attachment_offset_v": 0
    },
    {
      "uid": "leg_front_left_0",
      "type_id": "leg",
      "parent_part": "body_0",
      "parent_face": "left",
      "child_face": "right",
      "attachment_offset": 0.6,
      "attachment_offset_v": 0.3
    }
  ]
}
```

Each part is uniquely identified by its **`uid`** (from the instances array). All edit operations target parts by `uid`.

**3. Final voxel-based views (from Stage 3 output)**

6 standard orthographic projections as defined in **5.0 — Voxel Projection (Six Views)**. Each passed as a separately labeled image attachment (see 5.1).

The khaki background represents empty space. The grid lines mark 10-unit intervals for scale reference. Neither the background nor the grid lines are part of the object.

**4. Segmentation images**

Per-view 100×100 color-coded segmentation PNGs, generated in the same ray traversal pass as the color images (see 5.0 — Segmentation Image). Each pixel's color identifies the owning part. Allows Gemini to attribute visual artifacts to specific parts without relying on object color alone.

**5. Segmentation legend**

Text mapping of segmentation colors to `uid`, `part_name`, and `primitive_type` (see 5.0 — Segmentation Legend). Gemini identifies a part by its segmentation color, looks up its `uid` in the legend, and uses `uid` directly in all edit actions. No intermediate ID lookups.

#### Coordinate System

```
Grid: 100 × 100 × 100 (integer voxel space)
X-axis: left (−) to right (+)
Y-axis: back (−) to front (+)
Z-axis: bottom (0) to top (+)

Direction vocabulary (STRICT):
  left / right     → X-axis
  forward / backward → Y-axis
  up / down        → Z-axis

All dimensions are FULL extents (not radii or half-extents).
```

#### Spatial Conventions

These conventions match Stage 2 exactly. Gemini must follow them when proposing edits.

**Attachment model (2-face):**
- Each attachment specifies two faces: `parent_face` (which face of the parent) and `child_face` (which face of the child contacts the parent).
- Valid faces: `top`, `bottom`, `front`, `back`, `left`, `right`.
- Position on the parent face is controlled by two normalized offsets: `attachment_offset` (u-axis) and `attachment_offset_v` (v-axis).
- Range: `[-1, 1]` where `0` = face center, `±1` = face edges.
- The child is oriented so that its `child_face` aligns with the parent's `parent_face`. For example, `parent_face: "top", child_face: "bottom"` means the child sits on top of the parent.

**Attachment axis mapping (u, v per face):**

| Face | u axis | v axis |
|---|---|---|
| `top` / `bottom` | X (left → right) | Y (back → front) |
| `front` / `back` | X (left → right) | Z (bottom → top) |
| `left` / `right` | Y (back → front) | Z (bottom → top) |

**Rotation:**
- Specified as `{ "rx": float, "ry": float, "rz": float }` in degrees.
- Applied in Z → Y → X order (Euler).
- Rotation is hierarchical: children inherit all ancestor rotations.
- Root parts rotate around their own center.
- Child parts rotate around their attachment anchor (the point where they connect to the parent).

**Dimensions:**
- Always in **pre-scale, root-normalized object space** (root's largest dimension = 100 units).
- Specified as `{ "width": int, "depth": int, "height": int }` — full extents, not radii. Integer values in root-normalized space (root = 100).
- Resizing is anchored at the attachment face: if a child is attached to its parent's `top` face and its height increases, it grows upward (away from the attachment).

**Dimension reasoning (for resize and add_part):**
- All dimensions must be absolute integers in the same root-normalized space as the structural JSON (root = 100). Output the full intended `{ width, depth, height }`, not deltas.
- Do not guess dimensions from visual appearance alone. Cross-reference the current part's dimensions, surrounding parts' dimensions in the structural JSON, and visible proportions in the projections.
- Prefer small adjustments to the current values when proportions are roughly correct. Large changes are justified when proportions are clearly wrong relative to the overall object (e.g., a head that's 50% too small needs a significant resize, not a 2-unit nudge).
- Preserve proportional consistency with adjacent and connected parts — resizing one part should not make it visually inconsistent with its siblings or parent.

**Primitive types:**

| Type | Parameterization |
|---|---|
| `cuboid` | `width` (X), `depth` (Y), `height` (Z) — standard box |
| `cylinder` | `width` = `depth` = base diameter, `height` = Z extent |
| `ellipsoid` | `width` (X diameter), `depth` (Y diameter), `height` (Z diameter) |
| `cone_frustum` | `width` = `depth` = base diameter, `height` = Z extent, `top_radius` = top circle radius (0 = full cone) |

**Hierarchy and subtree behavior:**
- Parts form a tree rooted at the single root part (`parent_part: null`).
- When a part is modified (moved, rotated, resized), all its descendants move with it as a rigid subtree.
- Deleting a part deletes its entire subtree (all descendants are removed).

**Criticality:**
- `critical: true` — part is essential for recognition (must survive pipeline processing; at least one root part must be critical).
- `critical: false` — part is decorative or secondary (may be pruned by pipeline if it causes conflicts).

#### Color Palette

20 fixed colors. You MUST use exact `color_id` strings from this list:

| color_id | Hex | Typical use |
|---|---|---|
| `black` | #1B2A34 | Dark surfaces, outlines, tires |
| `white` | #F4F4F4 | Eyes, teeth, highlights |
| `dark_bluish_gray` | #6B7280 | Metal, machinery, stone |
| `light_bluish_gray` | #A8AFB8 | Light metal, concrete |
| `red` | #C91A09 | Primary red surfaces |
| `dark_red` | #720E0F | Dark accents, deep red |
| `blue` | #0055BF | Primary blue surfaces |
| `light_blue` | #5A93DB | Sky, water, light accents |
| `yellow` | #F2CD37 | Gold, highlights, skin (stylized) |
| `orange` | #D67923 | Warm accents, construction |
| `green` | #237841 | Foliage, primary green |
| `dark_green` | #184632 | Deep foliage, military |
| `tan` | #E4CD9E | Skin, sand, light wood |
| `dark_tan` | #B0A06F | Leather, aged surfaces |
| `brown` | #582A12 | Wood, earth, hair |
| `dark_brown` | #352100 | Dark wood, bark |
| `pink` | #FC97AC | Light accents, flowers |
| `purple` | #D3359D | Decorative, magical |
| `teal` | #069D9F | Water, accent |
| `lime` | #A5CA18 | Bright accent, alien/neon |

Do NOT use color names outside this list. Common aliases (e.g., "grey", "gold", "beige") are handled by the pipeline — always use the exact `color_id` above.

#### Allowed Actions

Gemini may ONLY use these seven actions:

**1. `translate`** — Move an existing part to a new attachment position.

Provide the target `uid` and the new attachment values. The pipeline diffs against the original to determine what changed.

```json
{
  "action": "translate",
  "uid": "leg_front_left_0",
  "parent_face": "left",
  "child_face": "right",
  "attachment_offset": 0.8,
  "attachment_offset_v": 0.3
}
```

Fields: `uid` (target), `parent_face`, `child_face`, `attachment_offset`, `attachment_offset_v` — the new desired values.

Changing `parent_face` or `child_face` (reattaching to a different face) is higher disruption than sliding offsets on the same face. Descendants move with the part through normal subtree propagation.

**2. `rotate`** — Rotate an existing part.

Provide the target `uid` and the new rotation values.

```json
{
  "action": "rotate",
  "uid": "leg_front_left_0",
  "rotation": { "rx": 0, "ry": 0, "rz": -45 }
}
```

Fields: `uid` (target), `rotation` object with `rx`, `ry`, `rz` in degrees — the new desired values.

Descendants inherit the transform through subtree propagation.

**3. `resize`** — Change part dimensions.

Provide the target `uid` and the new dimensions. Include `top_radius` if the part is a `cone_frustum`.

```json
{
  "action": "resize",
  "uid": "leg_front_left_0",
  "dimensions": { "width": 4, "depth": 4, "height": 18 },
  "top_radius": null
}
```

Fields: `uid` (target), `dimensions` object with `width`, `depth`, `height`. Optional: `top_radius` (`cone_frustum` only, `null` otherwise).

All values must be positive integers in root-normalized space. Minimum increment/decrement is 1 unit (≈ 1 voxel after Stage 3 scaling). If resize materially changes attachment geometry, downstream rebuild recomputes placement and collision.

**4. `recolor`** — Change a part's color.

```json
{
  "action": "recolor",
  "uid": "leg_front_left_0",
  "color_id": "dark_brown"
}
```

Fields: `uid` (target), `color_id` (must be valid palette color).

**5. `add_part`** — Add a new part attached to an existing parent.

Provide a `ref_id` for cross-referencing within the same edit batch, plus `part_type` and `instance` definitions. The system generates the actual `uid` — do NOT include `uid` in the instance.

```json
{
  "action": "add_part",
  "ref_id": "horn_left",
  "part_type": {
    "type_id": "horn_left",
    "part_name": "left horn",
    "primitive_type": "cone_frustum",
    "dimensions": { "width": 4, "depth": 4, "height": 10 },
    "rotation": { "rx": 0, "ry": 0, "rz": 20 },
    "top_radius": 1,
    "critical": false,
    "color_id": "dark_brown"
  },
  "instance": {
    "type_id": "horn_left",
    "parent_part": "head_0",
    "parent_face": "top",
    "child_face": "bottom",
    "attachment_offset": -0.4,
    "attachment_offset_v": 0
  }
}
```

`ref_id` can be any descriptive string. If a subsequent action in the same batch references this `ref_id` as a `uid` or `parent_part`, the pipeline resolves it to the system-generated `uid` (see Stage 6, §6.5).

`parent_part` must reference an existing part or a `ref_id` from an earlier `add_part` in the same batch.

**6. `toggle_critical`** — Flip the `critical` flag on one or more parts.

Provide a list of `uid`s. Each part's `critical` field is flipped (`true` → `false`, `false` → `true`).

```json
{
  "action": "toggle_critical",
  "uids": ["tail_0", "fin_left_0"]
}
```

Fields: `uids` (array of target part identifiers).

**Constraint:** A part cannot be set to `critical: false` if any of its descendants are `critical: true`. The pipeline skips that part (with warning) and processes the rest of the list.

**7. `delete`** — Remove a part and its entire subtree.

```json
{
  "action": "delete",
  "uid": "tail_0"
}
```

**WARNING:** Deletion is recursive — all children of the deleted part are also removed. If Gemini intends to replace a part (e.g., swap a cuboid leg for a cylinder leg), it must:
1. `delete` the old part (which removes the subtree), then
2. `add_part` the replacement part, then
3. `add_part` each child that was on the old part, re-attaching them to the new part.

**Re-parenting a part** (changing which parent it attaches to) requires `delete` + `add_part` — the entire subtree must be rebuilt under the new parent. This counts against the structural budget.

If a correction cannot be expressed using these seven actions — DO NOT include it.

#### Edit Budget

Matches the pipeline's enforcement limits:

| Category | Max | Actions |
|---|---|---|
| **Structural changes** | 5 | `add_part`, `delete` |
| **Local edits** | 20 | `translate`, `rotate`, `resize`, `recolor`, `toggle_critical` |

Each granular action counts as 1 against its category limit.

Use as many changes as needed to meaningfully improve the object, up to these limits.

**Note:** Replace operations (delete + re-add subtree) can consume multiple structural slots. A part with 3 children costs 5 structural changes to replace (1 delete + 4 adds). Budget tuning may be needed after testing.

#### Execution Strategy

Edits are executed by Stage 6 (Bounded Revision). See Stage 6 for ranking, sanity checks, rollback rules, and rebuild strategy.

#### Pipeline-Handled (DO NOT suggest fixes for these)

The pipeline has already handled and will re-handle after the final rebuild:
- Connectivity (bridging, pruning)
- Collision / overlap resolution
- Scaling and normalization to grid
- Grounding (Z = 0)
- Voxel ownership and color resolution
- Critical part survival (restoration)
- Structural validity of attachments

#### Focus Areas

Prioritize corrections related to:
- **Proportions** — part dimensions don't match intent
- **Missing parts** — something clearly absent from the silhouette
- **Placement** — attachment face or offset is wrong
- **Orientation** — rotation doesn't match intent
- **Silhouette** — overall shape isn't recognizable
- **Color** — wrong `color_id` for a part

Prefer local edits (`translate`, `rotate`, `resize`, `recolor`) over structural changes (`add_part`, `delete`) unless something is clearly missing.

#### Output Schema (Validation Mode)

```json
{
  "guess": "what this looks like from the views alone (blind test)",
  "confidence": "high | medium | low",
  "no_edits_needed": false,
  "issues": [
    "description of issue 1",
    "description of issue 2"
  ],
  "edits": [ ... ]
}
```

**Rules:**
- Each edit must address an identifiable issue. Multiple edits may address the same issue across symmetric or repeated parts (e.g., resizing all 4 legs that are too short).
- Each edit must specify concrete field values for a single action — no prose descriptions. Edits that affect a parent naturally cascade to its subtree; this is expected.
- `guess` is Gemini's independent visual impression before reading the structural representation — identify the object, note distinctive features (posture, held/worn items, silhouette), and flag anything ambiguous. This tests recognizability.
- If no corrections are needed, return `"no_edits_needed": true` with `"edits": []` and `"confidence": "high"`. This explicit flag distinguishes "object is good" from a malformed/truncated response that happens to have empty edits.

**Skip rebuild rule:** If Gemini returns `no_edits_needed: true` AND `confidence: "high"` AND `edits` is empty, the pipeline skips the Stage 2→3 rebuild entirely. All three conditions must be present — if any is missing, treat as a normal response and proceed through Stage 6.

#### Output Schema (Feedback Mode)

See Stage 7 for the feedback-mode prompt, output schema, and edit budget.

---

## Stage 6 — Bounded Revision

**Purpose:** Take Gemini's structured edits from Stage 5 and apply them to the canonical Stage 2 representation in a bounded, deterministic way.

**Execution:** Code

### 6.1 — Execution Contract

**Pre-step: UID pre-assignment** (runs once before ranking/execution):
1. Scan all `add_part` actions in the batch
2. For each, generate the real `uid` as `{type_id}_{next_counter}` (see §6.5)
3. Replace all `ref_id` references throughout the entire batch with the assigned `uid`s
4. Build a set of **pre-assigned UIDs** — these are known but not yet created in the representation

**Per-edit execution** (in ranked order per §6.2):

1. **Parse** — JSON must parse; `action` must be in allowed set (`translate`, `rotate`, `resize`, `recolor`, `toggle_critical`, `add_part`, `delete`)
2. **Reference validation** — target `uid` must exist in the representation OR in the pre-assigned UID set; `parent_part` must exist or be pre-assigned; `delete` cannot target root
3. **Parameter validation** — all required fields present; enum values valid (e.g., `parent_face`, `color_id`, `primitive_type`); numeric values within allowed range; dimensions positive
4. **Map to representation** — convert action fields to Stage 2 canonical mutations (see §6.3)
5. **Apply** — mutate Stage 2 representation; do NOT directly edit voxel grid
6. **Cheap sanity check** — compare pre/post metrics; reject if blast radius exceeds thresholds (see §6.4)

After all accepted changes:
- Rerun deterministic pipeline once: Stage 2 rebuild → Stage 3 voxelization → Stage 3 projection generation

### 6.2 — Change Application Order

Apply edits in ranked order from lowest disruption to highest disruption:

1. `recolor`
2. `toggle_critical`
3. `translate` — same-face offset changes only
4. `translate` — `parent_face` change
5. `rotate` — leaf-level parts
6. `resize` — leaf-level parts
7. `rotate` / `resize` — higher-level parts (more downstream dependents)
8. `add_part`
9. `delete`

**Tie-breakers** (within the same tier):
- Fewer downstream dependents first
- Smaller magnitude of change first
- Lower subtree depth first

**Within-tier ordering:** Gemini's relative ordering of edits is preserved within the same disruption tier. This is critical for delete-before-add replacement pairs — the pipeline re-ranks across tiers but does not reorder within a tier.

**Dependency constraint:** If an action targets a pre-assigned UID (from §6.1 pre-step), it must execute after the `add_part` that creates that part — regardless of disruption tier. This allows Gemini to add a part and then translate/resize/recolor it in the same batch without the ranking breaking the dependency.

**Rationale:** Full rebuild per edit is O(n × pipeline cost). This approach is O(n × cheap check + 1 × pipeline cost) — faster while still protecting against high-blast-radius edits.

### 6.3 — Representation Mapping

Each action maps to specific mutations on the Stage 2 canonical representation (`part_types` + `instances`). The pipeline does NOT edit the voxel grid directly — geometry changes are realized by the full rebuild at the end.

#### `translate` → Instance mutation

Mutates instance fields: `parent_face`, `child_face`, `attachment_offset`, `attachment_offset_v`.

- Gemini provides the new absolute values (not deltas); pipeline diffs against the original to determine what changed
- `attachment_offset` and `attachment_offset_v` must remain in [-1, 1] range
- Changing `parent_face` or `child_face` (reattaching to a different face) is geometrically larger than sliding offsets on the same face
- If the target is a subtree root, descendants move with it through normal propagation during rebuild
- Invalid `parent_face` for the parent's geometry → reject
- `child_face` determines which face of the child aligns with the parent face

#### `rotate` → Part type mutation

Mutates part_type fields: `rotation.rx`, `rotation.ry`, `rotation.rz`.

- Gemini provides the new absolute rotation values in degrees (not deltas)
- Descendants inherit the transform through subtree propagation during rebuild
- If rotation causes the part to clip through its parent or siblings, the sanity check catches it

#### `resize` → Part type mutation

Mutates part_type fields: `dimensions.width`, `dimensions.depth`, `dimensions.height`. Optionally: `top_radius` for `cone_frustum`.

- Gemini provides the new absolute dimension values (not deltas)
- All values must be positive integers in root-normalized space; minimum value 1
- If resize materially changes the part's attachment surface geometry, the rebuild recomputes downstream placement and collision
- `top_radius` only valid for `cone_frustum` parts; ignored (with warning) for other primitive types

#### `recolor` → Part type mutation

Mutates part_type field: `color_id`.

- Must be a valid `color_id` from the palette
- No geometry rebuild needed logically, but projections still regenerate in the final rebuild

#### `toggle_critical` → Part type mutation

Mutates part_type field: `critical` (boolean flip).

- For each `uid` in the `uids` list, flip `critical`: `true` → `false`, `false` → `true`
- **Guard:** Cannot set `critical: false` on a part that has any descendant with `critical: true`. The pipeline skips that part with a warning and processes the rest
- No geometry rebuild needed — affects ownership priority and critical-part survival logic only
- `add_part` already sets `critical` on new parts; `delete` handles cascade removal. This action only covers flipping on existing parts

#### `add_part` → Create new part_type + instance

- System generates `uid` (see §6.5)
- Creates new `part_type` entry with all required fields (`type_id`, `part_name`, `primitive_type`, `dimensions`, `rotation`, `top_radius`, `critical`, `color_id`)
- Creates new `instance` entry with system-assigned `uid`, `type_id`, `parent_part`, `parent_face`, `child_face`, `attachment_offset`, `attachment_offset_v`
- Inserts into canonical hierarchy under specified `parent_part`
- All attachment fields validated against parent geometry
- `parent_part` may reference an existing `uid` or a `ref_id` from an earlier `add_part` in the same batch

#### `delete` → Remove part + subtree

- Removes the target part and all descendants recursively from both `part_types` and `instances`
- Cannot leave orphaned descendants (enforced by recursive delete)
- Root deletion disallowed in v1

### 6.4 — Cheap Sanity Check

After each individual applied change, compute quick pre/post metrics **approximated from the Stage 2 representation** — no voxelization needed.

| Metric | Approximation from Stage 2 | What it catches |
|---|---|---|
| Bounding box overlap count | Count pairs of parts whose axis-aligned bounding boxes intersect, excluding valid parent-child overlaps (per §2.6) | Invalid part collisions |
| Disconnected parent-child joint count | For each parent-child pair, check whether the child's attachment point (derived from `parent_face` + offsets + parent bbox) falls on or near the parent's surface | Broken attachments after translate/resize |
| Connected component count | Walk the parent-child tree; count roots (should be 1) | Structure splitting into disconnected pieces |
| Total bounding volume delta | Sum of (width × depth × height) for all parts, compare pre/post | Gross size change (part disappeared or exploded) |
| Root bounding box delta | Compute axis-aligned bbox enclosing all parts, compare pre/post extents | Shape escaping expected bounds |

All metrics use primitive bounding boxes and the tree hierarchy — O(n) or O(n²) in part count, not voxel count. This keeps the per-edit check cheap relative to a full Stage 3 voxelization.

If any metric exceeds its threshold → reject that change, restore previous representation state, continue to next change.

Accepted changes update the baseline for subsequent checks. Rejected changes do not.

**Thresholds:** TODO — define after testing. Initial values should be conservative (reject more) and relax based on observed false-rejection rates.

### 6.5 — UID Pre-Assignment

System-controlled, executed in the §6.1 pre-step **before** ranking or applying any edits.

1. Scan the entire edit batch for `add_part` actions
2. For each `add_part`, generate a real `uid` as `{type_id}_{next_counter}` where counter ensures uniqueness across all existing uids AND all other pre-assigned uids in this batch
3. Build a `ref_id` → `uid` mapping for the batch
4. Replace every `ref_id` reference in the batch (any field: `uid`, `parent_part`) with the resolved `uid`
5. Record the set of pre-assigned UIDs so that reference validation (§6.1 step 2) accepts them as valid targets even before the `add_part` executes

This ensures:
- Deterministic, collision-free uid generation
- Consistent naming convention (`type_id` + numeric suffix)
- Gemini can build multi-part structures in a single batch (e.g., add a limb then add a hand attached to it)
- Gemini can add a part and then translate/resize/recolor it in the same batch — the dependency constraint in §6.2 ensures correct execution order

### 6.6 — Rollback Rule

Each change is applied **transactionally**:

- If reference validation fails → skip that change
- If parameter validation fails → skip that change
- If sanity check fails → rollback that change, restore previous state
- If representation becomes structurally invalid → rollback that change

In all cases: keep earlier accepted changes, continue to next change in the ranked order.

### 6.7 — Output

Stage 6 produces:

1. **Updated canonical Stage 2 representation** — `part_types` + `instances` with all accepted mutations applied
2. **Accepted changes** — list with action details and system-assigned `uid`s for any `add_part` actions
3. **Rejected changes** — list with action details and rejection reason (validation failure, sanity check exceeded, etc.)

Then: one full deterministic rebuild — Stage 2 (structural cleanup) → Stage 3 (voxelization + projection generation).

---

## Stage 7 — User Feedback

**Purpose:** Translate free-form user feedback into the same bounded edit schema used by Stage 5/6, then execute and rebuild. Not a separate system — this is Stage 5 in feedback mode.

**Execution:** LLM + Code (same Stage 5 → Stage 6 → rebuild flow)

### 7.1 — Flow

```
User feedback text
      ↓
Stage 5 (feedback mode) — LLM interprets feedback into edits
      ↓
Stage 6 — deterministic edit application
      ↓
Rebuild: Stage 2 → Stage 3 (new projections)
      ↓
Render → user sees result
```

Single pass per feedback round. No additional LLM loops — the user sees the result and decides whether to give more feedback.

### 7.2 — Inputs

Same as validation mode (§5.1), plus one additional input:

1. **Projections** — 6 orthographic views (current state)
2. **Segmentation maps + legend** — color → uid mapping
3. **Structural representation** — current canonical `part_types` + `instances` JSON
4. **User's original intent** — the original text/image description
5. **`user_feedback_text`** — free-form natural language from the user (NEW)

### 7.3 — Prompt (Feedback Mode)

#### Role Definition

```
You are translating user feedback into bounded, executable edits on an existing 3D object.

You are NOT independently critiquing the object.
You are NOT redesigning or rebuilding the object.
You are NOT adding your own aesthetic opinions unless needed to interpret the feedback.

Your role:
1. Read the user's feedback.
2. Examine the projections and structural representation to understand the current state.
3. Determine what the user wants changed — map their natural language to specific parts
   and specific actions.
4. Output structured edit operations that the pipeline will execute deterministically.

If the feedback is ambiguous, make your best interpretation and explain it in
feedback_interpretation. If the feedback requests something impossible within the edit
system (e.g., "make it photorealistic"), say so in feedback_interpretation and return
empty edits.
```

#### What changes from validation mode

| Aspect | Validation mode | Feedback mode |
|---|---|---|
| **Trigger** | Automatic after initial build | User submits feedback text |
| **Blind guess** | Yes — tests recognizability | No — object identity is known |
| **Who identifies issues** | Gemini self-critiques | User tells Gemini what to fix |
| **Independent critique** | Yes — Gemini finds problems | No — only interpret user's feedback |
| **Confidence meaning** | "How recognizable is this?" | "How well did I understand the user?" |
| **Priority** | Silhouette and recognizability | User's stated intent |

#### Shared sections (reused from validation mode)

The following sections apply identically in both modes — do NOT duplicate:
- Spatial Conventions (§5.2) — attachment model, rotation, dimensions, primitives
- Dimension reasoning — cross-reference structural JSON with projections
- Color Palette — same 20 fixed colors
- Allowed Actions ��� same 7 actions with same schemas
- Pipeline-Handled — same list of things NOT to fix
- Execution Strategy — edits executed by Stage 6
- All Stage 6 mechanics — ranking, UID pre-assignment, sanity checks, rollback

#### Focus Areas (Feedback Mode)

Prioritize translating the user's feedback into edits. Specifically:

- **Explicit requests** — "make the legs longer" → resize the legs. Direct mapping.
- **Implicit requests** — "it doesn't look like a spider" → compare against spider characteristics, identify the biggest gaps, fix those.
- **Ambiguous requests** — "make it better" → use your judgment on the most impactful improvements, but explain your interpretation.
- **Contradictory requests** — if the user asks for changes that conflict with each other or with the object's structure, apply the most reasonable subset and note what was skipped in `feedback_interpretation`.

Do NOT independently add fixes the user didn't ask for. If you notice something clearly broken while interpreting the feedback, you may include it but flag it in `feedback_interpretation` as an additional fix beyond what was requested.

### 7.4 — Edit Budget (Feedback Mode)

| Category | Max | Actions |
|---|---|---|
| **Structural changes** | 8 | `add_part`, `delete` |
| **Local edits** | 30 | `translate`, `rotate`, `resize`, `recolor`, `toggle_critical` |

Slightly higher than validation mode (5/20) because user feedback may request broader changes. Still bounded — if feedback implies more changes than the budget allows, prioritize by impact and note what was deferred in `feedback_interpretation`.

### 7.5 — Output Schema (Feedback Mode)

```json
{
  "feedback_interpretation": "What I understood the user wants: ...",
  "confidence": "high | medium | low",
  "edits": [ ... ]
}
```

| Field | Description |
|---|---|
| `feedback_interpretation` | Gemini's summary of what it understood from the user's feedback. Includes: what was requested, what was ambiguous, what (if anything) couldn't be done within the edit system, and any additional fixes included beyond the user's request. |
| `confidence` | How confident Gemini is in its interpretation. `high` = feedback was clear and directly mappable. `medium` = some inference required. `low` = feedback was vague or ambiguous, edits are best-effort. |
| `edits` | Same edit array as validation mode — same 7 actions, same field schemas. |

**Rules:**
- Each edit must specify concrete field values for a single action.
- Edits that affect a parent naturally cascade to its subtree; this is expected.
- If feedback cannot be expressed within the edit system, return `{ "edits": [] }` with an explanation in `feedback_interpretation`.
- If `confidence: "low"`, prefer fewer, safer edits over aggressive changes.

### 7.6 — Guardrails

- Do NOT attempt full structural rewrites — even if the user implies one ("start over", "redo the whole thing"). Apply the most impactful subset of local edits and explain the limitation.
- If feedback implies major re-architecture (changing the root part, rebuilding the hierarchy), apply only compatible local edits. Note in `feedback_interpretation` what couldn't be done.
- Prefer modifying existing parts over deleting and recreating large subtrees.
- Preserve overall object identity and hierarchy — the user wants to improve the object, not replace it.
- If the user requests something the pipeline handles automatically (connectivity, grounding, scaling), note it in `feedback_interpretation` and skip — the rebuild will handle it.

### 7.7 — UI Integration

- The user sees the current rendered object and submits feedback text via the existing UI.
- The pipeline runs Stage 5 (feedback mode) → Stage 6 → rebuild → re-render.
- The user sees the updated result and can submit more feedback (next iteration of the user loop).
- No UI redesign needed — reuse existing rendering pipeline and projection display.

---

## Coordinate System

- X = left/right
- Y = front/back
- Z = up
- All dimensions are FULL extents (not radii or half-extents)
- Grid: 100x100x100 (width x depth x height)

---

## LLM Layer (llm.py)

### Model
Gemini 2.5 Pro via `google-genai` SDK.

### Prompt Architecture: Strategy-Prescriptive

The SYSTEM_PROMPT uses a two-layer design:

**Layer 1 — Behavioral (top):** Strategy, archetypes, voxel traps, conflict resolution
**Layer 2 — Contract (bottom):** Schema, constraints, color palette, attachment mapping

The LLM generates a `_strategy` object FIRST as a binding contract:

```json
{
  "_strategy": {
    "archetype": "Vertical/Linear | Horizontal/Splayed | Boxy/Volumetric | Organic/Symbolic",
    "iconic_profile": "Front | Top | Side",
    "voxel_hazards": ["Blob", "Pancake", "Thin loss"],
    "layer_splay_method": "Depth layering | Radial splaying | Vertical stacking | Combination",
    "axis_mapping": "e.g. preserve X/Z, separate along Y"
  },
  "part_types": [...],
  "instances": [...]
}
```

**Why geometric archetypes instead of biological categories:** Previous prompts used categories (living being, object, structure, character) which caused rule leakage (e.g., spider logic applied to humans). Geometric archetypes are category-agnostic.

### Conflict Resolution Priority

When constraints conflict, resolve in this order:

1. **Iconic Recognition** (highest) — silhouette must be recognizable
2. **Voxel Survival** — no disappearing/collapsed parts
3. **Separation** — prevent merging where possible
4. **Structural Fidelity** (lowest) — symmetry, pose precision, exact proportions

### The Three Voxel Traps

| Trap | Cause | Fix |
|---|---|---|
| Monolith (Blob) | Parts parallel and touching | Separate along perpendicular axis |
| Pancake | No vertical volume | Ensure meaningful Z-axis presence |
| Ghost | Parts too thin (~1 unit) | Critical parts >= 2 units thick |

### Axis Mapping Rule

Separation occurs along the axis perpendicular to the Iconic Profile:
- Front profile: preserve X/Z, separate along Y
- Side profile: preserve Y/Z, separate along X
- Top profile: preserve X/Y, elevate along Z

**Coordinate space:** Axis mapping applies in **pre-rotation space** (the parent's canonical local frame). The LLM outputs offsets and dimensions in this frame. The pipeline's `apply_rotation()` step then transforms the entire subtree consistently, so separation relationships are preserved through rotation.

### Three API Call Types

| Call | Purpose | System instruction | Temperature |
|---|---|---|---|
| generate | Text -> Part JSON | SYSTEM_PROMPT | 0.7 |
| validate | Projections + parts -> edits | REFINE_PROMPT | 0.2 |
| describe_image | Image -> text description | None | 0.3 |

### Context Caching

Large system prompts are cached via Gemini's context caching API to avoid re-sending on every call.

- `_get_or_create_cache()` creates/reuses cached content resources
- Keyed by `(api_key, prompt_hash)` — reused within TTL
- TTL: 1 hour
- SYSTEM_PROMPT: cached (well above 4,096 token minimum for Pro)
- REFINE_PROMPT: falls back to inline (below 4,096 minimum)
- Graceful fallback if cache creation fails

### Instance Expansion Limits

Enforced in `_expand_instances()`:
- Max 20 part_types
- Max 10 instances per part_type
- Max 62 total instances

### Validation Change Budget

Enforced in `_apply_edits()`:

**Structural changes (max 5):**
- `add_part`
- `delete`

**Local edits (max 20):**
- `translate` (mutates `parent_face`, `child_face`, `attachment_offset`, `attachment_offset_v`)
- `rotate` (mutates `rotation`)
- `resize` (mutates `dimensions`, `top_radius`)
- `recolor` (mutates `color_id`)
- `toggle_critical` (flips `critical` flag)

Each granular action counts as 1 against its category limit. Fields not scoped to an action are immutable — `uid`, `type_id`, `parent_part`, `primitive_type` cannot be changed by any action.

### Validation Pipeline Awareness

The REFINE_PROMPT explicitly tells the LLM what the pipeline handles automatically, to prevent wasted edits and oscillation:

**Pipeline-handled (LLM should NOT fix):**
- Connectivity (bridging, pruning)
- Scaling (auto-fit to grid)
- Face-normal alignment
- Ownership conflicts (voxel priority resolution)
- Critical part survival (restoration)

**LLM should focus on:**
- Wrong proportions or dimensions
- Missing or unnecessary parts
- Incorrect attachment face or offset placement
- Poor silhouette or recognizability
- Wrong colors

### LLM Error Handling

Cross-cutting error handling for **all** Gemini calls (Stage 1 decomposition, Stage 5 validation, any future LLM stages). Applied in order — first matching category wins.

#### 1. Transient Errors

HTTP 503, 429, UNAVAILABLE, RESOURCE_EXHAUSTED, DEADLINE_EXCEEDED.

**Response:** Retry with exponential backoff. Max 3 retries, delay = 2^(attempt+1) seconds.

#### 2. Input Token Budget Exceeded

Request itself is too large for Gemini's context window (input token limit).

**Response:** Do NOT send the request. Pipeline should pre-check estimated input size before calling. If over limit:
- Stage 1: Return error to user — prompt/image is too complex.
- Stage 5: Reduce context — drop segmentation images (keep legend only), shorten structural representation, or reduce to 3 key views instead of 6. Retry once with reduced input.

#### 3. Output Truncation

Gemini response cut off mid-token due to output token limit. Detectable as JSON that fails to parse with an unexpected-end-of-input error.

**Response:** Do NOT retry with same input — will hit the same wall.
- Stage 1: Return error to user — object may need fewer parts or simpler description.
- Stage 5: Accept current state without edits. Log warning. The object is usable, just unrefined.

#### 4. Valid JSON + Wrapper Text

Gemini returns correct JSON but wrapped in markdown fences (` ```json ... ``` `) or with commentary text before/after.

**Response:** Strip markdown fences, extract the first complete JSON object. This is common LLM behavior, not an error. No retry needed.

#### 5. Malformed Output

After stripping wrappers (step 4), output is still not valid JSON.

**Response:** Retry once with a stricter system prefix: "You MUST respond with a single raw JSON object. No markdown, no commentary, no explanation." If second attempt also fails, treat as unrecoverable — return error to user.

#### 6. Refusal / Commentary

Gemini returns free text with no JSON at all (e.g., "I cannot decompose this into a tree structure because...").

**Detection:** No `{` found in response, or no parseable JSON object.

**Response:** Do NOT retry — Gemini has made a judgment call and retrying burns tokens for the same result. Surface Gemini's message to the user as the error reason.

#### 7. Partial Compliance

Valid JSON matching the expected schema, but with issues:
- Unknown `uid` references in edits
- Missing required fields on individual edits
- Edit budget exceeded (more edits than allowed)
- Enum values outside allowed set

**Response:** Apply valid edits, skip invalid ones. Log each skipped edit with reason. Do not reject the entire edit set for one bad entry. For budget overruns, apply edits up to the budget limit in order and discard the rest.

#### Summary Table

| Category | Detection | Retry? | Fallback |
|---|---|---|---|
| Transient HTTP | Status code | Yes (3x backoff) | Error to user |
| Input too large | Pre-check token estimate | No | Reduce context or error to user |
| Output truncated | JSON parse: unexpected end | No | Accept current state (Stage 5) or error (Stage 1) |
| JSON + wrapper | Markdown fences / surrounding text | No (strip & parse) | — |
| Malformed output | JSON parse failure | Once (stricter prompt) | Error to user |
| Refusal | No JSON in response | No | Surface Gemini's message to user |
| Partial compliance | Schema validation per-edit | No | Apply valid, skip invalid |

---

## Part Schema (schema.py)

### Part Model (Z=UP)

```
uid:                string (system-generated UUID)
part_name:          string
primitive_type:     "cuboid" | "cylinder" | "ellipsoid" | "cone_frustum"
parent_part:        uid | null (null = root)
parent_face:        "top" | "bottom" | "front" | "back" | "left" | "right" | null
child_face:         "top" | "bottom" | "front" | "back" | "left" | "right" | null
attachment_offset:  float [-1, 1] (primary axis on parent face)
attachment_offset_v: float [-1, 1] (secondary axis on parent face)
rotation:           {rx, ry, rz} degrees (Euler, applied Z -> Y -> X)
dimensions:         {width, depth, height} int (full extents, all > 0, root-normalized)
top_radius:         float | null (cone_frustum only; 0 = full cone)
critical:           boolean
color_id:           string (palette reference)
provenance:         string ("llm", "user", etc.)
```

### Attachment Axis Mapping

| Face | u (primary) axis | v (secondary) axis |
|---|---|---|
| top / bottom | X | Y |
| front / back | X | Z |
| left / right | Y | Z |

offset = -1 maps to one edge, 0 = center, +1 = opposite edge.

### Primitive Interpretation

| Type | width | depth | height | Notes |
|---|---|---|---|---|
| cuboid | X extent | Y extent | Z extent | Standard box |
| cylinder | base diameter | base diameter | Z axis length | Circular (width ~ depth) |
| ellipsoid | X diameter | Y diameter | Z diameter | Semiaxes = dim/2 |
| cone_frustum | base diameter | base diameter | Z axis length | top_radius on Part |

---

## Pipeline (pipeline.py)

### 12-Stage Part-World Pipeline

All stages execute in strict order via `build_part_world()`. Do NOT call individually in production.

| # | Stage | Purpose |
|---|---|---|
| 1 | `validate_graph` | Fix dangling parent refs, break cycles |
| 2 | `enforce_critical_closure` | Promote ancestors of critical parts to critical |
| 3 | `enforce_single_root` | Select one root (prefer critical + largest volume); reattach extras |
| 4 | `attach_parts` | Compute world-space centers + anchors via hierarchy |
| 5 | `compute_scale` | Derive uniform scale factor with 0.5x rotation buffer |
| 6 | `apply_scale` | Scale dims, centers, anchors uniformly |
| 7 | `final_placement` | Center model at (25,25), ground at z=0 |
| 8 | `apply_rotation` | Hierarchical scene-graph rotations (Z->Y->X); sets world_R |
| 9 | `voxelize` | Rasterize at 2x, downsample to 1x; max 62 parts |
| 10 | `apply_ownership` | Resolve contested voxels to single owners |
| 11 | `critical_restoration` | Restore one voxel for vanished critical parts |
| 12 | `enforce_connectivity` | Bridge disconnected critical parts; prune orphans |

Output: `(100, 100, 100)` int32 ndarray. `0 = empty`, `n = states[n-1]` owns voxel.

### Rotation

- Order: Z -> Y -> X (R = Rx @ Ry @ Rz)
- Hierarchical: children inherit all ancestor transforms
- Applied AFTER placement (pipeline step 8, after steps 4-7)
- Root rotates around its own center
- Children rotate around their attachment anchor

### Ownership Resolution Priority

When multiple parts claim the same voxel:
1. Descendant > ancestor
2. Critical > non-critical
3. Volume rule: smaller part wins if < 20% of larger
4. Smallest UID tiebreak

### Voxelization

- 2x bitmask grid, downsampled to 1x
- Max 62 parts (uint64 bitmask limit)

---

## Color Palette

20 fixed colors. LLM must use exact `color_id` strings.

| color_id | Hex |
|---|---|
| black | #1B2A34 |
| white | #F4F4F4 |
| dark_bluish_gray | #6B7280 |
| light_bluish_gray | #A8AFB8 |
| red | #C91A09 |
| dark_red | #720E0F |
| blue | #0055BF |
| light_blue | #5A93DB |
| yellow | #F2CD37 |
| orange | #D67923 |
| green | #237841 |
| dark_green | #184632 |
| tan | #E4CD9E |
| dark_tan | #B0A06F |
| brown | #582A12 |
| dark_brown | #352100 |
| pink | #FC97AC |
| purple | #D3359D |
| teal | #069D9F |
| lime | #A5CA18 |

Common aliases are auto-mapped: "grey" -> dark_bluish_gray, "gold" -> yellow, "beige" -> tan, etc.
Unknown colors are mapped to nearest palette entry by Euclidean RGB distance.

---

## Stage 8 — Finalization + Display

**Purpose:** Lock the approved state, render the final 3D view, and serve it to the user. This is the terminal stage — no further pipeline processing after this.

**Execution:** Code

### 8.1 — User Flow

The UI presents a simplified two-step interaction:

```
Step 1: User enters description → clicks "Generate"
        Pipeline runs: Stage 0 → 1 → 2 → 3 → 5 (validation) → 6 → rebuild (2→3)
        User sees: draft 3D voxel render

Step 2: User enters feedback → clicks "Refine"
        Pipeline runs: Stage 7 → 5 (feedback mode) → 6 → rebuild (2→3)
        User sees: updated 3D voxel render

User can repeat Step 2 or accept the result.
```

No intermediate steps exposed (no separate "generate parts" and "run pipeline" buttons). One click per stage.

### 8.2 ��� Finalization

When the user accepts the result (or after the final rebuild if no feedback):

1. **Lock state** — mark the canonical representation as final (no further edits)
2. **Export voxel grid** — the 100×100×100 ownership grid as the definitive output
3. **Export metadata** — part list, accepted/rejected edits history, final stats
4. **Render** — serve the 3D view via Three.js (existing renderer)

### 8.3 — Display

The Three.js renderer displays:
- **3D voxel model** — interactive (rotate, zoom, pan)
- **Feedback input** — text field + "Refine" button (available until user accepts)
- **Parts list** — collapsible panel showing part names, types, colors (visible on demand, not required for basic flow)

Debug mode (toggle) additionally shows:
- Voxel count, part count, scale factor
- Per-part voxel counts
- Accepted/rejected edit history

### 8.4 — Export Formats (v1)

- **Voxel JSON** — `[{x, y, z, color_id, uid}]` for all occupied voxels
- **Metadata JSON** — canonical `part_types` + `instances` + build stats

Future formats (not v1): OBJ/STL mesh export, LDraw parts list, shareable URL.

---

## API Endpoints (server.py)

### POST /api/generate

Full pipeline: description → draft voxel render.

- Input: `{ description: string, image?: base64, api_key?: string }`
- Output:
  ```json
  {
    "voxels": [{"x": int, "y": int, "z": int, "color_id": string, "uid": string}],
    "parts": [{ part metadata }],
    "stats": { "total_occupied": int, "grid_size": [100,100,100], "part_count": int },
    "validation": { "guess": string, "confidence": string, "issues": [...], "edits_applied": int, "edits_rejected": int },
    "session_id": string
  }
  ```
- Runs: Stage 0 → 1 → 2 → 3 → 5 (validation) → 6 → rebuild (2→3)
- Uses GEMINI_API_KEY env var if no api_key provided
- Returns `session_id` for subsequent feedback calls

### POST /api/feedback

Apply user feedback → updated voxel render.

- Input: `{ session_id: string, feedback: string, api_key?: string }`
- Output:
  ```json
  {
    "voxels": [{"x": int, "y": int, "z": int, "color_id": string, "uid": string}],
    "parts": [{ part metadata }],
    "stats": { "total_occupied": int, "grid_size": [100,100,100], "part_count": int },
    "feedback_interpretation": string,
    "confidence": string,
    "edits_applied": int,
    "edits_rejected": int,
    "session_id": string
  }
  ```
- Runs: Stage 5 (feedback mode) → 6 → rebuild (2→3)
- Requires `session_id` from a prior `/api/generate` call (carries forward the canonical representation)

### POST /api/export

Export finalized model.

- Input: `{ session_id: string, format?: "voxel_json" | "metadata_json" }`
- Output: Requested export format (default: voxel_json)

---

## Telemetry

### 8.5 — Pipeline Telemetry

Every pipeline run logs structured telemetry for diagnostics, cost tracking, and quality analysis.

#### Per-Request Log (`pipeline_log.jsonl`)

```json
{
  "session_id": "string",
  "timestamp": "ISO 8601",
  "endpoint": "/api/generate | /api/feedback",
  "description_length": int,
  "modality": "text | image | both",

  "stages": {
    "stage_1": { "latency_ms": int, "success": bool, "error": null },
    "stage_2": { "latency_ms": int, "parts_in": int, "parts_out": int },
    "stage_3": { "latency_ms": int, "occupied_voxels": int, "overlaps_resolved": int, "bridges_added": int },
    "stage_5": { "latency_ms": int, "mode": "validation | feedback", "confidence": "string", "edits_proposed": int },
    "stage_6": { "latency_ms": int, "edits_accepted": int, "edits_rejected": int, "rejection_reasons": ["string"] },
    "rebuild":  { "latency_ms": int }
  },

  "totals": {
    "end_to_end_latency_ms": int,
    "total_llm_calls": int,
    "total_tokens": int,
    "part_count": int,
    "occupied_voxels": int,
    "grid_utilization": float
  }
}
```

#### Per-LLM-Call Log (`usage_log.jsonl`)

Every Gemini call logs:

```json
{
  "session_id": "string",
  "call_type": "generate | describe_image | validate | feedback",
  "timestamp": "ISO 8601",
  "input_tokens": int,
  "output_tokens": int,
  "thinking_tokens": int,
  "cached_tokens": int,
  "total_tokens": int,
  "retries": int,
  "latency_ms": int,
  "success": bool,
  "error": "string | null"
}
```

Log line: `[call_type] session=ID tokens=N (in=N out=N thinking=N cached=N) retries=N latency=Nms ok=bool`

#### Aggregate Metrics (derived, not logged per-request)

For dashboarding / periodic analysis:

| Metric | Description |
|---|---|
| Requests per hour/day | Usage volume |
| Token cost per request | Average Gemini spend |
| P50/P95 end-to-end latency | Performance |
| Edit acceptance rate | Stage 6 quality (accepted / proposed) |
| Feedback rounds per session | How many iterations users need |
| Confidence distribution | How often validation/feedback is high/medium/low |
| Error rate by category | Which LLM error types are most common |
| Grid utilization | Average occupied voxels / 1M (quality signal) |

### Gemini Token Billing

| Bucket | Source | Billing |
|---|---|---|
| Prompt tokens | Input: prompt + system instruction + files + history | Input rate |
| Candidates tokens | Visible output text | Output rate |
| Thinking tokens | Hidden reasoning (Gemini 2.5 Pro) | Output rate |
| Cached tokens | Served from context cache | Reduced storage rate |

### Cost Optimization (Remaining)

1. **Thinking Budget** — `thinking_level` parameter to cap reasoning spend per call type
2. **Flash Models** — `gemini-2.0-flash` for simple tasks (describe_image) to avoid thinking overhead
3. **REFINE_PROMPT caching** — Currently below 4,096 token minimum; could pad or accept inline cost

---

## Design Decisions

### `_strategy` is chain-of-thought, not a runtime contract

The `_strategy` object is generated by the LLM but intentionally ignored by the pipeline code. It exists as a prompt engineering device — forcing the LLM to reason about spatial layout (archetype, profile, hazards, axis mapping) before emitting geometry. The pipeline only consumes `part_types` and `instances`.

Future option: add a lightweight pre-pipeline check that detects major strategy-geometry mismatches (e.g., strategy says "separate along Y" but all offsets are zero). Not currently implemented.

### Thickness rule + critical restoration = defense in depth

Two systems address thin/vanishing critical parts:
- **Prompt rule:** "Critical parts >= 2 units thick" (preventive — tells LLM to avoid the problem)
- **Pipeline stage 11:** `critical_restoration()` adds a single stub voxel at `floor(center)` if a critical part has zero voxels (last resort)

These are complementary, not conflicting. The prompt rule reduces how often restoration fires. Restoration ensures nothing silently disappears even if the LLM fails. Removing either weakens the system.

### Validation loop is scoped to avoid fighting the pipeline

The REFINE_PROMPT explicitly lists what the pipeline handles (connectivity, scaling, face-normal alignment, ownership, critical survival) and instructs the LLM not to fix those. Edits are scoped to what the pipeline cannot fix: proportions, missing parts, placement, silhouette, colors.

### Axis mapping operates in pre-rotation space

The LLM's axis mapping rules and attachment offsets are in the parent's canonical local frame. The pipeline's `apply_rotation()` step transforms the entire subtree consistently after placement, preserving separation relationships through rotation. This is correct and intentional — the LLM does not need to reason about post-rotation coordinates.
