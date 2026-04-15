"""
LEGO output schema.

Two coordinate systems exist in this file. They are strictly isolated:

  Shape / Position  — legacy model. Y=UP (plates along Y).
                      Used only by the original pipeline stages (load_image,
                      decompose_into_primitives, build_model).
                      Do NOT use these types in Part-based pipeline code.

  Part              — V1 spec model. Z=UP, X=left/right, Y=front/back.
                      All Part geometry (dimensions, attachment, rotation)
                      is expressed in this coordinate system throughout.
                      Do NOT mix Part geometry with Shape/Position types.

There is no reconciliation step between the two systems. They are parallel
representations used by different pipeline stages.
"""
from __future__ import annotations
import uuid
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

# Rotation is always a cardinal multiple of 90° around the Y axis (top-down).
# 0° = default orientation as described in each shape's dimension comments.
Rotation = Literal[0, 90, 180, 270]


BASEPLATE_WIDTH = 50  # studs (legacy Shape pipeline)
BASEPLATE_DEPTH = 50  # studs (legacy Shape pipeline)

# Part-world grid (spec: 100×100×100)
GRID_SIZE = 100


class PrimitiveType(str, Enum):
    CUBOID = "cuboid"
    CYLINDER = "cylinder"
    ELLIPSOID = "ellipsoid"
    CONE_FRUSTUM = "cone_frustum"
    WEDGE = "wedge"  # legacy Shape only; not valid in Part


class Position(BaseModel):
    x: int = Field(..., description="Studs from left edge of baseplate")
    y: int = Field(..., description="Plates from baseplate surface")
    z: int = Field(..., description="Studs from front edge of baseplate")


class CuboidDimensions(BaseModel):
    width: int = Field(..., ge=1, description="Studs along X")
    height: int = Field(..., ge=1, description="Plates along Y")
    depth: int = Field(..., ge=1, description="Studs along Z")


class CylinderDimensions(BaseModel):
    """
    Circular cross-section when depth_studs is None (diameter applies to both X and Z).
    Set depth_studs to a different value for an elliptical cross-section
    (diameter = extent along X, depth_studs = extent along Z).
    """
    diameter: int = Field(..., ge=1, description="Studs along X")
    height: int = Field(..., ge=1, description="Plates along Y")
    depth_studs: Optional[int] = Field(
        None, ge=1, description="Studs along Z; None means symmetric (circular)"
    )


class WedgeDimensions(BaseModel):
    width: int = Field(..., ge=1, description="Studs along X")
    height: int = Field(..., ge=1, description="Plates along Y")
    depth: int = Field(..., ge=1, description="Studs along Z")
    direction: Literal["left", "right", "front", "back"] = "right"


class Shape(BaseModel):
    id: str
    type: PrimitiveType
    position: Position
    dimensions: CuboidDimensions | CylinderDimensions | WedgeDimensions
    rotation: Rotation = Field(0, description="Degrees clockwise around Y axis")
    color: str = Field(..., description="Hex color, e.g. #FF0000")
    label: str = Field("", description="Semantic label, e.g. 'wheel', 'roof'")


class BoundingBox(BaseModel):
    x: int
    y: int
    z: int
    width: int   # studs
    height: int  # plates
    depth: int   # studs


class LegoModel(BaseModel):
    source_image: str = Field("", description="Original filename or path")
    baseplate: dict = Field(
        default={"width": BASEPLATE_WIDTH, "depth": BASEPLATE_DEPTH},
        description="Fixed 50x50 stud baseplate",
    )
    bounding_box: BoundingBox
    shapes: list[Shape]


# ---------------------------------------------------------------------------
# Part-world types  (Z=UP coordinate system — spec §5)
# ---------------------------------------------------------------------------
# All types below use: X = left/right, Y = front/back, Z = up.
# Do NOT mix these with Shape, Position, or any legacy Y=UP type above.

# Valid faces on a bounding cuboid for attachment (2-face model).
Face = Literal["top", "bottom", "front", "back", "left", "right"]
# Legacy alias
AttachFace = Face


class PartDimensions(BaseModel):
    """
    Full extents of a part in its canonical local frame (X=width, Y=depth, Z=height).

    Primitive interpretation:
      cuboid       — full box extents
      ellipsoid    — full diameters; internal semiaxes = dimension / 2
      cylinder     — width/depth = base diameter (V1: circular, so width ≈ depth);
                     height = axis length along Z
      cone_frustum — width/depth = base diameter; height = axis length along Z;
                     top_radius stored on Part (0 or None = full cone)

    All values are FULL extents, never radii or half-extents.
    Half-extents are always derived internally as dimension / 2.
    """
    width: float = Field(..., gt=0, description="Full extent along X (left/right)")
    depth: float = Field(..., gt=0, description="Full extent along Y (front/back)")
    height: float = Field(..., gt=0, description="Full extent along Z (up)")


class Rotation3D(BaseModel):
    """
    Euler angles in degrees. Applied in Z → Y → X order (spec §15).

    Rotation is hierarchical: each child inherits all ancestor transforms
    before applying its own rotation.

    Pivot:
      root  — rotates around its own center
      child — rotates around its attachment anchor after that anchor has been
              transformed by all ancestor rotations (spec §4)
    """
    rx: float = Field(0.0, description="Degrees around X axis (applied last)")
    ry: float = Field(0.0, description="Degrees around Y axis (applied second)")
    rz: float = Field(0.0, description="Degrees around Z axis (applied first)")


class Part(BaseModel):
    """
    Hierarchical part in Z=UP space (spec §4).

    uid is system-generated and must not be changed after creation.
    parent_part references the uid of the parent; None means root.
    parent_face is the face of the PARENT bounding cuboid this part attaches to.
    child_face is the face of the CHILD that contacts the parent face.
    attachment_offset shifts the anchor along the parent face: -1 = one edge,
      0 = center, +1 = opposite edge.
    top_radius applies only to cone_frustum: radius of the top face; 0 or None = full cone.
    critical parts are restored post-voxelization if they voxelize to zero.
    color_id references a fixed palette entry (palette defined separately).
    provenance records the source of this part, e.g. "llm" or "user".
    """
    uid: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Immutable system-generated identifier",
    )
    part_name: str
    primitive_type: Literal[
        PrimitiveType.CUBOID,
        PrimitiveType.CYLINDER,
        PrimitiveType.ELLIPSOID,
        PrimitiveType.CONE_FRUSTUM,
    ]
    parent_part: str | None = Field(None, description="uid of parent; None = root")
    parent_face: Face | None = Field(None, description="Face of the parent this part attaches to")
    child_face: Face | None = Field(None, description="Face of this part that contacts the parent")
    attachment_offset: float = Field(
        0.0, ge=-1.0, le=1.0,
        description="Anchor offset along parent face primary axis: -1/0/+1 = edge/center/edge",
    )
    attachment_offset_v: float = Field(
        0.0, ge=-1.0, le=1.0,
        description="Anchor offset along parent face secondary axis: -1/0/+1 = edge/center/edge",
    )

    rotation: Rotation3D = Field(default_factory=Rotation3D)
    dimensions: PartDimensions
    top_radius: Optional[float] = Field(
        None, ge=0.0,
        description="Top radius for cone_frustum; 0 or None = full cone. Ignored for other types.",
    )
    critical: bool = False
    color_id: str = Field("", description="Fixed palette color reference")
    provenance: str = Field("", description="Origin of this part, e.g. 'llm', 'user'")
