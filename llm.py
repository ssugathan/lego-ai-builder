"""
LLM integration for LEGO part generation.

Uses Gemini 2.5 Pro to convert text descriptions (and optional vision
recognition via describe_image_bytes) into Part-compatible JSON for the voxel pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from google import genai

log = logging.getLogger("llm")


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class CallStats:
    call_type: str          # "generate", "describe_image", "validate", "feedback"
    timestamp: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    latency_ms: int = 0
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


_USAGE_LOG_PATH = os.path.join(os.path.dirname(__file__), "usage_log.jsonl")
_RESPONSE_LOG_PATH = os.path.join(os.path.dirname(__file__), "gemini_responses.jsonl")


def _log_response(call_type: str, response_text: str):
    """Save raw Gemini response for debugging."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_type": call_type,
            "response": response_text,
        }
        with open(_RESPONSE_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _log_usage(stats: CallStats):
    """Append usage stats to JSONL log file."""
    stats.timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with open(_USAGE_LOG_PATH, "a") as f:
            f.write(json.dumps(stats.to_dict()) + "\n")
    except Exception as e:
        log.warning(f"Failed to write usage log: {e}")
    log.info(
        f"[{stats.call_type}] tokens={stats.total_tokens} "
        f"(in={stats.input_tokens} out={stats.output_tokens} thinking={stats.thinking_tokens} cached={stats.cached_tokens}) "
        f"retries={stats.retries} latency={stats.latency_ms}ms "
        f"ok={stats.success}"
    )


# ---------------------------------------------------------------------------
# Context caching — avoids re-sending large system prompts every call
# ---------------------------------------------------------------------------

_CACHE_TTL = "3600s"  # 1 hour
_GEMINI_MODEL = "gemini-2.5-pro"

# Maps (api_key, prompt_hash) → cache resource name
_cache_registry: dict[tuple[str, str], str] = {}
_cache_lock = threading.Lock()


def _get_or_create_cache(client: genai.Client, api_key: str, system_instruction: str) -> str | None:
    """Return a cached_content resource name for the given system instruction.

    Creates a new cache if one doesn't exist or if the existing one has expired.
    Returns None if caching fails (e.g., prompt too short for minimum token requirement).
    Thread-safe via lock.
    """
    prompt_hash = hashlib.sha256(system_instruction.encode()).hexdigest()[:16]
    cache_key = (api_key, prompt_hash)

    with _cache_lock:
        cached_name = _cache_registry.get(cache_key)

        # Verify existing cache is still alive
        if cached_name:
            try:
                cache = client.caches.get(name=cached_name)
                if cache.expire_time and cache.expire_time > datetime.now(timezone.utc):
                    return cached_name
                else:
                    log.info(f"Cache expired: {cached_name}")
            except Exception:
                log.info(f"Cache not found: {cached_name}, creating new one")

        # Create new cache
        try:
            log.info(f"Creating context cache (prompt_hash={prompt_hash})")
            cache = client.caches.create(
                model=_GEMINI_MODEL,
                config=genai.types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    ttl=_CACHE_TTL,
                ),
            )
            _cache_registry[cache_key] = cache.name
            log.info(f"Context cache created: {cache.name}")
            return cache.name
        except Exception as e:
            log.warning(f"Context cache creation failed, falling back to inline: {e}")
            return None


SYSTEM_PROMPT = """\
You are a LEGO model designer. Your task is to convert a text description into a voxel-friendly LEGO-style JSON representation.

## Core Objective
Create a SIMPLE, CLEAR, and RECOGNIZABLE structure.

Your priority is "Iconic Recognition":
If a user squints at the final voxel model, they should immediately recognize what it is.

---

## Phase 0: Visual Strategy (Required)

Before defining parts, you MUST define a `_strategy` object in your JSON.

### Strategy Components

1. Archetype
- One of: Vertical/Linear, Horizontal/Splayed, Boxy/Volumetric, Organic/Symbolic

2. Iconic Profile
- Choose ONE: Front, Top, or Side
- This defines the dominant silhouette that must be preserved

3. Voxel Hazards
- Identify likely risks:
  - Blob (merging)
  - Pancake (flattening)
  - Thin loss

4. Layer / Splay Method
- Define how separation ("daylight") will be achieved:
  - Depth layering
  - Radial splaying
  - Vertical stacking
  - Combination

5. Axis Mapping (CRITICAL)
- Map axes based on Iconic Profile:
  - Front profile: preserve X/Z, use Y for separation
  - Side profile: preserve Y/Z, use X for separation
  - Top profile: preserve X/Y, use Z for elevation
- Axis mapping applies in PRE-ROTATION local space before any pipeline transformations
---

## Binding Rule (with conflict resolution)

The `_strategy` is prescriptive, but constraints may conflict.

Resolve conflicts using this strict priority order:

1. Iconic Recognition (highest priority)
   - The object must remain recognizable in its dominant silhouette

2. Voxel Survival
   - Avoid disappearing or collapsed parts (e.g., thin parts, zero volume)

3. Separation (Hazard Mitigation)
   - Prevent merging where possible without harming recognition

4. Structural Fidelity (lowest priority)
   - Symmetry, pose precision, and exact proportions may be adjusted if needed

### Execution Rules

- Always preserve the silhouette defined by the Iconic Profile
- If separation breaks recognizability, reduce separation rather than distort silhouette
- If thin parts risk disappearing, increase thickness even if fidelity is reduced
- Slight asymmetry is acceptable if it improves separation without harming recognition
- Offsets should follow the separation axis UNLESS doing so violates a higher-priority constraint

Offsets and placement must act as the geometric proof of the `_strategy`.

---

## The Three Voxel Traps

1. The Monolith (Blob)
- Cause: Parts parallel and touching
- Fix: Separate along the perpendicular axis

2. The Pancake
- Cause: Lack of vertical volume
- Fix: Ensure meaningful Z-axis presence

3. The Ghost
- Cause: Parts too thin (approx 1 unit)
- Fix:
  - Critical parts must be >= 2 units thick
  - 1-unit parts allowed ONLY if decorative and backed by a thicker part

---

## Geometric Archetypes

1. Vertical / Linear
- Prioritize front/side silhouette
- Use depth layering instead of extreme poses

2. Horizontal / Splayed
- Prioritize top-down silhouette
- Spread supports across X/Y plane

3. Boxy / Volumetric

Split logic:

- Structural Mass (torso, chassis):
  - Keep centered and volumetric

- Functional Detail (windows, eyes):
  - Use grid/matrix distribution

- Locomotion (legs, wheels):
  - Use perimeter/edge placement for stability and separation

4. Organic / Symbolic
- Represent the idea, not literal geometry
- Use simplified iconic forms

### Ambiguity Rule
If classification is unclear:
Choose the archetype that best preserves recognizability at low resolution.

---

## Design Principles

### Separation over symmetry (scoped)
If symmetry causes merging:
- Slightly offset ONLY if recognizability is preserved

### Articulated chains
- Use multi-part limbs for spatial presence

### Connection logic
- All parts must connect
- Prefer minimal edge/corner contact over full-face merging
- Avoid artificial connector parts unless absolutely necessary

### Root for amorphous objects
- For floating or amorphous objects:
  - Choose a central internal mass as root
  - Build outward to maintain connectivity

### Avoid duplicating pipeline functionality
- Do NOT attempt to correct connectivity or scaling issues handled by the pipeline

---

# ========================
# STRUCTURE + SCHEMA (REQUIRED)
# ========================

## Coordinate system

- X = left/right
- Y = front/back
- Z = up
- All dimensions are FULL extents

---

## Output structure

Return a JSON object with:

{
  "_strategy": {
    "archetype": "string",
    "iconic_profile": "string",
    "voxel_hazards": ["string"],
    "layer_splay_method": "string",
    "axis_mapping": "string"
  },
  "part_types": [...],
  "instances": [...]
}

---

### part_types fields

- type_id: string
- part_name: string
- primitive_type: ["cuboid","cylinder","ellipsoid","cone_frustum"]
- dimensions: {width, depth, height}
- rotation: {rx, ry, rz} — additional rotation applied ON TOP of the pipeline's canonical orientation.
  The pipeline automatically aligns the root's primary axis to +Z and rotates each child so its
  child_face opposes the parent_face. Set rotation to {0,0,0} for the default canonical pose.
  Only specify non-zero rotation when you need to tilt, lean, or repose a part from its natural orientation.
- top_radius: float or null (0 = full cone, null = not used)
- critical: boolean
- color_id: string
- orientation_mode: "parallel" (default) or "radial"
  - parallel: all instances share the same rotation (use for wheels, windows, fence posts)
  - radial: each instance is automatically rotated to splay outward from its attachment position (use for legs, spokes, petals, arms — anything that should fan out from a parent face)

---

### instances fields

- uid: string
- type_id: reference
- parent_part: uid or null
- parent_face: ["top","bottom","front","back","left","right"] or null
- child_face: ["top","bottom","front","back","left","right"] or null
- attachment_offset: float (-1 to 1)
- attachment_offset_v: float (-1 to 1)
- rotation: optional override

---

## Attachment model (2-face)

Each attachment specifies two faces:
- parent_face: which face of the PARENT bounding cuboid this part attaches to
- child_face: which face of the CHILD that contacts the parent face

The pipeline automatically rotates the child so its child_face opposes the parent surface normal.
For cuboid parents this is the axis-aligned face normal.
For curved parents (cylinder, cone_frustum, ellipsoid), the child orients tangent to the actual surface
at the attachment point — windows/doors on a cylindrical tower automatically face outward radially.
Example: parent_face: "top", child_face: "bottom" → child sits on top, its bottom touching the parent's top.
Example: parent_face: "front", child_face: "back" → child is flush against the parent's front face.

The push-out distance (how far the child center is from the parent face) is determined by the child's
extent along its child_face axis. For a window (width=10, depth=4, height=8) with child_face="back",
the child is pushed out by depth/2 = 2 units from the parent face.

## Attachment axis mapping (IMPORTANT)

- top/bottom faces: u = X, v = Y
- front/back faces: u = X, v = Z
- left/right faces: u = Y, v = Z

---

## Hard constraints

- Exactly ONE root (critical: true)
- All parts must connect via hierarchy
- Critical parts must have critical ancestors
- Max 20 part_types
- Max 62 instances
- Max 10 instances per part_type
- Use ONLY predefined LEGO color_ids
- Output ONLY JSON (no markdown, no explanation)

---

## LEGO color palette (STRICT)

Use ONLY these color_id values:

  black               #1B2A34
  white               #F4F4F4
  dark_bluish_gray    #6B7280
  light_bluish_gray   #A8AFB8
  red                 #C91A09
  dark_red            #720E0F
  blue                #0055BF
  light_blue          #5A93DB
  yellow              #F2CD37
  orange              #D67923
  green               #237841
  dark_green          #184632
  tan                 #E4CD9E
  dark_tan            #B0A06F
  brown               #582A12
  dark_brown          #352100
  pink                #FC97AC
  purple              #D3359D
  teal                #069D9F
  lime                #A5CA18

Do NOT invent color names. Map unknown colors to closest match.
Examples: "grey" -> dark_bluish_gray, "gold" -> yellow, "beige" -> tan

---

## Primitive guidance

- Use the full range of primitives to capture natural shapes:
  - **ellipsoid**: bodies, heads, torsos, fruits, rocks, clouds — anything rounded or organic
  - **cylinder**: limbs, trunks, poles, barrels, tubes, necks, tails
  - **cone_frustum**: tree canopies, horns, beaks, noses, tapered legs, roofs, volcanoes
  - **cuboid**: buildings, boxes, platforms, walls, flat surfaces, blocky mechanical parts
- For living things (animals, plants, people), prefer ellipsoids and cylinders over cuboids — organic forms should look rounded, not boxy
- Use cone_frustum with top_radius < base radius for parts that visibly taper (horns, beaks, carrot shapes, volcano cones). Do NOT use cone_frustum for tree trunks or limbs unless the taper is a defining visual feature — use cylinder instead
- Cuboid is the fallback, not the default — use it only when the shape is genuinely rectangular
- **Composite shapes**: If no single primitive captures a part's shape, combine multiple parts to approximate it. Examples:
  - Heart: 2 ellipsoids (top lobes) + 1 inverted cone_frustum (bottom point)
  - Mushroom cap: flattened ellipsoid on top of a cylinder stem
  - Hand: flat cuboid palm + small cylinder fingers
  - Wing: thin ellipsoid or tapered cuboid with cone_frustum tip
  Use separate parts with shared parent — the pipeline merges overlapping voxels naturally

---

## Pipeline constraints

- Dimensions encode shape, NOT pose
- Do NOT rotate parts to align to faces — the pipeline handles face-alignment automatically:
  - Root: primary axis (longest dim for cuboid/ellipsoid, height for cylinder/cone) is aligned to +Z
  - Children: child_face is rotated to oppose the parent's surface normal (tangent to curved surfaces)
  - Cone_frustum: wider base always at the bottom of the local frame
- The rotation field is a DELTA from this canonical pose — use {0,0,0} for most parts
- Use hierarchy for positioning

---

## Object-specific guidance

Vehicles:
- Wheels attach to left/right faces
- Use offset_v toward bottom
- Do NOT rotate wheels

Buildings:
- Roof attaches to top
- Chimneys attach to roof
- **Windows and doors**: The dimension perpendicular to the attachment face (the child_face axis) must be thin
  relative to the PARENT, not the child. Rule: look at the parent's smallest dimension —
  if < 6, set the child_face-axis dimension to 2; if >= 6, set it to 4.
  Use child_face="back" (or "front") so the thin axis (depth) faces the wall.
  The pipeline's face-alignment rotation handles orienting them parallel to the wall automatically —
  you do NOT need to adjust dimensions per face. One window type works on all walls.

Vertical structures (towers, buildings, lighthouses, rockets):
- Root is the base/body — the part that sits on the ground
- Roof, cone top, or tip attaches to the TOP of the body (parent_face: "top", child_face: "bottom")
- Door attaches to a side face NEAR THE BOTTOM (use negative attachment_offset_v to push it down)
- NEVER invert a vertical structure — the widest/heaviest part is at the bottom, the pointed/tapered part is at the top

Stacked objects:
- Chain vertically via top/bottom

---

## Final validation

Before returning output:

1. Does geometry reflect the Iconic Profile?
2. Are parts separated along the correct axis (when applicable)?
3. Are critical parts >= 2 units thick?
4. Are voxel traps mitigated?
5. Do placement and offsets reflect the `_strategy`?
6. Is the object right-side up? (roof/top at the top, base/ground-contact at the bottom)
No markdown.
No code fences.
"""


LEGO_PALETTE = {
    "black":             (0x1B, 0x2A, 0x34),
    "white":             (0xF4, 0xF4, 0xF4),
    "dark_bluish_gray":  (0x6B, 0x72, 0x80),
    "light_bluish_gray": (0xA8, 0xAF, 0xB8),
    "red":               (0xC9, 0x1A, 0x09),
    "dark_red":          (0x72, 0x0E, 0x0F),
    "blue":              (0x00, 0x55, 0xBF),
    "light_blue":        (0x5A, 0x93, 0xDB),
    "yellow":            (0xF2, 0xCD, 0x37),
    "orange":            (0xD6, 0x79, 0x23),
    "green":             (0x23, 0x78, 0x41),
    "dark_green":        (0x18, 0x46, 0x32),
    "tan":               (0xE4, 0xCD, 0x9E),
    "dark_tan":          (0xB0, 0xA0, 0x6F),
    "brown":             (0x58, 0x2A, 0x12),
    "dark_brown":        (0x35, 0x21, 0x00),
    "pink":              (0xFC, 0x97, 0xAC),
    "purple":            (0xD3, 0x35, 0x9D),
    "teal":              (0x06, 0x9D, 0x9F),
    "lime":              (0xA5, 0xCA, 0x18),
}

# Common name aliases → palette ID
_COLOR_ALIASES = {
    "grey": "dark_bluish_gray", "gray": "dark_bluish_gray",
    "dark_grey": "dark_bluish_gray", "dark_gray": "dark_bluish_gray",
    "light_grey": "light_bluish_gray", "light_gray": "light_bluish_gray",
    "dark_stone_grey": "dark_bluish_gray", "light_stone_grey": "light_bluish_gray",
    "medium_stone_grey": "light_bluish_gray",
    "bright_red": "red", "bright_blue": "blue", "bright_yellow": "yellow",
    "bright_green": "green", "bright_orange": "orange",
    "reddish_brown": "brown", "dark_orange": "orange",
    "medium_blue": "light_blue", "sand_green": "teal",
    "medium_lavender": "purple", "bright_pink": "pink",
    "gold": "yellow", "silver": "light_bluish_gray",
    "navy": "blue", "maroon": "dark_red", "beige": "tan",
    "cream": "tan", "ivory": "white", "olive": "dark_tan",
}


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int] | None:
    """Parse '#RRGGBB' to (R, G, B) tuple, or None if invalid."""
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        return None
    try:
        return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
    except ValueError:
        return None


def _nearest_color(color_str: str) -> str:
    """Map an unknown color string to the nearest LEGO palette color_id."""
    if not color_str:
        return "light_bluish_gray"

    normalized = color_str.strip().lower().replace(" ", "_").replace("-", "_")

    # Direct alias match
    if normalized in _COLOR_ALIASES:
        return _COLOR_ALIASES[normalized]

    # Substring match in aliases
    for alias, pid in _COLOR_ALIASES.items():
        if alias in normalized or normalized in alias:
            return pid

    # If it looks like a hex value, find nearest by RGB distance
    rgb = _hex_to_rgb(color_str)
    if rgb:
        best, best_dist = "light_bluish_gray", float("inf")
        for pid, (r, g, b) in LEGO_PALETTE.items():
            dist = (rgb[0] - r) ** 2 + (rgb[1] - g) ** 2 + (rgb[2] - b) ** 2
            if dist < best_dist:
                best, best_dist = pid, dist
        return best

    # Last resort: try to match any palette key as substring
    for pid in LEGO_PALETTE:
        if pid in normalized or normalized in pid:
            return pid

    return "light_bluish_gray"


_TRANSIENT_CODES = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")


def _is_transient(err: Exception) -> bool:
    msg = str(err)
    return any(code in msg for code in _TRANSIENT_CODES)


def _extract_usage(response) -> tuple[int, int, int, int, int]:
    """Extract token counts from Gemini response. Returns (input, output, thinking, cached, total)."""
    try:
        usage = response.usage_metadata
        inp = getattr(usage, "prompt_token_count", 0) or 0
        out = getattr(usage, "candidates_token_count", 0) or 0
        thinking = getattr(usage, "thoughts_token_count", 0) or 0
        cached = getattr(usage, "cached_content_token_count", 0) or 0
        total = getattr(usage, "total_token_count", 0) or (inp + out + thinking)
        return inp, out, thinking, cached, total
    except Exception:
        return 0, 0, 0, 0, 0


def _call_gemini(
    client: genai.Client,
    contents,
    call_type: str,
    system_instruction: str | None = None,
    cached_content: str | None = None,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> tuple[str, CallStats]:
    """
    Call Gemini with retry + usage tracking.
    Returns (response_text, stats).

    Use cached_content (resource name) for cached system instructions.
    Falls back to system_instruction if cached_content is not provided.
    """
    stats = CallStats(call_type=call_type)
    t0 = time.monotonic()

    config = genai.types.GenerateContentConfig(temperature=temperature)
    if cached_content:
        config.cached_content = cached_content
    elif system_instruction:
        config.system_instruction = system_instruction

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            stats.input_tokens, stats.output_tokens, stats.thinking_tokens, stats.cached_tokens, stats.total_tokens = _extract_usage(response)
            stats.latency_ms = int((time.monotonic() - t0) * 1000)
            stats.retries = attempt
            _log_usage(stats)
            text = response.text.strip()
            _log_response(call_type, text)
            return text, stats
        except Exception as e:
            if attempt < max_retries - 1 and _is_transient(e):
                wait = 2 ** (attempt + 1)
                log.warning(f"Gemini transient error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            stats.latency_ms = int((time.monotonic() - t0) * 1000)
            stats.retries = attempt
            stats.success = False
            stats.error = str(e)[:200]
            _log_usage(stats)
            raise

def _expand_instances(data: dict) -> list[dict]:
    """
    Expand part_types + instances into a flat list of Part-compatible dicts.

    Validates constraints:
    - Max 20 part types
    - Max 10 instances per type
    - Max 62 total instances
    """
    part_types = data.get("part_types", [])
    instances = data.get("instances", [])

    if not part_types or not instances:
        raise ValueError("Expected 'part_types' and 'instances' arrays in response")

    # Validate limits
    if len(part_types) > 20:
        raise ValueError(f"Too many part types: {len(part_types)} (max 20)")
    if len(instances) > 62:
        raise ValueError(f"Too many instances: {len(instances)} (max 62)")

    # Index types by type_id
    types_by_id = {}
    for t in part_types:
        tid = t.get("type_id")
        if not tid:
            raise ValueError("Part type missing type_id")
        types_by_id[tid] = t

    # Count instances per type
    type_counts: dict[str, int] = {}
    for inst in instances:
        tid = inst.get("type_id", "")
        type_counts[tid] = type_counts.get(tid, 0) + 1
        if type_counts[tid] > 10:
            raise ValueError(f"Too many instances of type '{tid}': {type_counts[tid]} (max 10)")

    # Expand: merge type properties with instance placement
    parts = []
    for inst in instances:
        tid = inst.get("type_id", "")
        ptype = types_by_id.get(tid)
        if not ptype:
            raise ValueError(f"Instance '{inst.get('uid')}' references unknown type '{tid}'")

        # Instance rotation overrides type default if provided
        rotation = inst.get("rotation") or ptype.get("rotation", {"rx": 0, "ry": 0, "rz": 0})
        # Copy rotation so we can mutate it for radial mode
        rotation = dict(rotation)

        # Radial orientation: splay parts outward from their attachment position.
        # The base rotation's tilt (rx or ry) is reinterpreted as "tilt outward
        # by this many degrees" and decomposed into the correct rx/ry plane
        # based on each instance's radial angle.
        orientation_mode = ptype.get("orientation_mode", "parallel")
        if orientation_mode == "radial":
            u = inst.get("attachment_offset", 0.0)
            v = inst.get("attachment_offset_v", 0.0)
            if abs(u) > 0.001 or abs(v) > 0.001:
                import math
                theta = math.atan2(v, u)  # radial angle in radians
                face = inst.get("parent_face", "")

                if face in ("top", "bottom"):
                    # Face normal is Z. Tilt is the base rx value (outward lean).
                    # Decompose into rx/ry based on radial direction.
                    tilt = rotation.get("rx", 0)
                    tilt_rad = math.radians(tilt)
                    # Project tilt into the radial direction
                    rotation["rx"] = math.degrees(tilt_rad * -math.sin(theta))
                    rotation["ry"] = math.degrees(tilt_rad * math.cos(theta))
                    # Add radial spin around face normal
                    rotation["rz"] = rotation.get("rz", 0) + math.degrees(theta)
                elif face in ("front", "back"):
                    tilt = rotation.get("rz", 0)
                    tilt_rad = math.radians(tilt)
                    rotation["rz"] = math.degrees(tilt_rad * math.cos(theta))
                    rotation["rx"] = math.degrees(tilt_rad * -math.sin(theta))
                    rotation["ry"] = rotation.get("ry", 0) + math.degrees(theta)
                elif face in ("left", "right"):
                    tilt = rotation.get("rz", 0)
                    tilt_rad = math.radians(tilt)
                    rotation["rz"] = math.degrees(tilt_rad * math.cos(theta))
                    rotation["ry"] = math.degrees(tilt_rad * -math.sin(theta))
                    rotation["rx"] = rotation.get("rx", 0) + math.degrees(theta)

        # Use instance uid as part_name suffix if multiple instances of same type
        base_name = ptype.get("part_name", tid)
        display_name = inst["uid"] if type_counts.get(tid, 1) > 1 else base_name

        part = {
            "uid": inst["uid"],
            "part_name": display_name,
            "primitive_type": ptype["primitive_type"],
            "parent_part": inst.get("parent_part"),
            "parent_face": inst.get("parent_face"),
            "child_face": inst.get("child_face"),
            "attachment_offset": inst.get("attachment_offset", 0.0),
            "attachment_offset_v": inst.get("attachment_offset_v", 0.0),
            "dimensions": ptype["dimensions"],
            "rotation": rotation,
            "top_radius": ptype.get("top_radius"),
            "critical": ptype.get("critical", False),
            "color_id": ptype.get("color_id", "light_bluish_gray"),
        }
        parts.append(part)

    return parts


def generate_parts(description: str, api_key: str) -> tuple[list[dict], dict | None]:
    """
    Call Gemini 2.5 Pro to generate Part JSON from a text description.

    Gemini returns part_types + instances format.
    This function expands instances into a flat Part list ready for /api/run.
    Returns (parts, strategy) where strategy is the _strategy object or None.
    Raises ValueError on parse failure or API error.
    """
    client = genai.Client(api_key=api_key)
    cache_name = _get_or_create_cache(client, api_key, SYSTEM_PROMPT)
    text, stats = _call_gemini(
        client, description, call_type="generate",
        cached_content=cache_name, system_instruction=SYSTEM_PROMPT if not cache_name else None,
    )

    # Strip markdown code fences if the model wraps output
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    data = json.loads(text)

    # Extract _strategy before expansion (chain-of-thought, not used by pipeline)
    strategy = data.get("_strategy") if isinstance(data, dict) else None

    # Support both formats: new {part_types, instances} and legacy flat array
    if isinstance(data, list):
        # Legacy flat format — pass through with validation
        parts = data
    elif isinstance(data, dict) and "part_types" in data:
        parts = _expand_instances(data)
    else:
        raise ValueError(f"Unexpected response format: {text[:200]}")

    if len(parts) == 0:
        raise ValueError("No parts generated")

    # Basic validation
    uids = {p["uid"] for p in parts}
    roots = [p for p in parts if p.get("parent_part") is None]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly 1 root part, got {len(roots)}")
    for p in parts:
        parent = p.get("parent_part")
        if parent is not None and parent not in uids:
            raise ValueError(f"Part '{p['uid']}' references unknown parent '{parent}'")

    # Validate and fix color_ids
    for p in parts:
        cid = p.get("color_id", "")
        if cid not in LEGO_PALETTE:
            p["color_id"] = _nearest_color(cid)

    return parts, strategy, stats.to_dict()


def describe_image_bytes(image_bytes: bytes, mime_type: str, api_key: str) -> tuple[str, dict]:
    """
    Call Gemini 2.5 Pro with vision to describe image bytes for LEGO reconstruction.

    Returns (text_description, call_stats_dict) suitable for chaining into generate_parts().
    """
    client = genai.Client(api_key=api_key)
    contents = [
        genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        "Describe this object concisely for LEGO brick reconstruction. "
        "Include: overall shape, main structural parts, proportions, "
        "colors, and any distinctive features. Be specific about geometry "
        "(e.g. 'cylindrical wheels', 'rectangular body', 'cone-shaped roof'). "
        "Keep it to 2-4 sentences.",
    ]

    text, stats = _call_gemini(client, contents, call_type="describe_image", temperature=0.3)
    return text, stats.to_dict()


def describe_image(image_path: str, api_key: str) -> str:
    """
    Call Gemini 2.5 Pro with vision to describe an image for LEGO reconstruction.

    Returns a text description suitable for passing to generate_parts().
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    text, _stats = describe_image_bytes(image_bytes, mime_type, api_key)
    return text


# ---------------------------------------------------------------------------
# Shared spatial conventions for validation and feedback prompts
# ---------------------------------------------------------------------------

_SHARED_SPATIAL_CONVENTIONS = """\
## Coordinate System

Grid: 100 x 100 x 100 (integer voxel space)
X-axis: left (-) to right (+)
Y-axis: back (-) to front (+)
Z-axis: bottom (0) to top (+)

Direction vocabulary (STRICT):
  left / right       -> X-axis
  forward / backward -> Y-axis
  up / down          -> Z-axis

All dimensions are FULL extents (not radii or half-extents).

## Attachment Model (2-face)

Each attachment specifies two faces:
- parent_face: which face of the PARENT bounding cuboid this part attaches to
- child_face: which face of the CHILD that contacts the parent face
- Valid faces: top, bottom, front, back, left, right
- Position on the parent face is controlled by two normalized offsets:
  attachment_offset (u-axis) and attachment_offset_v (v-axis)
- Range: [-1, 1] where 0 = face center, +/-1 = face edges
- The pipeline automatically rotates the child so its child_face opposes the parent_face.
  Example: parent_face: "top", child_face: "bottom" means the child sits on top of the parent.
- The pipeline also aligns the root's primary axis to +Z (longest dim for cuboid/ellipsoid,
  height axis for cylinder/cone_frustum). The LLM rotation field is a delta from this canonical pose.

Attachment axis mapping (u, v per face):
  top / bottom:   u = X (left -> right),    v = Y (back -> front)
  front / back:   u = X (left -> right),    v = Z (bottom -> top)
  left / right:   u = Y (back -> front),    v = Z (bottom -> top)

## Rotation

Specified as { "rx": float, "ry": float, "rz": float } in degrees.
Applied in Z -> Y -> X order (Euler).
Rotation is hierarchical: children inherit all ancestor rotations.
Root parts rotate around their own center.
Child parts rotate around their attachment anchor.

## Dimensions

Always in pre-scale, root-normalized object space (root's largest dimension = 100 units).
Specified as { "width": int, "depth": int, "height": int } — full extents, not radii.
Resizing is anchored at the attachment face.

## Dimension Reasoning (for resize and add_part)

All dimensions must be absolute integers in the same root-normalized space as the structural JSON (root = 100).
Output the full intended { width, depth, height }, not deltas.
Do not guess dimensions from visual appearance alone. Cross-reference the current part's dimensions,
surrounding parts' dimensions in the structural JSON, and visible proportions in the projections.
Prefer small adjustments to the current values when proportions are roughly correct.
Large changes are justified when proportions are clearly wrong relative to the overall object.
Preserve proportional consistency with adjacent and connected parts.

## Primitive Types

  cuboid:       width (X), depth (Y), height (Z) — standard box
  cylinder:     width = depth = base diameter, height = Z extent
  ellipsoid:    width (X diameter), depth (Y diameter), height (Z diameter)
  cone_frustum: width = depth = base diameter, height = Z extent, top_radius = top circle radius (0 = full cone)

## Hierarchy and Subtree Behavior

Parts form a tree rooted at the single root part (parent_part: null).
When a part is modified (moved, rotated, resized), all its descendants move with it as a rigid subtree.
Deleting a part deletes its entire subtree (all descendants are removed).

## Criticality

critical: true — part is essential for recognition (must survive pipeline processing).
critical: false — part is decorative or secondary (may be pruned by pipeline if it causes conflicts).
"""

_SHARED_COLOR_PALETTE = """\
## Color Palette

20 fixed colors. You MUST use exact color_id strings from this list:

  black               #1B2A34   Dark surfaces, outlines, tires
  white               #F4F4F4   Eyes, teeth, highlights
  dark_bluish_gray    #6B7280   Metal, machinery, stone
  light_bluish_gray   #A8AFB8   Light metal, concrete
  red                 #C91A09   Primary red surfaces
  dark_red            #720E0F   Dark accents, deep red
  blue                #0055BF   Primary blue surfaces
  light_blue          #5A93DB   Sky, water, light accents
  yellow              #F2CD37   Gold, highlights, skin (stylized)
  orange              #D67923   Warm accents, construction
  green               #237841   Foliage, primary green
  dark_green          #184632   Deep foliage, military
  tan                 #E4CD9E   Skin, sand, light wood
  dark_tan            #B0A06F   Leather, aged surfaces
  brown               #582A12   Wood, earth, hair
  dark_brown          #352100   Dark wood, bark
  pink                #FC97AC   Light accents, flowers
  purple              #D3359D   Decorative, magical
  teal                #069D9F   Water, accent
  lime                #A5CA18   Bright accent, alien/neon

Do NOT use color names outside this list.
"""

_SHARED_ALLOWED_ACTIONS = """\
## Allowed Actions

You may ONLY use these seven actions:

1. translate — Move an existing part to a new attachment position.
{
  "action": "translate",
  "uid": "leg_front_left_0",
  "parent_face": "left",
  "child_face": "right",
  "attachment_offset": 0.8,
  "attachment_offset_v": 0.3
}
Fields: uid (target), parent_face, child_face, attachment_offset, attachment_offset_v.
Changing parent_face or child_face is higher disruption than sliding offsets on the same face.

2. rotate — Rotate an existing part.
{
  "action": "rotate",
  "uid": "leg_front_left_0",
  "rotation": { "rx": 0, "ry": 0, "rz": -45 }
}
Fields: uid (target), rotation object with rx, ry, rz in degrees — new absolute values.

3. resize — Change part dimensions.
{
  "action": "resize",
  "uid": "leg_front_left_0",
  "dimensions": { "width": 4, "depth": 4, "height": 18 },
  "top_radius": null
}
Fields: uid (target), dimensions object with width, depth, height. Optional: top_radius (cone_frustum only).

4. recolor — Change a part's color.
{
  "action": "recolor",
  "uid": "leg_front_left_0",
  "color_id": "dark_brown"
}
Fields: uid (target), color_id (must be valid palette color).

5. add_part — Add a new part attached to an existing parent.
Provide a ref_id for cross-referencing within the same edit batch, plus part_type and instance definitions.
The system generates the actual uid — do NOT include uid in the instance.
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
ref_id can be any descriptive string. If a subsequent action in the same batch references this
ref_id as a uid or parent_part, the pipeline resolves it to the system-generated uid.
parent_part must reference an existing part or a ref_id from an earlier add_part in the same batch.

6. toggle_critical — Flip the critical flag on one or more parts.
{
  "action": "toggle_critical",
  "uids": ["tail_0", "fin_left_0"]
}
Fields: uids (array of target part identifiers).
Constraint: A part cannot be set to critical: false if any of its descendants are critical: true.

7. delete — Remove a part and its entire subtree.
{
  "action": "delete",
  "uid": "tail_0"
}
WARNING: Deletion is recursive — all children of the deleted part are also removed.
If you intend to replace a part, you must: delete the old part, add_part the replacement,
then add_part each child that was on the old part re-attaching them to the new part.

If a correction cannot be expressed using these seven actions — DO NOT include it.
"""

_SHARED_PIPELINE_HANDLED = """\
## Pipeline-Handled (DO NOT suggest fixes for these)

The pipeline has already handled and will re-handle after the final rebuild:
- Connectivity (bridging, pruning)
- Collision / overlap resolution
- Scaling and normalization to grid
- Grounding (Z = 0)
- Voxel ownership and color resolution
- Critical part survival (restoration)
- Structural validity of attachments

NOTE: Orientation IS your responsibility. If the object is upside down or flipped (e.g. roof at the
bottom, door at the top), fix it using translate (change parent_face/child_face) or rotate actions.
The pipeline applies rotation but does not detect or correct wrong orientation.
"""

_SHARED_FOCUS_AREAS = """\
## Focus Areas

Prioritize corrections related to:
- Proportions — part dimensions don't match intent
- Missing parts — something clearly absent from the silhouette
- Placement — attachment face or offset is wrong
- Orientation — rotation doesn't match intent
- Silhouette — overall shape isn't recognizable
- Color — wrong color_id for a part

Prefer local edits (translate, rotate, resize, recolor) over structural changes (add_part, delete)
unless something is clearly missing.
"""


REFINE_PROMPT = f"""\
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
4. Output structured edit operations (translate, rotate, resize, recolor, add_part, delete, toggle_critical) that the pipeline will execute deterministically.

## Pipeline Context

What you're looking at: Earlier in this pipeline, you read a user's description and decomposed it
into a part-based structural representation. That representation was then processed by deterministic
pipeline stages — cleanup, scaling, collision resolution, voxelization — to produce the 3D object
shown in these views. Some of what you see may differ from your original intent due to pipeline
processing (overlap resolution, grounding, bridging). Your job is to identify where the result
diverges from the user's intent and propose targeted fixes. This is your one revision pass — make
it count. If something is fundamentally broken beyond what targeted edits can fix, say so in your
issues list and focus your edits on the highest-impact improvements you can make.

## Confidence-Driven Strategy

How your confidence level should guide your edits:
- high: Object is recognizable. Focus on fine-tuning — proportions, small placement adjustments,
  color corrections. Fewer edits is fine.
- medium: Object is identifiable but has notable issues. Fix the most impactful proportion,
  placement, or missing-part problems.
- low: Object is difficult to recognize. Prioritize silhouette and overall shape — large resize,
  reattachment, or add/delete to recover the intended form. Detail edits are wasted at this level.

{_SHARED_SPATIAL_CONVENTIONS}

{_SHARED_COLOR_PALETTE}

{_SHARED_ALLOWED_ACTIONS}

## Edit Budget (Validation Mode)

| Category | Max | Actions |
|---|---|---|
| Structural changes | 5 | add_part, delete |
| Local edits | 20 | translate, rotate, resize, recolor, toggle_critical |

Each granular action counts as 1 against its category limit.
Use as many changes as needed to meaningfully improve the object, up to these limits.

{_SHARED_PIPELINE_HANDLED}

{_SHARED_FOCUS_AREAS}

## Output Schema

Return ONLY valid JSON with this structure:

{{
  "guess": "what this looks like from the views alone (blind test)",
  "confidence": "high | medium | low",
  "no_edits_needed": false,
  "issues": [
    "description of issue 1",
    "description of issue 2"
  ],
  "edits": [ ... ]
}}

Rules:
- Each edit must address an identifiable issue.
- Each edit must specify concrete field values for a single action — no prose descriptions.
- guess is your independent visual impression before reading the structural representation.
- If no corrections are needed, return "no_edits_needed": true with "edits": [] and "confidence": "high".

No markdown. No code fences.
"""


FEEDBACK_PROMPT = f"""\
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

{_SHARED_SPATIAL_CONVENTIONS}

{_SHARED_COLOR_PALETTE}

{_SHARED_ALLOWED_ACTIONS}

## Edit Budget (Feedback Mode)

| Category | Max | Actions |
|---|---|---|
| Structural changes | 8 | add_part, delete |
| Local edits | 30 | translate, rotate, resize, recolor, toggle_critical |

Slightly higher than validation mode because user feedback may request broader changes.
Still bounded — if feedback implies more changes than the budget allows, prioritize by impact
and note what was deferred in feedback_interpretation.

{_SHARED_PIPELINE_HANDLED}

## Focus Areas (Feedback Mode)

Prioritize translating the user's feedback into edits. Specifically:

- Explicit requests — "make the legs longer" -> resize the legs. Direct mapping.
- Implicit requests — "it doesn't look like a spider" -> compare against spider characteristics,
  identify the biggest gaps, fix those.
- Ambiguous requests — "make it better" -> use your judgment on the most impactful improvements,
  but explain your interpretation.
- Contradictory requests — if the user asks for changes that conflict with each other or with the
  object's structure, apply the most reasonable subset and note what was skipped in feedback_interpretation.

Do NOT independently add fixes the user didn't ask for. If you notice something clearly broken
while interpreting the feedback, you may include it but flag it in feedback_interpretation as an
additional fix beyond what was requested.

## Guardrails

- Do NOT attempt full structural rewrites — even if the user implies one ("start over", "redo the whole thing").
  Apply the most impactful subset of local edits and explain the limitation.
- If feedback implies major re-architecture (changing the root part, rebuilding the hierarchy),
  apply only compatible local edits. Note in feedback_interpretation what couldn't be done.
- Prefer modifying existing parts over deleting and recreating large subtrees.
- Preserve overall object identity and hierarchy — the user wants to improve the object, not replace it.
- If the user requests something the pipeline handles automatically (connectivity, grounding, scaling),
  note it in feedback_interpretation and skip — the rebuild will handle it.

## Output Schema

Return ONLY valid JSON with this structure:

{{
  "feedback_interpretation": "What I understood the user wants: ...",
  "confidence": "high | medium | low",
  "edits": [ ... ]
}}

Rules:
- Each edit must specify concrete field values for a single action.
- Edits that affect a parent naturally cascade to its subtree; this is expected.
- If feedback cannot be expressed within the edit system, return {{ "edits": [] }} with an
  explanation in feedback_interpretation.
- If confidence is "low", prefer fewer, safer edits over aggressive changes.

No markdown. No code fences.
"""


def validate_and_refine(
    projection_images: list[tuple[bytes, str]],
    original_description: str,
    parts: list[dict],
    api_key: str,
) -> dict:
    """
    Single validation + edit-based refinement pass (Stage 5 validation mode).

    Args:
        projection_images: list of (image_bytes, mime_type) for 6 views
        original_description: what the user asked for
        parts: current flat parts list (dicts)
        api_key: Gemini API key

    Returns dict with:
        guess, confidence, no_edits_needed, issues, edits (raw from LLM),
        match (bool), refined_parts (parts with edits applied)
    """
    client = genai.Client(api_key=api_key)

    # Images FIRST (before structural data) so Gemini's guess is unbiased
    contents = []

    labels = ["Front", "Back", "Left", "Right", "Top", "Bottom"]
    for i, (img_bytes, mime) in enumerate(projection_images):
        label = labels[i] if i < len(labels) else f"View {i+1}"
        contents.append(f"{label} view:")
        contents.append(genai.types.Part.from_bytes(data=img_bytes, mime_type=mime))

    parts_json_str = json.dumps(parts, indent=2)
    contents.append(
        f"\n\nOriginal description: {original_description}\n\n"
        f"Current structural representation:\n{parts_json_str}"
    )

    refine_cache = _get_or_create_cache(client, api_key, REFINE_PROMPT)
    text, stats = _call_gemini(
        client, contents, call_type="validate",
        cached_content=refine_cache, system_instruction=REFINE_PROMPT if not refine_cache else None,
        temperature=0.2,
    )

    # Strip code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {
            "guess": text[:100], "confidence": "low", "match": False,
            "no_edits_needed": False,
            "issues": [], "edits": [],
            "feedback": "Could not parse validation response",
            "refined_parts": parts,
            "_call_stats": stats.to_dict(),
        }

    # Match check
    desc_words = set(original_description.lower().split())
    guess_words = set(result.get("guess", "").lower().split())
    stop = {"a", "an", "the", "with", "and", "of", "on", "in", "to", "is"}
    overlap = (desc_words - stop) & (guess_words - stop)
    result["match"] = len(overlap) > 0 or result.get("confidence") == "high"

    # Skip rebuild if no edits needed
    no_edits = result.get("no_edits_needed", False)
    edits = result.get("edits", [])

    if no_edits and result.get("confidence") == "high" and len(edits) == 0:
        result["refined_parts"] = parts
        result["edit_count"] = 0
        result["_call_stats"] = stats.to_dict()
        return result

    # Apply edits
    refined = _apply_edits(parts, edits)

    # Fix colors on any new/modified parts
    for p in refined:
        cid = p.get("color_id", "")
        if cid not in LEGO_PALETTE:
            p["color_id"] = _nearest_color(cid)

    result["refined_parts"] = refined
    result["edit_count"] = len(edits)
    result["_call_stats"] = stats.to_dict()

    return result


def feedback_refine(
    projection_images: list[tuple[bytes, str]],
    original_description: str,
    parts: list[dict],
    user_feedback_text: str,
    api_key: str,
) -> dict:
    """
    Stage 7 feedback refinement pass.

    Args:
        projection_images: list of (image_bytes, mime_type) for 6 views
        original_description: what the user originally asked for
        parts: current flat parts list (dicts)
        user_feedback_text: user's free-form feedback
        api_key: Gemini API key

    Returns dict with:
        feedback_interpretation, confidence, edits (raw from LLM),
        refined_parts (parts with edits applied)
    """
    client = genai.Client(api_key=api_key)

    parts_json_str = json.dumps(parts, indent=2)
    contents = []

    # Projections first
    labels = ["Front", "Back", "Left", "Right", "Top", "Bottom"]
    for i, (img_bytes, mime) in enumerate(projection_images):
        label = labels[i] if i < len(labels) else f"View {i+1}"
        contents.append(f"{label} view:")
        contents.append(genai.types.Part.from_bytes(data=img_bytes, mime_type=mime))

    contents.append(
        f"\n\nOriginal description: {original_description}\n\n"
        f"Current structural representation:\n{parts_json_str}\n\n"
        f"User feedback: {user_feedback_text}"
    )

    feedback_cache = _get_or_create_cache(client, api_key, FEEDBACK_PROMPT)
    text, stats = _call_gemini(
        client, contents, call_type="feedback",
        cached_content=feedback_cache, system_instruction=FEEDBACK_PROMPT if not feedback_cache else None,
        temperature=0.2,
    )

    # Strip code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {
            "feedback_interpretation": f"Could not parse feedback response: {text[:100]}",
            "confidence": "low",
            "edits": [],
            "refined_parts": parts,
            "_call_stats": stats.to_dict(),
        }

    # Apply edits
    edits = result.get("edits", [])
    refined = _apply_edits(parts, edits, structural_budget=8, local_budget=30)

    # Fix colors on any new/modified parts
    for p in refined:
        cid = p.get("color_id", "")
        if cid not in LEGO_PALETTE:
            p["color_id"] = _nearest_color(cid)

    result["refined_parts"] = refined
    result["edit_count"] = len(edits)
    result["_call_stats"] = stats.to_dict()

    return result


def _apply_edits(
    parts: list[dict],
    edits: list[dict],
    structural_budget: int = 5,
    local_budget: int = 20,
) -> list[dict]:
    """Apply a list of 7-action edits to a parts list. Returns new list.

    Actions: translate, rotate, resize, recolor, toggle_critical, add_part, delete.
    Budget limits are enforced per category.
    """
    import copy
    parts = copy.deepcopy(parts)
    by_uid = {p["uid"]: p for p in parts}

    # UID pre-assignment: scan add_parts first, generate uids, replace ref_ids
    ref_to_uid: dict[str, str] = {}
    type_counters: dict[str, int] = {}

    # Count existing instances per type_id pattern to avoid uid collisions
    for p in parts:
        uid = p.get("uid", "")
        # Parse pattern: type_id + "_" + counter
        if "_" in uid:
            prefix = uid.rsplit("_", 1)[0]
            suffix = uid.rsplit("_", 1)[1]
            if suffix.isdigit():
                current = type_counters.get(prefix, 0)
                type_counters[prefix] = max(current, int(suffix) + 1)

    for edit in edits:
        if edit.get("action") == "add_part":
            ref_id = edit.get("ref_id", "")
            pt = edit.get("part_type", {})
            type_id = pt.get("type_id", ref_id)
            counter = type_counters.get(type_id, 0)
            generated_uid = f"{type_id}_{counter}"
            type_counters[type_id] = counter + 1
            ref_to_uid[ref_id] = generated_uid

    # Replace ref_ids throughout the batch
    for edit in edits:
        action = edit.get("action", "")
        # Replace uid references
        if action in ("translate", "rotate", "resize", "recolor", "delete"):
            uid = edit.get("uid", "")
            if uid in ref_to_uid:
                edit["uid"] = ref_to_uid[uid]
        elif action == "toggle_critical":
            uids = edit.get("uids", [])
            edit["uids"] = [ref_to_uid.get(u, u) for u in uids]
        elif action == "add_part":
            inst = edit.get("instance", {})
            parent = inst.get("parent_part", "")
            if parent in ref_to_uid:
                inst["parent_part"] = ref_to_uid[parent]

    local_count = 0
    structural_count = 0

    _LOCAL_ACTIONS = {"translate", "rotate", "resize", "recolor", "toggle_critical"}
    _STRUCTURAL_ACTIONS = {"add_part", "delete"}

    for edit in edits:
        action = edit.get("action", "")

        if action in _LOCAL_ACTIONS and local_count >= local_budget:
            continue
        if action in _STRUCTURAL_ACTIONS and structural_count >= structural_budget:
            continue

        if action == "translate":
            uid = edit.get("uid", "")
            if uid in by_uid:
                target = by_uid[uid]
                if "parent_face" in edit:
                    target["parent_face"] = edit["parent_face"]
                if "child_face" in edit:
                    target["child_face"] = edit["child_face"]
                if "attachment_offset" in edit:
                    target["attachment_offset"] = max(-1.0, min(1.0, float(edit["attachment_offset"])))
                if "attachment_offset_v" in edit:
                    target["attachment_offset_v"] = max(-1.0, min(1.0, float(edit["attachment_offset_v"])))
                local_count += 1

        elif action == "rotate":
            uid = edit.get("uid", "")
            rotation = edit.get("rotation", {})
            if uid in by_uid and rotation:
                target = by_uid[uid]
                target["rotation"] = {
                    "rx": rotation.get("rx", 0),
                    "ry": rotation.get("ry", 0),
                    "rz": rotation.get("rz", 0),
                }
                local_count += 1

        elif action == "resize":
            uid = edit.get("uid", "")
            dims = edit.get("dimensions", {})
            if uid in by_uid and dims:
                target = by_uid[uid]
                target["dimensions"] = {
                    "width": dims.get("width", target["dimensions"]["width"]),
                    "depth": dims.get("depth", target["dimensions"]["depth"]),
                    "height": dims.get("height", target["dimensions"]["height"]),
                }
                if "top_radius" in edit:
                    target["top_radius"] = edit["top_radius"]
                local_count += 1

        elif action == "recolor":
            uid = edit.get("uid", "")
            color_id = edit.get("color_id", "")
            if uid in by_uid and color_id:
                by_uid[uid]["color_id"] = color_id
                local_count += 1

        elif action == "toggle_critical":
            uids = edit.get("uids", [])
            # Build children map for descendant check
            children_map: dict[str, list[str]] = {}
            for p in parts:
                parent = p.get("parent_part")
                if parent:
                    children_map.setdefault(parent, []).append(p["uid"])

            def _has_critical_descendant(uid: str) -> bool:
                for child_uid in children_map.get(uid, []):
                    child = by_uid.get(child_uid)
                    if child and child.get("critical", False):
                        return True
                    if _has_critical_descendant(child_uid):
                        return True
                return False

            for uid in uids:
                if uid in by_uid:
                    target = by_uid[uid]
                    # Guard: can't unset critical if descendants are critical
                    if target.get("critical", False) and _has_critical_descendant(uid):
                        log.warning(f"toggle_critical: skipping {uid} — has critical descendants")
                        continue
                    target["critical"] = not target.get("critical", False)
            local_count += 1

        elif action == "add_part":
            ref_id = edit.get("ref_id", "")
            pt = edit.get("part_type", {})
            inst = edit.get("instance", {})
            generated_uid = ref_to_uid.get(ref_id)

            if not generated_uid or generated_uid in by_uid:
                continue

            # Instance rotation overrides type default if provided
            rotation = inst.get("rotation") or pt.get("rotation", {"rx": 0, "ry": 0, "rz": 0})

            new_part = {
                "uid": generated_uid,
                "part_name": pt.get("part_name", pt.get("type_id", ref_id)),
                "primitive_type": pt.get("primitive_type", "cuboid"),
                "parent_part": inst.get("parent_part"),
                "parent_face": inst.get("parent_face"),
                "child_face": inst.get("child_face"),
                "attachment_offset": inst.get("attachment_offset", 0.0),
                "attachment_offset_v": inst.get("attachment_offset_v", 0.0),
                "dimensions": pt.get("dimensions", {"width": 5, "depth": 5, "height": 5}),
                "rotation": rotation,
                "top_radius": pt.get("top_radius"),
                "critical": pt.get("critical", False),
                "color_id": pt.get("color_id", "light_bluish_gray"),
            }
            parts.append(new_part)
            by_uid[generated_uid] = new_part
            structural_count += 1

        elif action == "delete":
            uid = edit.get("uid", "")
            if uid not in by_uid:
                continue
            # Don't remove root
            if by_uid[uid].get("parent_part") is None:
                log.warning(f"delete: skipping root part {uid}")
                continue

            # Collect entire subtree to delete
            to_delete = set()

            def _collect_subtree(u: str):
                to_delete.add(u)
                for p in parts:
                    if p.get("parent_part") == u:
                        _collect_subtree(p["uid"])

            _collect_subtree(uid)

            parts = [p for p in parts if p["uid"] not in to_delete]
            for d_uid in to_delete:
                by_uid.pop(d_uid, None)
            structural_count += 1

    return parts
