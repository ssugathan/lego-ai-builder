"""
Server-side voxel projection rendering for validation.

Generates 6 orthographic views (front, back, left, right, top, bottom)
as PNG images from a voxel grid, plus segmentation images and legend.

Grid coords: X=left/right(0-99), Y=front/back(0-99), Z=up(0-99)
Grid shape: (100, 100, 100) indexed as grid[x, y, z]
"""
from __future__ import annotations

import io
from typing import Sequence

import numpy as np


# LEGO palette: color_id → (R, G, B) normalized 0-1
_LEGO_RGB = {
    "black": (0.106, 0.165, 0.204), "white": (0.957, 0.957, 0.957),
    "dark_bluish_gray": (0.420, 0.447, 0.502), "light_bluish_gray": (0.659, 0.686, 0.722),
    "red": (0.788, 0.102, 0.035), "dark_red": (0.447, 0.055, 0.059),
    "blue": (0.0, 0.333, 0.749), "light_blue": (0.353, 0.576, 0.859),
    "yellow": (0.949, 0.804, 0.216), "orange": (0.839, 0.475, 0.137),
    "green": (0.137, 0.471, 0.255), "dark_green": (0.094, 0.275, 0.196),
    "tan": (0.894, 0.804, 0.620), "dark_tan": (0.690, 0.627, 0.435),
    "brown": (0.345, 0.165, 0.071), "dark_brown": (0.208, 0.129, 0.0),
    "pink": (0.988, 0.592, 0.675), "purple": (0.827, 0.208, 0.616),
    "teal": (0.024, 0.616, 0.624), "lime": (0.647, 0.792, 0.094),
}

# 40 maximally-distinct segmentation colors (HSL-spaced, alternating lightness)
_SEG_COLORS = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0),
    (0.0, 0.5, 0.0), (0.5, 0.5, 0.0), (0.0, 0.5, 1.0), (1.0, 0.0, 0.5),
    (0.5, 1.0, 0.0), (0.0, 0.0, 0.5), (1.0, 0.5, 0.5), (0.5, 1.0, 0.5),
    (0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.5),
    (0.75, 0.25, 0.0), (0.0, 0.75, 0.25), (0.25, 0.0, 0.75), (0.75, 0.0, 0.25),
    (0.25, 0.75, 0.0), (0.0, 0.25, 0.75), (0.75, 0.75, 0.25), (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75), (0.25, 0.25, 0.0), (0.0, 0.25, 0.25), (0.25, 0.0, 0.25),
    (1.0, 0.75, 0.25), (0.25, 1.0, 0.75), (0.75, 0.25, 1.0), (1.0, 0.25, 0.75),
    (0.75, 1.0, 0.25), (0.25, 0.75, 1.0), (0.6, 0.4, 0.2), (0.2, 0.6, 0.4),
]

# Fallback palette indexed by part_idx
_FALLBACK = [
    (0.788, 0.102, 0.035), (0.0, 0.333, 0.749), (0.137, 0.471, 0.255),
    (0.949, 0.804, 0.216), (0.839, 0.475, 0.137), (0.827, 0.208, 0.616),
    (0.024, 0.616, 0.624), (0.420, 0.447, 0.502), (0.353, 0.576, 0.859),
    (0.647, 0.792, 0.094),
]

# Khaki background for color images
_BG_COLOR = (0.765, 0.690, 0.569)
# Black background for segmentation images
_SEG_BG = (0.0, 0.0, 0.0)


def _project_view(grid: np.ndarray, view: str) -> np.ndarray:
    """Project grid along a view axis, returning 2D array of part indices (first hit)."""
    # Ray traversal per spec §5.0: find FIRST non-empty voxel along ray direction
    if view == "front":
        # Ray: -Y (y=99→0), image: X(h) × Z(v)
        for y in range(grid.shape[1] - 1, -1, -1):
            pass
        # Use argmax along Y from front (high Y)
        proj = np.zeros((grid.shape[0], grid.shape[2]), dtype=np.int32)
        for x in range(grid.shape[0]):
            for z in range(grid.shape[2]):
                for y in range(grid.shape[1] - 1, -1, -1):
                    if grid[x, y, z] != 0:
                        proj[x, z] = grid[x, y, z]
                        break
        return proj
    elif view == "back":
        proj = np.zeros((grid.shape[0], grid.shape[2]), dtype=np.int32)
        for x in range(grid.shape[0]):
            for z in range(grid.shape[2]):
                for y in range(grid.shape[1]):
                    if grid[x, y, z] != 0:
                        proj[x, z] = grid[x, y, z]
                        break
        return proj
    elif view == "left":
        proj = np.zeros((grid.shape[1], grid.shape[2]), dtype=np.int32)
        for y_idx in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                for x in range(grid.shape[0]):
                    if grid[x, y_idx, z] != 0:
                        proj[y_idx, z] = grid[x, y_idx, z]
                        break
        return proj
    elif view == "right":
        proj = np.zeros((grid.shape[1], grid.shape[2]), dtype=np.int32)
        for y_idx in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                for x in range(grid.shape[0] - 1, -1, -1):
                    if grid[x, y_idx, z] != 0:
                        proj[y_idx, z] = grid[x, y_idx, z]
                        break
        return proj
    elif view == "top":
        proj = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.int32)
        for x in range(grid.shape[0]):
            for y_idx in range(grid.shape[1]):
                for z in range(grid.shape[2] - 1, -1, -1):
                    if grid[x, y_idx, z] != 0:
                        proj[x, y_idx] = grid[x, y_idx, z]
                        break
        return proj
    elif view == "bottom":
        proj = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.int32)
        for x in range(grid.shape[0]):
            for y_idx in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    if grid[x, y_idx, z] != 0:
                        proj[x, y_idx] = grid[x, y_idx, z]
                        break
        return proj
    raise ValueError(f"Unknown view: {view}")


def _render_image(
    proj: np.ndarray,
    color_fn,
    bg: tuple[float, float, float],
    label: str,
    grid_overlay: bool = False,
) -> bytes:
    """Render a 2D projection to PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = proj.shape
    img = np.full((h, w, 3), bg, dtype=np.float32)
    for idx in np.unique(proj):
        if idx == 0:
            continue
        mask = proj == idx
        img[mask] = color_fn(int(idx))

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=25)  # 100×100 pixels
    # Transpose so the vertical (or "north") world axis becomes the row axis;
    # origin="lower" then places world index 0 at the bottom of the image.
    display = img.transpose(1, 0, 2)
    ax.imshow(display, origin="lower", interpolation="nearest")

    if grid_overlay:
        # Grid lines every 10 units
        for i in range(0, max(h, w) + 1, 10):
            ax.axhline(y=i - 0.5, color="gray", linewidth=0.3, alpha=0.4)
            ax.axvline(x=i - 0.5, color="gray", linewidth=0.3, alpha=0.4)

    ax.set_title(label.capitalize(), color="white", fontsize=10)
    ax.axis("off")
    fig.patch.set_facecolor("#141414")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_projections(
    grid: np.ndarray,
    parts_meta: list[dict] | None = None,
) -> list[tuple[bytes, str, str]]:
    """
    Render 6 orthographic color projection views of the voxel grid.

    Args:
        grid: (100, 100, 100) int32 array, 0=empty, n=part index (1-based)
        parts_meta: optional list of part dicts with "color_id" field

    Returns:
        List of (png_bytes, mime_type, label) for each view.
        Views: Front, Back, Left, Right, Top, Bottom
    """
    # Build color map: part_idx → RGB
    color_map = {}
    if parts_meta:
        for p in parts_meta:
            idx = p.get("idx", 0)
            cid = p.get("color_id", "")
            color_map[idx] = _LEGO_RGB.get(cid, _FALLBACK[(idx - 1) % len(_FALLBACK)])

    def _get_color(part_idx):
        if part_idx in color_map:
            return color_map[part_idx]
        return _FALLBACK[(part_idx - 1) % len(_FALLBACK)]

    view_names = ["front", "back", "left", "right", "top", "bottom"]
    results = []
    for view in view_names:
        proj = _project_view(grid, view)
        png = _render_image(proj, _get_color, _BG_COLOR, view, grid_overlay=True)
        results.append((png, "image/png", view))

    return results


def render_segmentation(
    grid: np.ndarray,
    parts_meta: list[dict] | None = None,
) -> tuple[list[tuple[bytes, str, str]], str]:
    """
    Render 6 segmentation images and generate legend text.

    Each pixel is colored by the owning part's segmentation color.
    Background (empty) is black.

    Args:
        grid: (100, 100, 100) int32 array
        parts_meta: list of part dicts with "uid", "part_name", "primitive_type"

    Returns:
        (segmentation_images, legend_text)
        segmentation_images: list of (png_bytes, mime_type, label)
        legend_text: string mapping colors to uid/part_name/primitive_type
    """
    # Assign segmentation colors to parts
    # Simple assignment: part_idx → color from _SEG_COLORS
    seg_color_map = {}
    legend_lines = ["Segmentation map legend:"]

    if parts_meta:
        for p in parts_meta:
            idx = p.get("idx", 0)
            if idx == 0:
                continue
            color_idx = (idx - 1) % len(_SEG_COLORS)
            seg_color_map[idx] = _SEG_COLORS[color_idx]
            r, g, b = _SEG_COLORS[color_idx]
            hex_color = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
            uid = p.get("uid", f"part_{idx}")
            name = p.get("name", p.get("part_name", ""))
            ptype = p.get("primitive_type", "")
            legend_lines.append(f'- {hex_color} → uid "{uid}", "{name}" ({ptype})')

    def _get_seg_color(part_idx):
        if part_idx in seg_color_map:
            return seg_color_map[part_idx]
        return _SEG_COLORS[(part_idx - 1) % len(_SEG_COLORS)]

    view_names = ["front", "back", "left", "right", "top", "bottom"]
    results = []
    for view in view_names:
        proj = _project_view(grid, view)
        png = _render_image(proj, _get_seg_color, _SEG_BG, f"{view}_seg", grid_overlay=False)
        results.append((png, "image/png", f"{view}_seg"))

    return results, "\n".join(legend_lines)
