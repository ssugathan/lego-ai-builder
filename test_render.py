"""Golden tests for render._project_view.

The reference implementation below is a copy of the original triple-nested
loop version. The tests assert that the vectorized implementation in
render.py produces identical output for random grids across all 6 views.
"""
from __future__ import annotations

import numpy as np
import pytest

from render import _project_view

VIEWS = ["front", "back", "left", "right", "top", "bottom"]


def _project_view_reference(grid: np.ndarray, view: str) -> np.ndarray:
    """Original loop-based implementation, kept as the golden reference."""
    if view == "front":
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


def _random_grid(rng: np.random.Generator, shape, density: float) -> np.ndarray:
    grid = rng.integers(1, 8, size=shape).astype(np.int32)
    grid[rng.random(shape) > density] = 0
    return grid


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("seed,shape,density", [
    (0, (10, 10, 10), 0.3),
    (1, (10, 10, 10), 0.05),
    (2, (10, 10, 10), 0.9),
    (3, (7, 11, 5), 0.4),
    (4, (12, 6, 9), 0.5),
])
def test_project_view_matches_reference(view, seed, shape, density):
    rng = np.random.default_rng(seed)
    grid = _random_grid(rng, shape, density)
    expected = _project_view_reference(grid, view)
    actual = _project_view(grid, view)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("view", VIEWS)
def test_project_view_empty_grid(view):
    grid = np.zeros((8, 8, 8), dtype=np.int32)
    expected = _project_view_reference(grid, view)
    actual = _project_view(grid, view)
    np.testing.assert_array_equal(actual, expected)
    assert not actual.any()


@pytest.mark.parametrize("view", VIEWS)
def test_project_view_full_grid(view):
    rng = np.random.default_rng(42)
    grid = rng.integers(1, 5, size=(6, 6, 6)).astype(np.int32)
    expected = _project_view_reference(grid, view)
    actual = _project_view(grid, view)
    np.testing.assert_array_equal(actual, expected)


def test_project_view_single_voxel_orientation():
    """One voxel at a known position lands at the right pixel in each view."""
    grid = np.zeros((5, 5, 5), dtype=np.int32)
    grid[1, 2, 3] = 7
    assert _project_view(grid, "front")[1, 3] == 7
    assert _project_view(grid, "back")[1, 3] == 7
    assert _project_view(grid, "left")[2, 3] == 7
    assert _project_view(grid, "right")[2, 3] == 7
    assert _project_view(grid, "top")[1, 2] == 7
    assert _project_view(grid, "bottom")[1, 2] == 7


def test_project_view_first_hit_ordering():
    """Occluded voxels are hidden: the first voxel along the ray wins."""
    grid = np.zeros((4, 4, 4), dtype=np.int32)
    grid[2, 0, 2] = 1  # low y
    grid[2, 3, 2] = 2  # high y
    assert _project_view(grid, "front")[2, 2] == 2  # ray from high y
    assert _project_view(grid, "back")[2, 2] == 1   # ray from low y

    grid = np.zeros((4, 4, 4), dtype=np.int32)
    grid[0, 2, 2] = 1  # low x
    grid[3, 2, 2] = 2  # high x
    assert _project_view(grid, "left")[2, 2] == 1   # ray from low x
    assert _project_view(grid, "right")[2, 2] == 2  # ray from high x

    grid = np.zeros((4, 4, 4), dtype=np.int32)
    grid[2, 2, 0] = 1  # low z
    grid[2, 2, 3] = 2  # high z
    assert _project_view(grid, "top")[2, 2] == 2    # ray from high z
    assert _project_view(grid, "bottom")[2, 2] == 1 # ray from low z


def test_project_view_unknown_view_raises():
    grid = np.zeros((3, 3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        _project_view(grid, "diagonal")
