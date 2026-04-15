"""
Tests for pipeline stages 10–12:
  apply_ownership, critical_restoration, enforce_connectivity.
"""
from collections import deque

import numpy as np
import pytest

from pipeline import (
    GRID_X,
    GRID_Y,
    GRID_Z,
    PartState,
    apply_ownership,
    critical_restoration,
    enforce_connectivity,
)
from schema import Part, PartDimensions, PrimitiveType


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _part(
    uid: str,
    parent: str | None = None,
    critical: bool = False,
    dims: tuple[float, float, float] = (5.0, 5.0, 5.0),
    primitive: PrimitiveType = PrimitiveType.CUBOID,
    parent_face=None,
) -> Part:
    return Part(
        uid=uid,
        part_name=uid,
        primitive_type=primitive,
        parent_part=parent,
        parent_face=parent_face,
        critical=critical,
        dimensions=PartDimensions(width=dims[0], depth=dims[1], height=dims[2]),
    )


def _state(
    uid: str,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dims: tuple[float, float, float] = (5.0, 5.0, 5.0),
) -> PartState:
    return PartState(
        uid=uid,
        dims=PartDimensions(width=dims[0], depth=dims[1], height=dims[2]),
        center=center,
    )


def _connected(grid: np.ndarray, pos1: tuple, pos2: tuple) -> bool:
    """Return True if pos1 and pos2 are in the same 6-connected component."""
    if grid[pos1] == 0 or grid[pos2] == 0:
        return False
    visited: set = {pos1}
    q: deque = deque([pos1])
    while q:
        x, y, z = q.popleft()
        if (x, y, z) == pos2:
            return True
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            nb = (x + dx, y + dy, z + dz)
            if nb not in visited and 0 <= nb[0] < GRID_X and 0 <= nb[1] < GRID_Y and 0 <= nb[2] < GRID_Z:
                if grid[nb] != 0:
                    visited.add(nb)
                    q.append(nb)
    return False


# ---------------------------------------------------------------------------
# apply_ownership
# ---------------------------------------------------------------------------

class TestApplyOwnership:
    def test_uncontested_voxel_assigned_directly(self):
        parts = [_part("A")]
        states = [_state("A")]
        claims = {(1, 1, 1): frozenset({"A"})}
        grid = apply_ownership(parts, states, claims)
        assert grid.shape == (GRID_X, GRID_Y, GRID_Z)
        assert grid[1, 1, 1] == 1   # A → states[0] → index 1
        assert grid[0, 0, 0] == 0   # empty elsewhere

    def test_empty_claims_map_gives_zero_grid(self):
        parts = [_part("A")]
        states = [_state("A")]
        grid = apply_ownership(parts, states, {})
        assert np.all(grid == 0)

    def test_descendant_beats_ancestor(self):
        parent = _part("P")
        child = _part("C", parent="P")
        parts = [parent, child]
        states = [_state("P"), _state("C")]
        claims = {(5, 5, 5): frozenset({"P", "C"})}
        grid = apply_ownership(parts, states, claims)
        # C (states[1], index 2) is descendant of P → C wins.
        assert grid[5, 5, 5] == 2

    def test_grandchild_beats_grandparent(self):
        gp = _part("GP")
        p = _part("P", parent="GP")
        gc = _part("GC", parent="P")
        parts = [gp, p, gc]
        states = [_state("GP"), _state("P"), _state("GC")]
        claims = {(5, 5, 5): frozenset({"GP", "GC"})}
        grid = apply_ownership(parts, states, claims)
        # GC is descendant of GP → GC (index 3) wins.
        assert grid[5, 5, 5] == 3

    def test_critical_beats_noncritical(self):
        nc = _part("NC")
        cr = _part("CR", critical=True)
        parts = [nc, cr]
        states = [_state("NC"), _state("CR")]
        claims = {(5, 5, 5): frozenset({"NC", "CR"})}
        grid = apply_ownership(parts, states, claims)
        # CR (index 2) wins over NC.
        assert grid[5, 5, 5] == 2

    def test_volume_rule_smaller_wins(self):
        # small: 1×1×1 = 1; large: 10×10×10 = 1000; 1 < 0.2 × 1000
        small = _part("small", dims=(1.0, 1.0, 1.0))
        large = _part("large", dims=(10.0, 10.0, 10.0))
        parts = [small, large]
        states = [_state("small", dims=(1.0, 1.0, 1.0)), _state("large", dims=(10.0, 10.0, 10.0))]
        claims = {(5, 5, 5): frozenset({"small", "large"})}
        grid = apply_ownership(parts, states, claims)
        # small (index 1) wins via volume rule.
        assert grid[5, 5, 5] == 1

    def test_volume_rule_does_not_apply_when_close(self):
        # equal dims → volume rule doesn't trigger; fall through to UID tiebreak
        a = _part("aaa", dims=(5.0, 5.0, 5.0))
        b = _part("bbb", dims=(5.0, 5.0, 5.0))
        parts = [a, b]
        states = [_state("aaa"), _state("bbb")]
        claims = {(5, 5, 5): frozenset({"aaa", "bbb"})}
        grid = apply_ownership(parts, states, claims)
        # "aaa" < "bbb" → "aaa" (index 1) wins via UID tiebreak.
        assert grid[5, 5, 5] == 1

    def test_uid_tiebreak_smaller_wins(self):
        a = _part("zzz")
        b = _part("aaa")
        parts = [a, b]
        states = [_state("zzz"), _state("aaa")]
        claims = {(3, 3, 3): frozenset({"zzz", "aaa"})}
        grid = apply_ownership(parts, states, claims)
        # "aaa" < "zzz" → "aaa" (index 2) wins.
        assert grid[3, 3, 3] == 2

    def test_critical_ancestor_vs_noncritical_descendant_descendant_wins(self):
        # Rule 1 (descendant) must override rule 2 (critical).
        ancestor = _part("A", critical=True)
        descendant = _part("D", parent="A", critical=False)
        parts = [ancestor, descendant]
        states = [_state("A"), _state("D")]
        claims = {(5, 5, 5): frozenset({"A", "D"})}
        grid = apply_ownership(parts, states, claims)
        # D is descendant of A → D wins even though A is critical and D is not.
        assert grid[5, 5, 5] == 2

    def test_three_claimants_transitive_consistency(self):
        # Three-way contest: grandparent (critical, large) vs parent (non-critical, large)
        # vs child (non-critical, small).
        # Expected resolution chain:
        #   child beats parent (descendant rule)
        #   child beats grandparent (descendant rule, transitive)
        gp = _part("GP", critical=True, dims=(10.0, 10.0, 10.0))
        p = _part("P", parent="GP", dims=(10.0, 10.0, 10.0))
        c = _part("C", parent="P", dims=(1.0, 1.0, 1.0))
        parts = [gp, p, c]
        states = [_state("GP", dims=(10.0, 10.0, 10.0)),
                  _state("P",  dims=(10.0, 10.0, 10.0)),
                  _state("C",  dims=(1.0, 1.0, 1.0))]
        claims = {(5, 5, 5): frozenset({"GP", "P", "C"})}
        grid = apply_ownership(parts, states, claims)
        # C (index 3) is the deepest descendant → wins over both GP and P.
        assert grid[5, 5, 5] == 3

    def test_multiple_voxels_resolved_independently(self):
        parent = _part("P")
        child = _part("C", parent="P")
        parts = [parent, child]
        states = [_state("P"), _state("C")]
        claims = {
            (1, 1, 1): frozenset({"P"}),          # uncontested → P
            (2, 2, 2): frozenset({"P", "C"}),      # contested → C wins
            (3, 3, 3): frozenset({"C"}),            # uncontested → C
        }
        grid = apply_ownership(parts, states, claims)
        assert grid[1, 1, 1] == 1
        assert grid[2, 2, 2] == 2
        assert grid[3, 3, 3] == 2


# ---------------------------------------------------------------------------
# critical_restoration
# ---------------------------------------------------------------------------

class TestCriticalRestoration:
    def test_critical_with_voxels_unchanged(self):
        p = _part("P", critical=True)
        s = _state("P", center=(5.0, 5.0, 5.0))
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        grid[5, 5, 5] = 1
        result = critical_restoration([p], [s], grid)
        assert np.sum(result != 0) == 1
        assert result[5, 5, 5] == 1

    def test_critical_without_voxels_gets_one_voxel(self):
        p = _part("P", critical=True)
        s = _state("P", center=(10.7, 20.2, 30.9))
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        result = critical_restoration([p], [s], grid)
        # floor(10.7)=10, floor(20.2)=20, floor(30.9)=30
        assert result[10, 20, 30] == 1
        assert np.sum(result != 0) == 1

    def test_noncritical_without_voxels_not_restored(self):
        p = _part("P", critical=False)
        s = _state("P", center=(10.0, 10.0, 10.0))
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        result = critical_restoration([p], [s], grid)
        assert np.all(result == 0)

    def test_center_clamped_to_grid_bounds(self):
        p = _part("P", critical=True)
        # center outside grid bounds
        s = _state("P", center=(-5.0, 120.0, 150.0))
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        result = critical_restoration([p], [s], grid)
        # Should clamp: i=0, j=GRID_Y-1, k=GRID_Z-1
        assert result[0, GRID_Y - 1, GRID_Z - 1] == 1

    def test_multiple_critical_parts_each_restored(self):
        pa = _part("A", critical=True)
        pb = _part("B", critical=True)
        sa = _state("A", center=(5.0, 5.0, 5.0))
        sb = _state("B", center=(20.0, 20.0, 20.0))
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        # Neither has voxels yet.
        result = critical_restoration([pa, pb], [sa, sb], grid)
        assert result[5, 5, 5] == 1    # A → states[0]
        assert result[20, 20, 20] == 2  # B → states[1]


# ---------------------------------------------------------------------------
# enforce_connectivity
# ---------------------------------------------------------------------------

class TestEnforceConnectivity:
    def test_already_connected_unchanged(self):
        # Parent and child voxels are directly adjacent — no bridge needed.
        parent_p = _part("P", critical=True)
        child_p = _part("C", parent="P", critical=True, parent_face="top")
        parts = [parent_p, child_p]
        states = [_state("P"), _state("C")]
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        grid[5, 5, 5] = 1   # parent
        grid[5, 5, 6] = 2   # child adjacent to parent
        result = enforce_connectivity(parts, states, grid)
        assert result[5, 5, 5] == 1
        assert result[5, 5, 6] == 2

    def test_bridge_created_for_disconnected_critical_pair(self):
        parent_p = _part("P", critical=True)
        child_p = _part("C", parent="P", critical=True, parent_face="top")
        parts = [parent_p, child_p]
        states = [
            _state("P", center=(5.5, 5.5, 5.5)),
            _state("C", center=(5.5, 5.5, 15.5)),
        ]
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        grid[5, 5, 5] = 1   # parent
        grid[5, 5, 15] = 2  # child, 9 voxels away in Z
        result = enforce_connectivity(parts, states, grid)
        # Both endpoints must survive.
        assert result[5, 5, 5] == 1
        assert result[5, 5, 15] == 2
        # They must now be 6-connected.
        assert _connected(result, (5, 5, 5), (5, 5, 15))

    def test_bridge_does_not_overwrite_critical_voxels(self):
        # Third critical part's voxel lies on the bridge path.
        root_p = _part("R", critical=True)
        child_p = _part("C", parent="R", critical=True, parent_face="top")
        third_p = _part("T", parent="R", critical=True)
        parts = [root_p, child_p, third_p]
        states = [
            _state("R", center=(5.5, 5.5, 5.5)),
            _state("C", center=(5.5, 5.5, 15.5)),
            _state("T", center=(5.5, 5.5, 10.5)),
        ]
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        grid[5, 5, 5] = 1   # R
        grid[5, 5, 15] = 2  # C
        grid[5, 5, 10] = 3  # T — lies on expected bridge path
        result = enforce_connectivity(parts, states, grid)
        # T's voxel must NOT be overwritten by the bridge.
        assert result[5, 5, 10] == 3

    def test_disconnected_noncritical_component_removed(self):
        crit_p = _part("CR", critical=True)
        nc_p = _part("NC", critical=False)
        parts = [crit_p, nc_p]
        states = [
            _state("CR", center=(5.5, 5.5, 5.5)),
            _state("NC", center=(40.5, 40.5, 90.5)),
        ]
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        grid[5, 5, 5] = 1   # critical — isolated
        grid[40, 40, 90] = 2  # non-critical — isolated, far away
        result = enforce_connectivity(parts, states, grid)
        assert result[5, 5, 5] == 1    # critical preserved
        assert result[40, 40, 90] == 0  # non-critical removed

    def test_no_critical_parts_returns_grid(self):
        # No critical parts → no bridging, no removal (no critical component).
        nc = _part("NC")
        parts = [nc]
        states = [_state("NC")]
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        grid[5, 5, 5] = 1
        result = enforce_connectivity(parts, states, grid)
        # Function exits early; grid returned unchanged.
        assert result[5, 5, 5] == 1

    def test_full_chain_grandparent_parent_child_all_disconnected(self):
        # Three critical parts each in an isolated voxel cluster, none adjacent.
        # After enforce_connectivity all three must be in one 6-connected component.
        gp_p = _part("GP", critical=True)
        p_p = _part("P",  parent="GP", critical=True, parent_face="top")
        c_p = _part("C",  parent="P",  critical=True, parent_face="top")
        parts = [gp_p, p_p, c_p]
        states = [
            _state("GP", center=(5.5,  5.5,  5.5)),
            _state("P",  center=(5.5,  5.5, 25.5)),
            _state("C",  center=(5.5,  5.5, 45.5)),
        ]
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        grid[5, 5, 5]  = 1  # grandparent
        grid[5, 5, 25] = 2  # parent
        grid[5, 5, 45] = 3  # child

        result = enforce_connectivity(parts, states, grid)

        # All three original voxels preserved with correct ownership.
        assert result[5, 5, 5]  == 1
        assert result[5, 5, 25] == 2
        assert result[5, 5, 45] == 3

        # All three must be 6-connected to each other.
        assert _connected(result, (5, 5, 5),  (5, 5, 25))
        assert _connected(result, (5, 5, 25), (5, 5, 45))
        assert _connected(result, (5, 5, 5),  (5, 5, 45))

    def test_single_critical_part_connected_to_itself(self):
        p = _part("P", critical=True)
        parts = [p]
        states = [_state("P")]
        grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.int32)
        grid[5, 5, 5] = 1
        result = enforce_connectivity(parts, states, grid)
        assert result[5, 5, 5] == 1
