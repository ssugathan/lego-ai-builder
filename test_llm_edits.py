"""Unit tests for the pure dict/list helpers in llm.py.

_expand_instances: part_types + instances → flat parts list.
_apply_edits: 7-action edit application with budget enforcement.
"""
from __future__ import annotations

import pytest

# llm.py imports google-genai at module level; skip cleanly if it is absent.
pytest.importorskip("google.genai")

from llm import _apply_edits, _expand_instances


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cuboid_type(type_id: str, **overrides) -> dict:
    t = {
        "type_id": type_id,
        "part_name": f"{type_id} name",
        "primitive_type": "cuboid",
        "dimensions": {"width": 4, "depth": 4, "height": 4},
        "color_id": "red",
    }
    t.update(overrides)
    return t


def _instance(uid: str, type_id: str, parent: str | None = None, **overrides) -> dict:
    inst = {"uid": uid, "type_id": type_id, "parent_part": parent}
    if parent is not None:
        inst.setdefault("parent_face", "top")
        inst.setdefault("child_face", "bottom")
    inst.update(overrides)
    return inst


def _part(uid: str, parent: str | None = None, **overrides) -> dict:
    p = {
        "uid": uid,
        "part_name": uid,
        "primitive_type": "cuboid",
        "parent_part": parent,
        "parent_face": "top" if parent else None,
        "child_face": "bottom" if parent else None,
        "attachment_offset": 0.0,
        "attachment_offset_v": 0.0,
        "dimensions": {"width": 4, "depth": 4, "height": 4},
        "rotation": {"rx": 0, "ry": 0, "rz": 0},
        "top_radius": None,
        "critical": False,
        "color_id": "red",
    }
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# _expand_instances
# ---------------------------------------------------------------------------

class TestExpandInstances:
    def test_flattens_type_properties_onto_instances(self):
        data = {
            "part_types": [_cuboid_type("body")],
            "instances": [_instance("body_0", "body")],
        }
        parts = _expand_instances(data)
        assert len(parts) == 1
        p = parts[0]
        assert p["uid"] == "body_0"
        assert p["primitive_type"] == "cuboid"
        assert p["dimensions"] == {"width": 4, "depth": 4, "height": 4}
        assert p["color_id"] == "red"
        # Single instance of a type keeps the type's part_name
        assert p["part_name"] == "body name"

    def test_multiple_instances_use_uid_as_display_name(self):
        data = {
            "part_types": [_cuboid_type("body"), _cuboid_type("wheel")],
            "instances": [
                _instance("body_0", "body"),
                _instance("wheel_0", "wheel", parent="body_0"),
                _instance("wheel_1", "wheel", parent="body_0"),
            ],
        }
        parts = _expand_instances(data)
        names = {p["uid"]: p["part_name"] for p in parts}
        assert names["body_0"] == "body name"
        assert names["wheel_0"] == "wheel_0"
        assert names["wheel_1"] == "wheel_1"

    def test_instance_rotation_overrides_type_rotation(self):
        data = {
            "part_types": [_cuboid_type("body", rotation={"rx": 0, "ry": 0, "rz": 45})],
            "instances": [
                _instance("body_0", "body"),
                _instance("body_1", "body", parent="body_0",
                          rotation={"rx": 90, "ry": 0, "rz": 0}),
            ],
        }
        parts = _expand_instances(data)
        by_uid = {p["uid"]: p for p in parts}
        assert by_uid["body_0"]["rotation"] == {"rx": 0, "ry": 0, "rz": 45}
        assert by_uid["body_1"]["rotation"] == {"rx": 90, "ry": 0, "rz": 0}

    def test_missing_arrays_raise(self):
        with pytest.raises(ValueError):
            _expand_instances({"part_types": [], "instances": []})
        with pytest.raises(ValueError):
            _expand_instances({"part_types": [_cuboid_type("a")], "instances": []})

    def test_too_many_part_types_raises(self):
        data = {
            "part_types": [_cuboid_type(f"t{i}") for i in range(21)],
            "instances": [_instance("t0_0", "t0")],
        }
        with pytest.raises(ValueError, match="part types"):
            _expand_instances(data)

    def test_too_many_total_instances_raises(self):
        types = [_cuboid_type(f"t{i}") for i in range(7)]
        instances = [
            _instance(f"t{i}_{j}", f"t{i}")
            for i in range(7) for j in range(9)
        ]  # 63 total
        with pytest.raises(ValueError, match="instances"):
            _expand_instances({"part_types": types, "instances": instances})

    def test_too_many_instances_of_one_type_raises(self):
        data = {
            "part_types": [_cuboid_type("wheel")],
            "instances": [_instance(f"wheel_{i}", "wheel") for i in range(11)],
        }
        with pytest.raises(ValueError, match="wheel"):
            _expand_instances(data)

    def test_unknown_type_reference_raises(self):
        data = {
            "part_types": [_cuboid_type("body")],
            "instances": [_instance("ghost_0", "ghost")],
        }
        with pytest.raises(ValueError, match="unknown type"):
            _expand_instances(data)


# ---------------------------------------------------------------------------
# _apply_edits
# ---------------------------------------------------------------------------

class TestApplyEdits:
    def test_translate_updates_attachment_and_clamps_offsets(self):
        parts = [_part("root"), _part("arm_0", parent="root")]
        edits = [{
            "action": "translate", "uid": "arm_0",
            "parent_face": "left", "child_face": "right",
            "attachment_offset": 2.5, "attachment_offset_v": -3.0,
        }]
        out = _apply_edits(parts, edits)
        arm = next(p for p in out if p["uid"] == "arm_0")
        assert arm["parent_face"] == "left"
        assert arm["child_face"] == "right"
        assert arm["attachment_offset"] == 1.0   # clamped to [-1, 1]
        assert arm["attachment_offset_v"] == -1.0

    def test_rotate_replaces_rotation(self):
        parts = [_part("root")]
        edits = [{"action": "rotate", "uid": "root", "rotation": {"rz": 90}}]
        out = _apply_edits(parts, edits)
        assert out[0]["rotation"] == {"rx": 0, "ry": 0, "rz": 90}

    def test_resize_merges_partial_dimensions(self):
        parts = [_part("root")]
        edits = [{"action": "resize", "uid": "root", "dimensions": {"width": 10}}]
        out = _apply_edits(parts, edits)
        assert out[0]["dimensions"] == {"width": 10, "depth": 4, "height": 4}

    def test_recolor(self):
        parts = [_part("root")]
        edits = [{"action": "recolor", "uid": "root", "color_id": "blue"}]
        out = _apply_edits(parts, edits)
        assert out[0]["color_id"] == "blue"

    def test_toggle_critical_flips_flag(self):
        parts = [_part("root"), _part("arm_0", parent="root", critical=True)]
        edits = [{"action": "toggle_critical", "uids": ["root", "arm_0"]}]
        out = _apply_edits(parts, edits)
        by_uid = {p["uid"]: p for p in out}
        assert by_uid["root"]["critical"] is True
        assert by_uid["arm_0"]["critical"] is False

    def test_toggle_critical_guard_keeps_critical_ancestors(self):
        parts = [
            _part("root", critical=True),
            _part("arm_0", parent="root", critical=True),
        ]
        # Unsetting root is refused while a descendant is still critical.
        edits = [{"action": "toggle_critical", "uids": ["root"]}]
        out = _apply_edits(parts, edits)
        by_uid = {p["uid"]: p for p in out}
        assert by_uid["root"]["critical"] is True

    def test_add_part_generates_sequential_uid(self):
        parts = [_part("root"), _part("wheel_0", parent="root")]
        edits = [{
            "action": "add_part", "ref_id": "new_wheel",
            "part_type": _cuboid_type("wheel"),
            "instance": {"parent_part": "root", "parent_face": "left",
                         "child_face": "right"},
        }]
        out = _apply_edits(parts, edits)
        assert len(out) == 3
        added = out[-1]
        # Counter skips past the existing wheel_0
        assert added["uid"] == "wheel_1"
        assert added["parent_part"] == "root"
        assert added["primitive_type"] == "cuboid"

    def test_add_part_ref_id_resolves_as_parent_of_later_add(self):
        parts = [_part("root")]
        edits = [
            {"action": "add_part", "ref_id": "mast",
             "part_type": _cuboid_type("mast"),
             "instance": {"parent_part": "root", "parent_face": "top",
                          "child_face": "bottom"}},
            {"action": "add_part", "ref_id": "flag",
             "part_type": _cuboid_type("flag"),
             "instance": {"parent_part": "mast", "parent_face": "top",
                          "child_face": "bottom"}},
        ]
        out = _apply_edits(parts, edits)
        by_uid = {p["uid"]: p for p in out}
        assert "mast_0" in by_uid and "flag_0" in by_uid
        assert by_uid["flag_0"]["parent_part"] == "mast_0"

    def test_delete_removes_whole_subtree(self):
        parts = [
            _part("root"),
            _part("arm_0", parent="root"),
            _part("hand_0", parent="arm_0"),
            _part("leg_0", parent="root"),
        ]
        out = _apply_edits(parts, [{"action": "delete", "uid": "arm_0"}])
        uids = {p["uid"] for p in out}
        assert uids == {"root", "leg_0"}

    def test_delete_root_is_refused(self):
        parts = [_part("root"), _part("arm_0", parent="root")]
        out = _apply_edits(parts, [{"action": "delete", "uid": "root"}])
        assert {p["uid"] for p in out} == {"root", "arm_0"}

    def test_structural_budget_caps_at_5(self):
        parts = [_part("root")] + [
            _part(f"arm_{i}", parent="root") for i in range(6)
        ]
        edits = [{"action": "delete", "uid": f"arm_{i}"} for i in range(6)]
        out = _apply_edits(parts, edits)
        # 5 deletes applied, the 6th is dropped by the budget
        assert len(out) == 2
        assert "arm_5" in {p["uid"] for p in out}

    def test_local_budget_caps_at_20(self):
        parts = [_part("root")] + [
            _part(f"arm_{i}", parent="root") for i in range(21)
        ]
        edits = [
            {"action": "recolor", "uid": f"arm_{i}", "color_id": "blue"}
            for i in range(21)
        ]
        out = _apply_edits(parts, edits)
        recolored = [p for p in out if p["color_id"] == "blue"]
        assert len(recolored) == 20

    def test_unknown_uid_is_noop_and_does_not_consume_budget(self):
        parts = [_part("root")] + [
            _part(f"arm_{i}", parent="root") for i in range(20)
        ]
        edits = [{"action": "recolor", "uid": "ghost", "color_id": "blue"}] + [
            {"action": "recolor", "uid": f"arm_{i}", "color_id": "blue"}
            for i in range(20)
        ]
        out = _apply_edits(parts, edits)
        recolored = [p for p in out if p["color_id"] == "blue"]
        # The ghost edit was a no-op, so all 20 real edits fit the budget
        assert len(recolored) == 20

    def test_custom_budgets_are_respected(self):
        parts = [_part("root"), _part("arm_0", parent="root"),
                 _part("arm_1", parent="root")]
        edits = [
            {"action": "delete", "uid": "arm_0"},
            {"action": "delete", "uid": "arm_1"},
        ]
        out = _apply_edits(parts, edits, structural_budget=1)
        assert {p["uid"] for p in out} == {"root", "arm_1"}

    def test_input_list_is_not_mutated(self):
        parts = [_part("root")]
        _apply_edits(parts, [{"action": "recolor", "uid": "root",
                              "color_id": "blue"}])
        assert parts[0]["color_id"] == "red"
