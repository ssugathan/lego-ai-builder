"""
Local debug server for the Lego Builder pipeline.

Usage:
    uvicorn server:app --reload --port 8000
    then open http://localhost:8000

Endpoints:
    POST /api/generate  — full pipeline: text and/or image → voxel render + session
    POST /api/feedback  — refine with user feedback using session context
    POST /api/export    — export finalized model
    GET  /              — serves static/index.html

Deprecated (kept for backward compatibility):
    POST /api/run       — run pipeline from raw parts JSON
    POST /api/validate  — validate + refine (old multi-step flow)
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import run_part_world
from schema import Part, GRID_SIZE
from render import render_projections
from llm import describe_image_bytes, generate_parts, validate_and_refine

app = FastAPI(title="Lego Builder")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LOG_PATH = Path(__file__).parent / "pipeline_log.jsonl"

# ---------------------------------------------------------------------------
# Session store (in-memory)
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def _create_session(description: str, parts: list[dict], grid: np.ndarray) -> str:
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "description": description,
        "parts": parts,
        "grid": grid,
    }
    return sid


def _update_session(session_id: str, parts: list[dict], grid: np.ndarray) -> None:
    if session_id in _sessions:
        _sessions[session_id]["parts"] = parts
        _sessions[session_id]["grid"] = grid


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
def _log_telemetry(entry: dict) -> None:
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass  # telemetry is best-effort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parts_dicts_to_objects(parts_dicts: list[dict]) -> list[Part]:
    """Convert raw part dicts to Part pydantic objects, tolerating legacy fields."""
    out = []
    for d in parts_dicts:
        # Handle legacy attach_face → parent_face/child_face
        if "attach_face" in d and "parent_face" not in d:
            d["parent_face"] = d.pop("attach_face")
        if "attach_face" in d:
            d.pop("attach_face", None)
        out.append(Part(**d))
    return out


def _run_pipeline(parts_dicts: list[dict]) -> dict:
    """Run the voxel pipeline. Returns the debug dict from run_part_world."""
    part_objects = _parts_dicts_to_objects(parts_dicts)
    return run_part_world(part_objects, debug=True)


def _build_voxel_response(
    result: dict, parts_dicts: list[dict], part_objects: list[Part] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    Build the voxels list, parts metadata, and stats dict from pipeline result.

    Returns (voxels, parts_meta, stats).
    """
    grid: np.ndarray = result["grid"]
    states = result["states"]
    voxel_counts: dict[str, int] = result["voxel_counts"]
    total_occupied: int = result["total_occupied"]

    # Build uid → dict lookup
    by_uid = {d.get("uid", ""): d for d in parts_dicts}

    # Ordered part metadata aligned with states (topological order post-pipeline)
    parts_meta = []
    for i, s in enumerate(states):
        d = by_uid.get(s.uid, {})
        parts_meta.append({
            "idx": i + 1,
            "uid": s.uid,
            "name": d.get("part_name", s.uid),
            "type": d.get("primitive_type", ""),
            "critical": d.get("critical", False),
            "voxel_count": voxel_counts.get(s.uid, 0),
            "color_id": d.get("color_id", ""),
        })

    # Sparse voxel list with color_id and uid
    uid_by_idx = {i + 1: s.uid for i, s in enumerate(states)}
    color_by_idx = {}
    for pm in parts_meta:
        color_by_idx[pm["idx"]] = pm["color_id"]

    occ = np.argwhere(grid > 0)
    voxels = []
    for r in occ:
        idx = int(grid[r[0], r[1], r[2]])
        voxels.append({
            "x": int(r[0]),
            "y": int(r[1]),
            "z": int(r[2]),
            "color_id": color_by_idx.get(idx, ""),
            "uid": uid_by_idx.get(idx, ""),
            # Keep part_idx for backward compat with old renderer
            "part_idx": idx,
        })

    gs = GRID_SIZE
    stats = {
        "total_occupied": total_occupied,
        "grid_size": [gs, gs, gs],
        "part_count": len(parts_meta),
    }

    return voxels, parts_meta, stats


def _decode_request_image(image_field: str | None) -> tuple[bytes, str] | None:
    """
    Decode optional base64 image from JSON body.
    Accepts raw base64 or a data URL (data:image/png;base64,...).
    Returns (bytes, mime_type) or None if field empty.
    """
    if image_field is None:
        return None
    s = str(image_field).strip()
    if not s:
        return None

    mime_type = "image/jpeg"
    if s.startswith("data:"):
        m = re.match(r"data:([^;]+);base64,(.+)", s, re.DOTALL | re.IGNORECASE)
        if not m:
            raise ValueError("Invalid image data URL (expected data:<mime>;base64,...)")
        mime_type = m.group(1).strip() or mime_type
        b64 = m.group(2).strip()
    else:
        b64 = s

    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image: {exc}") from exc

    max_bytes = 20 * 1024 * 1024
    if len(raw) > max_bytes:
        raise ValueError(f"Image too large (max {max_bytes // (1024 * 1024)} MB)")

    return raw, mime_type


def _render_and_validate(
    grid: np.ndarray, parts_meta: list[dict], parts_dicts: list[dict],
    description: str, api_key: str,
) -> dict:
    """Run projection rendering + LLM validation. Returns validation result dict."""
    projections = render_projections(grid, parts_meta)
    images = [(img_bytes, mime) for img_bytes, mime, _label in projections]
    return validate_and_refine(images, description, parts_dicts, api_key)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    description: str = ""
    image: str | None = None  # raw base64 or data:image/...;base64,...
    api_key: str | None = None


class FeedbackRequest(BaseModel):
    session_id: str
    feedback: str
    api_key: str | None = None


class ExportRequest(BaseModel):
    session_id: str
    format: str = "voxel_json"


# Deprecated request models (backward compat)
class RunRequest(BaseModel):
    parts: list[Part]
    debug: bool = False


class ValidateRequest(BaseModel):
    parts: list[Part]
    description: str
    debug: bool = False


# ---------------------------------------------------------------------------
# POST /api/generate — full pipeline
# ---------------------------------------------------------------------------
@app.post("/api/generate")
def api_generate(req: GenerateRequest) -> JSONResponse:
    """
    Full pipeline: description and/or image → (optional vision) → generate parts →
    run pipeline → validate → apply edits → rebuild → return voxels.
    """
    key = req.api_key or GEMINI_API_KEY
    if not key:
        return JSONResponse(
            {"error": "No API key. Set GEMINI_API_KEY env var or pass api_key in request."},
            status_code=400,
        )

    user_description = (req.description or "").strip()
    try:
        image_payload = _decode_request_image(req.image)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    if not user_description and not image_payload:
        return JSONResponse(
            {"error": "Provide a text description and/or upload an image."},
            status_code=400,
        )

    t0 = time.monotonic()
    modality = "text"
    if image_payload and user_description:
        modality = "both"
    elif image_payload:
        modality = "image"

    telemetry: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/api/generate",
        "description_length": len(user_description),
        "modality": modality,
        "stages": {},
    }

    try:
        recognized_text: str | None = None

        # ── Stage 1a: Vision — image → recognition text ─────────
        if image_payload:
            img_bytes, img_mime = image_payload
            t0a = time.monotonic()
            recognized_text, img_stats = describe_image_bytes(img_bytes, img_mime, key)
            t0a_end = time.monotonic()
            telemetry["stages"]["S1a_LLM_ImageRecognition"] = {
                "label": "Stage 1a — Gemini recognizes uploaded image",
                "latency_ms": int((t0a_end - t0a) * 1000),
                "tokens": {
                    "input": img_stats.get("input_tokens", 0),
                    "output": img_stats.get("output_tokens", 0),
                    "thinking": img_stats.get("thinking_tokens", 0),
                    "cached": img_stats.get("cached_tokens", 0),
                    "total": img_stats.get("total_tokens", 0),
                },
            }

        if user_description and recognized_text:
            effective_description = (
                f"{user_description}\n\n"
                f"Reference subject from uploaded image:\n{recognized_text}"
            )
        elif recognized_text:
            effective_description = recognized_text
        else:
            effective_description = user_description

        telemetry["description_length"] = len(effective_description)

        # ── Stage 1: LLM Generation ──────────────────────────────
        t1 = time.monotonic()
        parts_dicts, strategy, gen_stats = generate_parts(effective_description, key)
        t1_end = time.monotonic()
        telemetry["stages"]["S1_LLM_Generation"] = {
            "label": "Stage 1 — Gemini generates part structure from description",
            "latency_ms": int((t1_end - t1) * 1000),
            "parts_generated": len(parts_dicts),
            "tokens": {
                "input": gen_stats.get("input_tokens", 0),
                "output": gen_stats.get("output_tokens", 0),
                "thinking": gen_stats.get("thinking_tokens", 0),
                "cached": gen_stats.get("cached_tokens", 0),
                "total": gen_stats.get("total_tokens", 0),
            },
        }

        # ── Stages 2-3: Deterministic Pipeline ───────────────────
        t2 = time.monotonic()
        result = _run_pipeline(parts_dicts)
        t2_end = time.monotonic()
        telemetry["stages"]["S2_S3_Pipeline"] = {
            "label": "Stages 2-3 — Structural cleanup + voxelization (deterministic)",
            "latency_ms": int((t2_end - t2) * 1000),
            "occupied_voxels": result["total_occupied"],
            "part_count": len(result["states"]),
        }

        voxels, parts_meta, stats = _build_voxel_response(result, parts_dicts)

        # ── Stage 5: LLM Validation ──────────────────────────────
        t5 = time.monotonic()
        val_result = _render_and_validate(
            result["grid"], parts_meta, parts_dicts, effective_description, key,
        )
        t5_end = time.monotonic()
        val_stats = val_result.get("_call_stats", {})
        telemetry["stages"]["S5_LLM_Validation"] = {
            "label": "Stage 5 — Gemini evaluates projections and proposes edits",
            "latency_ms": int((t5_end - t5) * 1000),
            "mode": "validation",
            "confidence": val_result.get("confidence", ""),
            "edits_proposed": val_result.get("edit_count", 0),
            "no_edits_needed": val_result.get("no_edits_needed", False),
            "tokens": {
                "input": val_stats.get("input_tokens", 0),
                "output": val_stats.get("output_tokens", 0),
                "thinking": val_stats.get("thinking_tokens", 0),
                "cached": val_stats.get("cached_tokens", 0),
                "total": val_stats.get("total_tokens", 0),
            },
        }

        # ── Stage 6 + Rebuild: Apply edits ───────────────────────
        edits_applied = 0
        edits_rejected = 0
        refined_parts = val_result.get("refined_parts")

        if refined_parts and val_result.get("edit_count", 0) > 0:
            t6 = time.monotonic()
            edits_applied = val_result.get("edit_count", 0)
            parts_dicts = refined_parts

            # Rebuild with refined parts
            result = _run_pipeline(parts_dicts)
            voxels, parts_meta, stats = _build_voxel_response(result, parts_dicts)
            t6_end = time.monotonic()
            telemetry["stages"]["S6_Edit_Execution_and_Rebuild"] = {
                "label": "Stage 6 — Apply edits + rebuild pipeline (deterministic)",
                "latency_ms": int((t6_end - t6) * 1000),
                "edits_applied": edits_applied,
                "post_rebuild_voxels": result["total_occupied"],
            }

        # Create session (full generation context for feedback / export)
        session_id = _create_session(effective_description, parts_dicts, result["grid"])
        telemetry["session_id"] = session_id

        validation_summary = {
            "guess": val_result.get("guess", ""),
            "confidence": val_result.get("confidence", ""),
            "issues": val_result.get("issues", []),
            "edits_applied": edits_applied,
            "edits_rejected": edits_rejected,
        }

        response = {
            "voxels": voxels,
            "parts": parts_meta,
            "stats": stats,
            "validation": validation_summary,
            "session_id": session_id,
        }

        if strategy:
            response["strategy"] = strategy

        if recognized_text is not None:
            response["image_recognition"] = recognized_text

    except Exception as exc:
        telemetry["error"] = str(exc)
        _log_telemetry(telemetry)
        return JSONResponse(
            {"error": str(exc), "detail": traceback.format_exc()},
            status_code=400,
        )

    # Aggregate token totals across all LLM calls
    total_tokens = 0
    total_input = 0
    total_output = 0
    for stage in telemetry["stages"].values():
        t = stage.get("tokens", {})
        total_tokens += t.get("total", 0)
        total_input += t.get("input", 0)
        total_output += t.get("output", 0)

    telemetry["totals"] = {
        "end_to_end_latency_ms": int((time.monotonic() - t0) * 1000),
        "total_llm_calls": sum(1 for s in telemetry["stages"] if "tokens" in telemetry["stages"][s]),
        "total_tokens": total_tokens,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "part_count": stats.get("part_count", 0),
        "occupied_voxels": stats.get("total_occupied", 0),
    }
    _log_telemetry(telemetry)
    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# POST /api/feedback — refine with user feedback
# ---------------------------------------------------------------------------
@app.post("/api/feedback")
def api_feedback(req: FeedbackRequest) -> JSONResponse:
    """
    Refine the model using natural-language feedback.
    Requires a session_id from a prior /api/generate call.
    """
    session = _get_session(req.session_id)
    if session is None:
        return JSONResponse(
            {"error": f"Unknown session_id: {req.session_id}"},
            status_code=404,
        )

    key = req.api_key or GEMINI_API_KEY
    if not key:
        return JSONResponse(
            {"error": "No API key. Set GEMINI_API_KEY env var or pass api_key in request."},
            status_code=400,
        )

    t0 = time.monotonic()
    telemetry: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/api/feedback",
        "session_id": req.session_id,
        "stages": {},
    }

    try:
        parts_dicts = session["parts"]
        description = session["description"]
        grid = session["grid"]

        # Build parts_meta for projections
        part_objects = _parts_dicts_to_objects(parts_dicts)
        temp_result = run_part_world(part_objects, debug=True)
        by_uid = {d.get("uid", ""): d for d in parts_dicts}
        parts_meta = []
        for i, s in enumerate(temp_result["states"]):
            d = by_uid.get(s.uid, {})
            parts_meta.append({
                "idx": i + 1,
                "uid": s.uid,
                "name": d.get("part_name", s.uid),
                "type": d.get("primitive_type", ""),
                "critical": d.get("critical", False),
                "voxel_count": temp_result["voxel_counts"].get(s.uid, 0),
                "color_id": d.get("color_id", ""),
            })

        # Render projections from current grid
        projections = render_projections(temp_result["grid"], parts_meta)

        # Call feedback_refine (provided by another agent)
        t5 = time.monotonic()
        try:
            from llm import feedback_refine
            images = [(img_bytes, mime) for img_bytes, mime, _label in projections]
            fb_result = feedback_refine(images, description, parts_dicts, req.feedback, key)
        except ImportError:
            # feedback_refine not yet implemented — fall back to validate_and_refine
            images = [(img_bytes, mime) for img_bytes, mime, _label in projections]
            fb_result = validate_and_refine(images, description, parts_dicts, key)
            fb_result["feedback_interpretation"] = "Used validation as fallback (feedback_refine not yet available)"

        t5_end = time.monotonic()
        fb_stats = fb_result.get("_call_stats", {})
        telemetry["stages"]["S5_LLM_Feedback"] = {
            "label": "Stage 5 (feedback mode) — Gemini interprets user feedback into edits",
            "latency_ms": int((t5_end - t5) * 1000),
            "mode": "feedback",
            "confidence": fb_result.get("confidence", ""),
            "edits_proposed": fb_result.get("edit_count", 0),
            "tokens": {
                "input": fb_stats.get("input_tokens", 0),
                "output": fb_stats.get("output_tokens", 0),
                "thinking": fb_stats.get("thinking_tokens", 0),
                "cached": fb_stats.get("cached_tokens", 0),
                "total": fb_stats.get("total_tokens", 0),
            },
        }

        # ── Stage 6 + Rebuild ────────────────────────────────────
        edits_applied = 0
        edits_rejected = 0
        refined_parts = fb_result.get("refined_parts")

        if refined_parts and fb_result.get("edit_count", 0) > 0:
            t6 = time.monotonic()
            edits_applied = fb_result.get("edit_count", 0)
            parts_dicts = refined_parts

            result = _run_pipeline(parts_dicts)
            voxels, parts_meta, stats = _build_voxel_response(result, parts_dicts)
            _update_session(req.session_id, parts_dicts, result["grid"])

            t6_end = time.monotonic()
            telemetry["stages"]["S6_Edit_Execution_and_Rebuild"] = {
                "label": "Stage 6 — Apply feedback edits + rebuild pipeline (deterministic)",
                "latency_ms": int((t6_end - t6) * 1000),
                "edits_applied": edits_applied,
                "post_rebuild_voxels": result["total_occupied"],
            }
        else:
            # No edits — return current state
            voxels, parts_meta, stats = _build_voxel_response(temp_result, parts_dicts)

        response = {
            "voxels": voxels,
            "parts": parts_meta,
            "stats": stats,
            "feedback_interpretation": fb_result.get("feedback_interpretation", fb_result.get("guess", "")),
            "confidence": fb_result.get("confidence", ""),
            "edits_applied": edits_applied,
            "edits_rejected": edits_rejected,
            "session_id": req.session_id,
        }

    except Exception as exc:
        telemetry["error"] = str(exc)
        _log_telemetry(telemetry)
        return JSONResponse(
            {"error": str(exc), "detail": traceback.format_exc()},
            status_code=400,
        )

    total_tokens = 0
    total_input = 0
    total_output = 0
    for stage in telemetry["stages"].values():
        t = stage.get("tokens", {})
        total_tokens += t.get("total", 0)
        total_input += t.get("input", 0)
        total_output += t.get("output", 0)

    telemetry["totals"] = {
        "end_to_end_latency_ms": int((time.monotonic() - t0) * 1000),
        "total_llm_calls": sum(1 for s in telemetry["stages"] if "tokens" in telemetry["stages"][s]),
        "total_tokens": total_tokens,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
    }
    _log_telemetry(telemetry)
    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# POST /api/export
# ---------------------------------------------------------------------------
@app.post("/api/export")
def api_export(req: ExportRequest) -> JSONResponse:
    """Export the finalized model in the requested format."""
    session = _get_session(req.session_id)
    if session is None:
        return JSONResponse(
            {"error": f"Unknown session_id: {req.session_id}"},
            status_code=404,
        )

    parts_dicts = session["parts"]

    if req.format == "metadata_json":
        return JSONResponse(content={
            "parts": parts_dicts,
            "description": session["description"],
        })

    # Default: voxel_json
    result = _run_pipeline(parts_dicts)
    voxels, parts_meta, stats = _build_voxel_response(result, parts_dicts)
    return JSONResponse(content={
        "voxels": voxels,
        "stats": stats,
    })


# ---------------------------------------------------------------------------
# GET /api/telemetry — view recent pipeline logs (JSON)
# ---------------------------------------------------------------------------
@app.get("/api/telemetry")
def api_telemetry() -> JSONResponse:
    """Return the last 50 pipeline log entries."""
    if not LOG_PATH.exists():
        return JSONResponse(content={"entries": []})
    lines = LOG_PATH.read_text().strip().split("\n")
    entries = []
    for line in lines[-50:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return JSONResponse(content={"entries": entries})


RESPONSE_LOG_PATH = Path(__file__).parent / "gemini_responses.jsonl"

@app.get("/api/responses")
def api_responses(last: int = 10) -> JSONResponse:
    """Return the last N Gemini responses for debugging."""
    if not RESPONSE_LOG_PATH.exists():
        return JSONResponse(content={"entries": []})
    lines = RESPONSE_LOG_PATH.read_text().strip().split("\n")
    entries = []
    for line in lines[-last:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return JSONResponse(content={"entries": entries})


# ---------------------------------------------------------------------------
# GET /telemetry — dashboard UI
# ---------------------------------------------------------------------------
from fastapi.responses import HTMLResponse

@app.get("/telemetry", response_class=HTMLResponse)
def telemetry_dashboard():
    """Telemetry dashboard with top-level metrics and per-job stage breakdown."""
    return HTMLResponse(content=_TELEMETRY_HTML)


_TELEMETRY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pipeline Telemetry</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #111; color: #e0e0e0; padding: 20px; }
  h1 { color: #fff; margin-bottom: 16px; font-size: 22px; }
  .tabs { display: flex; gap: 0; margin-bottom: 20px; border-bottom: 2px solid #333; }
  .tab { padding: 10px 24px; cursor: pointer; color: #888; font-size: 14px;
         border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s; }
  .tab:hover { color: #ccc; }
  .tab.active { color: #4fc3f7; border-bottom-color: #4fc3f7; }
  .panel { display: none; }
  .panel.active { display: block; }

  /* Summary cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
           gap: 12px; margin-bottom: 24px; }
  .card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 16px; }
  .card .label { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { color: #fff; font-size: 28px; font-weight: 600; margin-top: 4px; }
  .card .unit { color: #666; font-size: 13px; }

  /* Jobs table */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; color: #888; font-weight: 500;
       border-bottom: 1px solid #333; font-size: 11px; text-transform: uppercase; }
  td { padding: 8px 12px; border-bottom: 1px solid #222; vertical-align: top; }
  tr:hover { background: #1a1a1a; }
  .mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .tag-llm { background: #1a3a4a; color: #4fc3f7; }
  .tag-code { background: #1a3a1a; color: #66bb6a; }
  .tag-err { background: #3a1a1a; color: #ef5350; }
  .confidence-high { color: #66bb6a; }
  .confidence-medium { color: #ffa726; }
  .confidence-low { color: #ef5350; }

  /* Stage breakdown */
  .job-detail { display: none; }
  .job-detail.open { display: table-row; }
  .job-detail td { padding: 0; }
  .stage-table { width: 100%; margin: 0; background: #0d0d0d; }
  .stage-table td { padding: 6px 12px 6px 32px; border-bottom: 1px solid #1a1a1a; }
  .stage-table .stage-name { color: #4fc3f7; font-weight: 500; }
  .bar-bg { background: #222; border-radius: 3px; height: 6px; width: 120px; display: inline-block; vertical-align: middle; }
  .bar-fill { height: 6px; border-radius: 3px; display: inline-block; }
  .bar-llm { background: #4fc3f7; }
  .bar-code { background: #66bb6a; }
  .tok { color: #888; font-size: 11px; }
  .clickable { cursor: pointer; }
  .expand-icon { color: #555; margin-right: 6px; transition: transform 0.2s; display: inline-block; }
  .expand-icon.open { transform: rotate(90deg); }
  .refresh-btn { padding: 6px 16px; background: #1a1a1a; border: 1px solid #333;
                 color: #ccc; border-radius: 6px; cursor: pointer; font-size: 12px; margin-left: 12px; }
  .refresh-btn:hover { background: #252525; }
  .empty { color: #555; text-align: center; padding: 40px; }
</style>
</head>
<body>
<div style="display: flex; align-items: center;">
  <h1>Pipeline Telemetry</h1>
  <button class="refresh-btn" onclick="loadData()">Refresh</button>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('summary')">Summary</div>
  <div class="tab" onclick="switchTab('jobs')">Jobs</div>
</div>

<div id="summary" class="panel active"></div>
<div id="jobs" class="panel"></div>

<script>
let entries = [];

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(tab).classList.add('active');
}

function fmt(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return ms + 'ms';
  return (ms / 1000).toFixed(1) + 's';
}

function fmtTokens(n) {
  if (n == null || n === 0) return '—';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return String(n);
}

function confClass(c) {
  if (c === 'high') return 'confidence-high';
  if (c === 'medium') return 'confidence-medium';
  return 'confidence-low';
}

function renderSummary() {
  const el = document.getElementById('summary');
  if (entries.length === 0) { el.innerHTML = '<div class="empty">No telemetry data yet. Generate an object first.</div>'; return; }

  const total = entries.length;
  const genJobs = entries.filter(e => e.endpoint === '/api/generate');
  const fbJobs = entries.filter(e => e.endpoint === '/api/feedback');
  const errors = entries.filter(e => e.error);

  const latencies = entries.map(e => e.totals?.end_to_end_latency_ms).filter(Boolean).sort((a,b) => a-b);
  const p50 = latencies[Math.floor(latencies.length * 0.5)] || 0;
  const p95 = latencies[Math.floor(latencies.length * 0.95)] || 0;

  const totalTokens = entries.reduce((s, e) => s + (e.totals?.total_tokens || 0), 0);
  const totalInput = entries.reduce((s, e) => s + (e.totals?.total_input_tokens || 0), 0);
  const totalOutput = entries.reduce((s, e) => s + (e.totals?.total_output_tokens || 0), 0);
  const avgTokens = total > 0 ? Math.round(totalTokens / total) : 0;

  const llmCalls = entries.reduce((s, e) => s + (e.totals?.total_llm_calls || 0), 0);

  el.innerHTML = `
    <div class="cards">
      <div class="card"><div class="label">Total Requests</div><div class="value">${total}</div>
        <div class="unit">${genJobs.length} generate · ${fbJobs.length} feedback</div></div>
      <div class="card"><div class="label">Errors</div><div class="value" style="color:${errors.length?'#ef5350':'#66bb6a'}">${errors.length}</div>
        <div class="unit">${total > 0 ? ((errors.length/total)*100).toFixed(0) : 0}% error rate</div></div>
      <div class="card"><div class="label">P50 Latency</div><div class="value">${fmt(p50)}</div>
        <div class="unit">end-to-end</div></div>
      <div class="card"><div class="label">P95 Latency</div><div class="value">${fmt(p95)}</div>
        <div class="unit">end-to-end</div></div>
      <div class="card"><div class="label">Total Tokens</div><div class="value">${fmtTokens(totalTokens)}</div>
        <div class="unit">${fmtTokens(totalInput)} in · ${fmtTokens(totalOutput)} out</div></div>
      <div class="card"><div class="label">Avg Tokens / Request</div><div class="value">${fmtTokens(avgTokens)}</div>
        <div class="unit">${llmCalls} total LLM calls</div></div>
    </div>
  `;
}

function toggleJob(idx) {
  const row = document.getElementById('detail-' + idx);
  const icon = document.getElementById('icon-' + idx);
  if (row) { row.classList.toggle('open'); icon.classList.toggle('open'); }
}

function renderJobs() {
  const el = document.getElementById('jobs');
  if (entries.length === 0) { el.innerHTML = '<div class="empty">No telemetry data yet.</div>'; return; }

  const reversed = [...entries].reverse();
  let html = `<table>
    <tr><th></th><th>Time</th><th>Endpoint</th><th>Latency</th><th>Tokens</th><th>LLM Calls</th><th>Parts</th><th>Status</th></tr>`;

  reversed.forEach((e, i) => {
    const ts = e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : '—';
    const ep = e.endpoint || '—';
    const lat = fmt(e.totals?.end_to_end_latency_ms);
    const tok = fmtTokens(e.totals?.total_tokens);
    const llm = e.totals?.total_llm_calls || 0;
    const parts = e.totals?.part_count || e.stages?.S2_S3_Pipeline?.part_count || '—';
    const err = e.error;
    const statusTag = err
      ? `<span class="tag tag-err">error</span>`
      : `<span class="tag tag-code">ok</span>`;

    html += `<tr class="clickable" onclick="toggleJob(${i})">
      <td><span class="expand-icon" id="icon-${i}">▶</span></td>
      <td class="mono">${ts}</td>
      <td class="mono">${ep}</td>
      <td class="mono">${lat}</td>
      <td class="mono">${tok}</td>
      <td>${llm}</td>
      <td>${parts}</td>
      <td>${statusTag}</td>
    </tr>`;

    // Stage breakdown row
    html += `<tr class="job-detail" id="detail-${i}"><td colspan="8">`;
    const stages = e.stages || {};
    const stageKeys = Object.keys(stages);
    if (stageKeys.length === 0) {
      html += `<div style="padding:12px 32px;color:#555;">No stage data</div>`;
    } else {
      // Find max latency for bar scaling
      const maxLat = Math.max(...stageKeys.map(k => stages[k].latency_ms || 0), 1);
      html += `<table class="stage-table">`;
      stageKeys.forEach(k => {
        const s = stages[k];
        const isLLM = k.includes('LLM');
        const barClass = isLLM ? 'bar-llm' : 'bar-code';
        const tagClass = isLLM ? 'tag-llm' : 'tag-code';
        const tagLabel = isLLM ? 'LLM' : 'CODE';
        const lat = s.latency_ms || 0;
        const pct = Math.max(2, (lat / maxLat) * 100);
        const tokens = s.tokens;

        let details = [];
        if (s.parts_generated) details.push(s.parts_generated + ' parts');
        if (s.occupied_voxels) details.push(s.occupied_voxels.toLocaleString() + ' voxels');
        if (s.part_count) details.push(s.part_count + ' parts');
        if (s.confidence) details.push(`<span class="${confClass(s.confidence)}">${s.confidence}</span>`);
        if (s.edits_proposed) details.push(s.edits_proposed + ' edits');
        if (s.edits_applied) details.push(s.edits_applied + ' applied');
        if (s.post_rebuild_voxels) details.push(s.post_rebuild_voxels.toLocaleString() + ' voxels');

        let tokenStr = '';
        if (tokens && tokens.total) {
          tokenStr = `<span class="tok">${fmtTokens(tokens.total)} tok (${fmtTokens(tokens.input)} in · ${fmtTokens(tokens.output)} out`;
          if (tokens.thinking) tokenStr += ` · ${fmtTokens(tokens.thinking)} think`;
          if (tokens.cached) tokenStr += ` · ${fmtTokens(tokens.cached)} cached`;
          tokenStr += `)</span>`;
        }

        html += `<tr>
          <td class="stage-name">${s.label || k}</td>
          <td style="width:160px">
            <div class="bar-bg"><div class="bar-fill ${barClass}" style="width:${pct}%"></div></div>
            <span class="mono" style="margin-left:8px">${fmt(lat)}</span>
          </td>
          <td><span class="tag ${tagClass}">${tagLabel}</span></td>
          <td>${details.join(' · ')}</td>
          <td>${tokenStr}</td>
        </tr>`;
      });
      html += `</table>`;
    }

    if (e.error) {
      html += `<div style="padding:8px 32px;color:#ef5350;font-size:12px;">Error: ${e.error}</div>`;
    }
    html += `</td></tr>`;
  });

  html += `</table>`;
  el.innerHTML = html;
}

async function loadData() {
  try {
    const resp = await fetch('/api/telemetry');
    const data = await resp.json();
    entries = data.entries || [];
    renderSummary();
    renderJobs();
  } catch (err) {
    document.getElementById('summary').innerHTML = '<div class="empty">Failed to load telemetry: ' + err + '</div>';
  }
}

loadData();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# DEPRECATED: POST /api/run  (old multi-step flow)
# ---------------------------------------------------------------------------
@app.post("/api/run")
def api_run(req: RunRequest) -> JSONResponse:
    """
    [DEPRECATED] Run the Part-world pipeline from raw parts JSON.
    Use POST /api/generate instead for the full pipeline flow.
    """
    try:
        result = run_part_world(req.parts, debug=True)
    except Exception as exc:
        return JSONResponse(
            {"error": str(exc), "detail": traceback.format_exc()},
            status_code=400,
        )

    grid: np.ndarray = result["grid"]
    states = result["states"]
    voxel_counts: dict[str, int] = result["voxel_counts"]
    total_occupied: int = result["total_occupied"]

    by_uid = {p.uid: p for p in req.parts}

    parts_meta = [
        {
            "idx": i + 1,
            "uid": s.uid,
            "name": by_uid[s.uid].part_name if s.uid in by_uid else s.uid,
            "critical": by_uid[s.uid].critical if s.uid in by_uid else False,
            "voxel_count": voxel_counts.get(s.uid, 0),
            "color_id": by_uid[s.uid].color_id if s.uid in by_uid else "",
        }
        for i, s in enumerate(states)
    ]

    occ = np.argwhere(grid > 0)
    voxels = [
        {
            "x": int(r[0]),
            "y": int(r[1]),
            "z": int(r[2]),
            "part_idx": int(grid[r[0], r[1], r[2]]),
        }
        for r in occ
    ]

    response: dict = {
        "voxels": voxels,
        "parts": parts_meta,
        "stats": {
            "total_occupied": total_occupied,
            "grid_size": {"x": 50, "y": 50, "z": 100},
        },
    }

    if req.debug:
        response["debug"] = {
            "scale": float(result["scale"]),
            "voxel_counts": voxel_counts,
        }

    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# DEPRECATED: POST /api/validate  (old multi-step flow)
# ---------------------------------------------------------------------------
@app.post("/api/validate")
def api_validate(req: ValidateRequest) -> JSONResponse:
    """
    [DEPRECATED] Validate + refine from raw parts JSON.
    Use POST /api/generate + POST /api/feedback instead.
    """
    key = GEMINI_API_KEY
    if not key:
        return JSONResponse(
            {"error": "No GEMINI_API_KEY set for validation."},
            status_code=400,
        )

    try:
        result = run_part_world(req.parts, debug=True)
        grid = result["grid"]
        by_uid = {p.uid: p for p in req.parts}

        parts_meta = [
            {
                "idx": i + 1, "uid": s.uid,
                "name": by_uid[s.uid].part_name if s.uid in by_uid else s.uid,
                "critical": by_uid[s.uid].critical if s.uid in by_uid else False,
                "voxel_count": result["voxel_counts"].get(s.uid, 0),
                "color_id": by_uid[s.uid].color_id if s.uid in by_uid else "",
            }
            for i, s in enumerate(result["states"])
        ]

        projections = render_projections(grid, parts_meta)
        images = [(img_bytes, mime) for img_bytes, mime, _label in projections]

        parts_for_llm = [p.model_dump(mode="json") for p in req.parts]
        val_result = validate_and_refine(images, req.description, parts_for_llm, key)

        refined_parts = val_result.pop("refined_parts", None)
        response = {"validation": val_result}

        if refined_parts and val_result.get("edit_count", 0) > 0:
            response["refined_parts"] = refined_parts

    except Exception as exc:
        return JSONResponse(
            {"error": str(exc), "detail": traceback.format_exc()},
            status_code=400,
        )

    return JSONResponse(content=response)


# Serve the frontend.  Must be mounted after all API routes.
_static = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
