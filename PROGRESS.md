# Lego AI Builder — Progress

Durable progress ledger. Source of truth for what's done and pending. Git is authoritative for "done" (merged to main = done; a branch/PR = in flight); this file holds the "why" and "what's next." See CLAUDE.md → Project state for the workflow.

## Status

**Done**
- Core pipeline implemented end-to-end: text/image input → pipeline (LLM + code) → voxel grid → 3D render. Code: `pipeline.py`, `llm.py` (Gemini), `render.py`, `schema.py`, `server.py` (FastAPI on :8000), Three.js 0.161.0 frontend in `static/`.
- Image upload support via Gemini vision (latest code change, Apr 16).
- Spec written (`SPEC.md`); test suite present (`test_pipeline.py`).
- Repo wired for Mac ⇄ web cloud sync: `CLAUDE.md`, `.claude/commands/` (spec-agent, status, place-files), and cloud-sync convention all committed.

**In progress**
- Working branch `dev`, current with `origin/dev`. No open PRs.

**Open questions**
- **Shape reversion bug** — output can revert to very simple shapes under certain conditions; root cause not yet identified. This is the primary target of the first QA pass.
- **Spec vs. code drift** — `SPEC.md` describes a 9-stage pipeline; recent design direction points toward a larger Part-world pipeline and a 100³ voxel grid. Reconcile the spec against the actual pipeline before the QA sprint so the QA agent works from accurate ground truth.
- **Not yet deployed** — Vercel hosting + app store submission still pending.

## Next up
1. QA pass: reproduce and characterize the shape-reversion bug → prioritized bug list. (This is the first real product run through the PM OS Dev & QA agent — see `~/pm-os-build`, `qa_only` mode.)
2. Scope the fixes → implement → verify against the bug list.
3. Deploy to Vercel; prepare app store submission.

## Work log
- 2026-06-02 — Created PROGRESS.md; added cloud-sync + project-state workflow to CLAUDE.md. Seeded Status/Next up from git log, code, and SPEC.
- 2026-04-16 — Added image upload support via Gemini vision.
- 2026-04-14 — Initial commit: pipeline, llm, render, schema, server, tests, SPEC.
