# Lego AI Builder — Progress

Durable progress ledger. Source of truth for what's done and pending. Git is authoritative for "done" (merged to main = done; a branch/PR = in flight); this file holds the "why" and "what's next." See AGENTS.md → Project state for the workflow.

## Status

**Done**
- Core pipeline implemented end-to-end: text/image input → pipeline (LLM + code) → voxel grid → 3D render. Code: `pipeline.py`, `llm.py` (Gemini), `render.py`, `schema.py`, `server.py` (FastAPI on :8000), Three.js 0.161.0 frontend in `static/`.
- Image upload support via Gemini vision (latest code change, Apr 16).
- Spec written (`SPEC.md`); test suite present (`test_pipeline.py`).
- Repo wired for Mac ⇄ web cloud sync: `CLAUDE.md`, `.claude/commands/` (spec-agent, status, place-files), and cloud-sync convention all committed.

**In progress (as of 2026-09-05)**
- Three branches ready to merge to main via PR, in order: (1) `dev` (this branch lineage: CLAUDE.md, slash commands, PROGRESS.md); (2) `fix/demo-security` (contains `refactor/safe-pass`; Sep 2 refactor: fixed requirements.txt, vectorized renderer with golden tests, dead prototype removed, README rewritten, +68 tests — then security pass: tracebacks out of responses, debug endpoints gated, deprecated endpoints removed, session LRU, server-side input limits; 117/117 tests green); (3) `docs/agents-setup` (AGENTS.md canonical, CLAUDE.md shim, this update).

**Open questions**
- **Shape reversion bug** — output can revert to very simple shapes under certain conditions; root cause not yet identified. This is the primary target of the first QA pass.
- **Spec vs. code drift — mostly resolved (Sep 2 assessment):** the code IS the Part-world pipeline on a 100³ grid; the rewritten README describes it accurately. Remaining spec-ahead-of-code items: SPEC §0 InputBundle (server-side MIME validation), §5.0 segmentation renders in validation, §5.1 original image passed to the validator. These are the real finish line for the image-input feature.
- **Not yet deployed** — Vercel hosting + app store submission still pending.

## Next up
1. Merge the three PRs to main in order (dev, fix/demo-security, docs/agents-setup).
2. QA pass: reproduce and characterize the shape-reversion bug → prioritized bug list. (First real product run through the PM OS Dev & QA agent — see `~/pm-os-build`, `qa_only` mode.)
3. Scope the fixes → implement → verify against the bug list.
4. Deploy to Vercel; before any public redeploy, confirm fix/demo-security is merged; prepare app store submission.

## Work log
- 2026-09-05 — Multi-agent setup: AGENTS.md canonical, CLAUDE.md import shim, .env.example; ledger updated with Sep 2 refactor + security branches.
- 2026-09-02 — refactor/safe-pass (6 commits) + fix/demo-security (6 commits) built and verified in cloud session; see In progress.
- 2026-06-02 — Created PROGRESS.md; added cloud-sync + project-state workflow to CLAUDE.md. Seeded Status/Next up from git log, code, and SPEC.
- 2026-04-16 — Added image upload support via Gemini vision.
- 2026-04-14 — Initial commit: pipeline, llm, render, schema, server, tests, SPEC.
