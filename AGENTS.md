# Lego AI Builder — Agent Instructions

Canonical instructions for ANY coding agent working in this repo (Claude Code, Codex, Cursor, or others). CLAUDE.md imports this file; do not duplicate content there.

## What this is

Text or image → Gemini 2.5 Pro parts generation → deterministic 12-stage voxel pipeline on a 100^3 grid → 6-view render → LLM self-validation and refinement → Three.js viewer. `README.md` (post-refactor) is the accurate product description; `SPEC.md` is the detailed design doc and is ahead of the code in places (input validation, segmentation-in-validation).

## Repo structure

- `pipeline.py` — deterministic Part-world voxel pipeline (attach, scale, place, rotate, voxelize, ownership, connectivity)
- `llm.py` — Gemini integration (generation, image description, validation/refinement, edit application)
- `render.py` — voxel grid → 6 orthographic projections
- `schema.py` — pydantic models, GRID_SIZE
- `server.py` — FastAPI app; static frontend in `static/`
- `test_pipeline.py`, `test_render.py`, `test_llm_edits.py` — pytest suites
- `PROGRESS.md` — cross-session status ledger (see Project state below)

## Commands

- `pip install -r requirements.txt` — install
- `cp .env.example .env` and set `GEMINI_API_KEY` — live runs only; tests need no key or network
- `uvicorn server:app --reload` — run locally at :8000
- `pytest` — run all suites (must be green before any push)

## Conventions

- No secrets in code or committed files, ever; keys live in `.env` (gitignored).
- Debug/telemetry endpoints stay behind `DEBUG_ENDPOINTS=1`; never expose them in a public deploy.
- New pipeline or LLM-edit logic ships with tests; render changes ship with golden tests against the previous implementation.
- This repo is public interview proof-of-work: keep the README accurate and free of scratch files.

## Cloud sync

This repo moves between a Mac, Claude Code on the web, and other coding agents, with GitHub as the source of truth. Keep the remote current.

- At session start: fetch and fast-forward the working branch before doing anything. If it can't fast-forward, stop and tell me — don't force.
- Ask before pushing: when a logical block is done — a sub-task or fix is complete, the tree builds and tests pass, and the change stands on its own as a single commit — pause and ask whether to push. Show a one-line summary of what changed plus a proposed commit message so I can answer fast.
- Don't ask mid-task, on a broken or failing state, or for trivial edits. Never push without my confirmation.
- Push to the working branch (e.g. fix/sprint-N), never straight to main. main only updates through a PR.

## Project state

The repo is the source of truth for what's done and pending, because sessions start cold and run across machines and tools.

- At the start of every session (after the Cloud sync fetch): read `PROGRESS.md`, then check the recent git log and any open PRs. Reconcile them — if the log or PRs show work the file doesn't reflect, update the file. Never start work without doing this.
- Before ending any session or asking to push: update `PROGRESS.md` — move finished items to done, add a one-line Work log entry, set Next up — and stage it with the code so it travels in the same commit. Never end a session with unpushed commits or a stale ledger.
- Git is authoritative for "done": merged to main = done; an open PR or working branch = in flight. `PROGRESS.md` holds the "why" and "what's next" git can't express.

## Tool-specific notes

- `.claude/commands/` holds Claude Code slash commands (spec-agent, status, place-files); other agents can ignore that directory.
- CLAUDE.md exists only as an import shim for Claude Code plus Claude-specific notes.
- Context: this project doubles as the first real product run through the PM OS orchestrator (QA pass on the shape-reversion bug is the planned first sprint).

## Cross-agent handoff

This repository is worked on by multiple coding agents (Claude Code in the cloud, Codex CLI on the laptop, possibly others). GitHub is the sole source of truth; chat history is not project state.

- Begin every new task from the latest `main`; continuing your own in-flight branch is fine after a fetch.
- Read `AGENTS.md` and `PROGRESS.md` before making changes.
- Use a dedicated branch for every task. Never push directly to `main`.
- Update `PROGRESS.md` with completed work, remaining work, and how it was verified.
- Commit and push all intended changes before ending the session.
- Record any unverified behavior or unresolved blocker explicitly in `PROGRESS.md`.
- Do not continue work from another agent's unmerged branch unless explicitly instructed.
- Run the verification suite before pushing: `pytest` (all three suites) — must be green.
