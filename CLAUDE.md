# Lego AI Builder

AI-powered Lego simulator. Uses Gemini for generation, renders 3D Lego models.

## Repo structure
- `pipeline.py` — main generation pipeline
- `llm.py` — LLM integration (Gemini)
- `render.py` — 3D rendering
- `schema.py` — data models
- `server.py` — web server
- `test_pipeline.py` — tests
- `SPEC.md` — product specification
- `examples/` — sample outputs
- `static/` — web assets
- `screenshots/` — output captures

## Commands
- `python server.py` — start web server
- `python pipeline.py` — run generation pipeline
- `python test_pipeline.py` — run tests
- `pip install -r requirements.txt` — install dependencies

## Known issues
- Can revert to very simple shapes under certain conditions
- Needs deployment to Vercel + app store
- This will be the first real product run through the PM OS system (minor fix cycles)

## Context
This project will be used to test the PM OS orchestrator. The first sprint is a QA pass to identify shape reversion bugs and other issues.

## Cloud sync
This repo moves between my Mac and Claude Code on the web, with GitHub as the source of truth. Keep the remote current.

- At session start: fetch and fast-forward the working branch before doing anything. If it can't fast-forward, stop and tell me — don't force.
- Ask before pushing: when a logical block is done — a sub-task or fix is complete, the tree builds and tests pass, and the change stands on its own as a single commit — pause and ask whether to push. Show a one-line summary of what changed plus a proposed commit message so I can answer fast.
- Don't ask mid-task, on a broken or failing state, or for trivial edits. Never push without my confirmation.
- Push to the working branch (e.g. fix/sprint-N), never straight to main. main only updates through a PR.
