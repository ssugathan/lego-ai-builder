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
