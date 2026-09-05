Look in ~/Downloads/files/ for markdown, svg, and other files from a claude.ai session.

For each file, determine the correct destination based on filename:
- session-handoff.md → ~/pm-os-build/session-handoff.md
- agent-*.md → ~/pm-os-build/agent-specs/
- blog-post-*.md, linkedin-post-*.md, hero-image-*, architecture-diagram-* → ~/pm-os-build/blog/
- eval-notes-*, raw-evals*, benchmark-*, capability-* → ~/pm-os-build/model-eval/
- orchestrator-design.md → ~/pm-os/_system/orchestrator/design.md
- backlog.md → ~/pm-os-build/backlog.md

Show me the proposed file placements before executing. Ask if anything looks wrong. Then copy files and commit both repos with a descriptive message.

If $ARGUMENTS is provided, use that as the source directory instead of ~/Downloads/files/.
