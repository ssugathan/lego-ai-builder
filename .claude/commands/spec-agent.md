Read the agent spec template at ~/pm-os-build/agent-specs/agent-research.md. Use it as the template to define a new agent spec for: $ARGUMENTS

Include:
- Interaction modes (config-driven judgment points, automated vs collaborative)
- Self-configuration logic (if this agent runs before product definition)
- Inputs table (human-provided + orchestrator-provided, including sprint-context.md for non-MVP cycles)
- Outputs table (artifact locations and formats)
- Decision log format
- Sub-tasks table (description, execution type, primary dimensions, model allocation based on scores in ~/pm-os-build/model-eval/capability-scoring.md)
- Judgment points table with default config
- Gate definition
- Notes on anything unique to this agent

Save the spec to ~/pm-os-build/agent-specs/agent-[name].md
