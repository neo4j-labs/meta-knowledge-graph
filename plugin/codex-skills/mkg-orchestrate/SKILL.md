---
name: mkg-orchestrate
description: Run memory-grounded multiagent orchestration with Meta Knowledge Graph in Codex. Use when the user asks to execute a task through MKG, recall prior learnings first, route memory into subagents, verify phases, and capture durable learnings or decisions.
---

# MKG Orchestrate

Mirror the Claude Code `/mkg-orchestrate` command as closely as Codex allows.

Before acting, read the canonical command spec at
`../../commands/mkg-orchestrate.md` and follow it as the source of truth. Treat
this skill file as the Codex adapter layer, not a fork of the workflow.

## Codex Adaptations

- Treat the user's explicit skill invocation text or trailing request as the
  command's `$ARGUMENTS`.
- Start with MKG recall. Prefer the `$mkg-recall` skill when available; otherwise
  call the mounted `project_get_context` or `neo4j_read_cypher` MCP tools
  directly.
- Use Codex subagent or multi-agent tools when they are available. If this
  session does not expose subagent tools, keep the same phase discipline in the
  main session: recall first, plan, implement one phase at a time, verify with
  commands, then capture only durable signal.
- Keep subagents read-only against MKG. The main orchestrator owns any
  `project_add_learning` writes to avoid duplicate or
  conflicting captures.
- When querying Neo4j through MCP, wrap temporal values with `toString(...)` so
  date/datetime values do not serialize as `{}`.
- Preserve the canonical command's anti-pattern checks, verification before
  advancement, and over-capture guardrails.

## Output

Return the memory brief, phase status, verification evidence, and any deliberate
MKG captures. Do not dump raw transcripts or secrets.
