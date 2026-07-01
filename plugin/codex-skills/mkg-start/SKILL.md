---
name: mkg-start
description: Onboard or check a Meta Knowledge Graph session in Codex. Use when the user asks to start MKG, verify mounted MKG MCP tools, inspect graph state, choose between the RoadFlex sales demo and a custom persona, or run the Codex equivalent of the Claude Code /mkg-start command.
---

# MKG Start

Mirror the Claude Code `/mkg-start` command as closely as Codex allows.

Before acting, read the canonical command spec at `../../commands/mkg-start.md`
and follow it as the source of truth. Treat this skill file as the Codex adapter
layer, not a fork of the workflow.

## Codex Adaptations

- Treat the user's explicit skill invocation text or trailing request as the
  command's `$ARGUMENTS`.
- Discover the actual MCP tool names available in this Codex session before
  using them. The server may appear as `mcp__meta-knowledge-graph__*`,
  `mcp__plugin_meta-knowledge-graph_meta-knowledge-graph__*`, or another
  Codex-normalized prefix.
- When the command says to invoke `meta-knowledge-graph:sales_agent_demo`, use
  the `$sales-agent-demo` skill if it is available; otherwise read
  `../../commands/sales_agent_demo.md` and follow it directly.
- Preserve every mutation guardrail from the canonical command. Do not edit env
  files, seed Neo4j, create or replace BigQuery tables, or rebuild Neocarta
  without explicit user approval.
- If the MCP tools are missing, stop the flow and tell the user which env file
  to configure: `~/.config/meta-knowledge-graph/.env`.

## Output

Report the mounted capability groups, the current graph/persona state, and the
next concrete step. Keep secrets out of the response.
