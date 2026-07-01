---
name: sales-agent-demo
description: Set up or verify the MKG RoadFlex sales-agent demo in Codex. Use when the user asks for the sales demo, RoadFlex, demo seeding, Neo4j/BigQuery/Neocarta/Diffbot setup, or the Codex equivalent of the Claude Code sales_agent_demo command.
---

# Sales Agent Demo

Mirror the Claude Code `sales_agent_demo` command as closely as Codex allows.

Before acting, read the canonical command spec at
`../../commands/sales_agent_demo.md` and follow it as the source of truth. Treat
this skill file as the Codex adapter layer, not a fork of the workflow.

## Codex Adaptations

- Treat the user's explicit skill invocation text or trailing request as the
  command's `$ARGUMENTS`.
- Preserve the mutation confirmation guardrail exactly. Before editing env
  files, seeding Neo4j, creating or replacing BigQuery tables, or rebuilding
  Neocarta, summarize targets and commands and wait for explicit approval.
- Prefer `~/.config/meta-knowledge-graph/.env` for installed-plugin runtime
  credentials. From a repo checkout, the repo `.env` can still drive local demo
  seeding.
- To resolve the plugin or checkout root in shell snippets, prefer this Codex
  and Claude compatible sequence:

```bash
ROOT="${CODEX_PLUGIN_ROOT:-${MKG_PLUGIN_ROOT:-${CLAUDE_PLUGIN_ROOT:-}}}"
[ -f "$ROOT/pyproject.toml" ] || ROOT="$(ls -d "${CODEX_HOME:-$HOME/.codex}"/plugins/cache/*/meta-knowledge-graph/*/ 2>/dev/null | sort -V | tail -1)"
[ -f "$ROOT/pyproject.toml" ] || ROOT="$(ls -d "${CLAUDE_CONFIG_DIR:-$HOME/.claude}"/plugins/cache/*/meta-knowledge-graph/*/ 2>/dev/null | sort -V | tail -1)"
[ -f "$ROOT/pyproject.toml" ] || ROOT="$PWD"
```

- Discover actual mounted MCP tool names before verification. Codex may prefix
  plugin MCP tools differently than Claude Code.

## Output

Name what is configured, what is missing, and the exact command you propose to
run next. Never print secret values.
