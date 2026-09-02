---
name: mkg-skill-review
description: Human-in-the-loop skill publishing for the Meta Knowledge Graph in Codex. Use when MKG_SKILL_ACTIVATION=human, when session start reports distilled skill proposals awaiting review, or when the user asks to review, validate, approve, publish, or discard MKG skills before agents can use them.
---

# MKG Skill Review

Mirror the Claude Code `/mkg-skill-review` command as closely as Codex allows.

Before acting, read the canonical command spec at
`../../commands/mkg-skill-review.md` and follow it as the source of truth.
Treat this skill file as the Codex adapter layer, not a fork of the workflow.

## Codex Adaptations

- Treat the user's explicit skill invocation text or trailing request as the
  command's `$ARGUMENTS` (a project id, or one skill slug).
- Discover the actual MCP tool names available in this Codex session before
  using them. The server may appear as `mcp__meta-knowledge-graph__*`,
  `mcp__plugin_meta-knowledge-graph_meta-knowledge-graph__*`, or another
  Codex-normalized prefix. `skill_review_queue` lists the queue;
  `project_resolve_skill` is the only writer you may use to publish, fix, or
  discard a proposal.
- Preserve every guardrail from the canonical command: show each proposal's
  full procedure before a decision, run the validation checklist and report it,
  let the user decide, never resolve an undecided proposal, never edit the
  graph with raw Cypher, and treat any safety finding as a discard.
- Verify each applied decision with `skill_fetch` (published / fixed) or
  `skill_review_queue` (discarded) before reporting it.
- When querying Neo4j through MCP for extra context, wrap temporal values with
  `toString(...)` so date/datetime values do not serialize as `{}`.

## Output

Return the triage list, the per-proposal findings and decisions, the verified
outcomes, and what is still queued. Do not dump raw JSON, transcripts, or
secrets.
