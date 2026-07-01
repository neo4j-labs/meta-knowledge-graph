---
name: mkg-recall
description: Read-only Meta Knowledge Graph memory lookup for Codex. Use to fetch and summarize relevant project-scoped and user-scoped Learning or Decision context from Neo4j before planning, reviewing, or orchestrating work.
---

# MKG Recall

Retrieve relevant memory from the Meta Knowledge Graph and return only the
useful facts. This is the Codex skill equivalent of the Claude Code
`mkg-recall` subagent.

## Workflow

1. Identify whether the request needs project-scoped facts, user-scoped facts, or
   both.
2. Use read-only MKG tools only. Prefer `project_get_context` for ranked recall;
   use `neo4j_read_cypher` for custom read-only queries.
3. Never write to the graph from this skill.
4. Prefer `approved` learnings as policy. Treat `candidate` learnings as hints
   and `Decision` nodes as context, not policy.
5. When querying Neo4j through MCP, wrap temporal values with `toString(...)` so
   date/datetime values do not serialize as `{}`.

## Output

Return a short ranked list of relevant learnings and decisions, including status
and confidence when available, followed by one or two sentences of synthesis. If
nothing relevant exists, say so plainly.
